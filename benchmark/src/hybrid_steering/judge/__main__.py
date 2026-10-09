"""Judge a saved experiment or any JSONL of prompt/response pairs."""

import argparse
import fcntl
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .concept_only import request_label, validate
from .core import (
    ROOT,
    digest,
    fingerprint,
    now,
    read_jsonl,
    render,
    task_id,
    write,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results", type=Path, help="Experiment folder with manifest.json and answers.jsonl")
    source.add_argument("--input", type=Path, help="Any JSONL with prompt and response fields")
    parser.add_argument("--concept", help="Concept guide name; required with --input")
    parser.add_argument("--output", type=Path, help="Output folder for --input (default: results/<concept>/judge-only/<input hash>)")
    parser.add_argument("--settings", type=Path, default=ROOT / "config/judge.yaml")
    parser.add_argument("--data-root", type=Path, help="Host-specific data root; Judge writes under its results/")
    parser.add_argument("--key-file", type=Path, help="Private OpenRouter key file; overrides OPENROUTER_API_KEY")
    parser.add_argument("--run", action="store_true", help="Send paid API requests; otherwise prepare only")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 32:
        parser.error("--workers must be between 1 and 32")
    if args.input and not args.concept:
        parser.error("--concept is required with --input")
    if args.results and (args.concept or args.output):
        parser.error("--concept and --output apply only to --input")
    return args


def prepare_tasks(rows: list[dict], protocol: dict) -> tuple[list[dict], list[dict]]:
    """Deduplicate identical texts while retaining each input-row binding."""
    tasks = {}
    bindings = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not all(isinstance(row.get(name), str) for name in ("prompt", "response")):
            raise ValueError(f"Input row {index + 1} needs string prompt and response")
        anonymous_id = task_id(row, protocol)
        tasks[anonymous_id] = {
            "anonymous_id": anonymous_id,
            "prompt": row["prompt"],
            "response": row["response"],
        }
        bindings.append({
            "anonymous_id": anonymous_id,
            "method": row.get("method"),
            "layer": row.get("layer"),
            "strength": row.get("strength"),
            "category": row.get("category"),
            "suite": row.get("suite"),
            "key": row.get("key", row.get("id", index)),
        })
    return list(tasks.values()), bindings


def load_labels(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    saved = read_jsonl(path)
    labels = {row["anonymous_id"]: row for row in saved}
    assert len(labels) == len(saved), "Duplicate judgment IDs"
    for label in saved:
        if any(key in label for key in ("evaluable", "content_quality", "flags", "reason")):
            raise ValueError("Old Judge protocol labels cannot be mixed with v4")
        try:
            current = {key: label[key] for key in (
                "concept_score", "probability_source", "p0", "p1", "p2", "p3", "p4")}
        except KeyError as error:
            raise ValueError("Incomplete v4 Judge label") from error
        validate(current)
    return labels


def annotate(
    pending: list[dict], api: dict, system: str, output: Path,
    workers: int, labels: dict[str, dict],
) -> None:
    if api.get("api_key_file"):
        key_path = Path(api["api_key_file"])
        if key_path.stat().st_mode & 0o077:
            raise ValueError("API key file must be private (mode 600)")
        secret = key_path.read_text().strip()
    else:
        secret = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not secret or "\n" in secret:
        raise ValueError("Set OPENROUTER_API_KEY or pass --key-file")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(request_label, api, system, row, secret, output): row["anonymous_id"]
            for row in pending
        }
        for future in as_completed(futures):
            anonymous_id = futures[future]
            try:
                label = future.result()
            except Exception as error:
                print(f"Missing {anonymous_id}: {type(error).__name__}", flush=True)
                continue
            with (output / "labels.jsonl").open("a") as stream:
                stream.write(json.dumps(label, ensure_ascii=False) + "\n")
                stream.flush()
            labels[label["anonymous_id"]] = label


def source_paths(args: argparse.Namespace, concept: str) -> tuple[Path, Path]:
    results_root = (args.data_root.resolve() / "results") if args.data_root else ROOT / "data/results"
    if args.results:
        folder = args.results.resolve()
        source = folder / "answers.jsonl"
    else:
        source = args.input.resolve()
        folder = (args.output or results_root / concept / "judge-only" / digest(source)[:12]).resolve()
    if not folder.is_relative_to(results_root):
        raise ValueError(f"Judge output must stay under {results_root}")
    return source, folder


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.results:
        experiment = json.loads((args.results / "manifest.json").read_text())["identity"]
        concept = experiment["concept"]
    else:
        concept = args.concept
    system = render(concept, "judge_v4.md")
    source, folder = source_paths(args, concept)
    api = json.loads(args.settings.read_text())
    if args.key_file:
        api["api_key_file"] = str(args.key_file.resolve())
    protocol = {
        "version": ("common-v6-structured-token-logprobs" if api.get("score_response_format")
                    else "common-v4-concept-only"),
        "concept": concept,
        "system": system,
        "api": {key: value for key, value in api.items() if key != "api_key_file"},
        "transport_sha256": {
            name: digest(ROOT / "src/hybrid_steering/judge" / name)
            for name in ("openrouter.py", "concept_only.py")
        },
    }
    output = folder / "judge" / api["model"].replace("/", "--") / fingerprint(protocol)[:12]
    output.mkdir(parents=True, exist_ok=True)

    with (output / "annotation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            assert json.loads(manifest_path.read_text())["identity"] == protocol
        else:
            write(manifest_path, {"identity": protocol, "created_at": now()})

        rows = read_jsonl(source)
        tasks, bindings = prepare_tasks(rows, protocol)
        labels = load_labels(output / "labels.jsonl")
        write(output / "bindings.jsonl", bindings)
        write(output / "input-snapshot.json", {
            "source": str(source),
            "answers_sha256": digest(source),
            "n": len(rows),
            "at": now(),
        })

        pending = [task for task in tasks if task["anonymous_id"] not in labels]
        if args.run and pending:
            annotate(pending, api, system, output, args.workers, labels)

        completed = sum(task["anonymous_id"] in labels for task in tasks)
        progress = {
            "expected": len(tasks),
            "completed": completed,
            "answer_rows": len(rows),
            "complete": completed == len(tasks),
            "at": now(),
        }
        write(output / "progress.json", progress)
        print(json.dumps({"output": str(output), **progress}))
        if args.run and not progress["complete"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
