"""Run the cheap GPU smoke test for repaired candidate features."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).parents[2]
LEGACY_RUNNER = ROOT / "experiments" / "style-singleton-screen" / "run.py"
spec = importlib.util.spec_from_file_location("style_screen_runner", LEGACY_RUNNER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {LEGACY_RUNNER}")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

FEATURES = ("humorous", "numbered_list", "technical", "persuasive")
PROMPTS = [
    "How should a team decide whether to postpone a software release?",
    "Explain how a household can reduce its electricity use.",
    "What should a student do when two sources disagree?",
    "How can a small shop improve its inventory planning?",
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("run", "summary"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--feature", choices=FEATURES)
    parser.add_argument("--rank", choices=("full", "rank1", "rank4"), default="rank4")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def tensor_map(path: Path) -> dict[int, torch.Tensor]:
    return {
        int(key.removeprefix("layer_")): value
        for key, value in load_file(path, device="cpu").items()
    }


def build_directions(out: Path, model: Any, tokenizer: Any) -> None:
    direction_dir = out / "directions"
    direction_dir.mkdir(parents=True, exist_ok=True)
    for feature in FEATURES:
        full_path = direction_dir / f"{feature}-full.safetensors"
        if full_path.exists():
            full = tensor_map(full_path)
        else:
            payload = json.loads(
                (out / "data" / f"{feature}_pairs.json").read_text(encoding="utf-8")
            )
            diffs = []
            for index, pair in enumerate(payload["pairs"]):
                positive = runner.extract_recurrent(
                    runner.prefill(model, tokenizer, pair["positive_text"])
                )
                negative = runner.extract_recurrent(
                    runner.prefill(model, tokenizer, pair["negative_text"])
                )
                diffs.append(runner.subtract_states(positive, negative))
                if (index + 1) % 16 == 0:
                    print(
                        f"direction {feature}: {index + 1}/{len(payload['pairs'])}",
                        flush=True,
                    )
            full = runner.mean_direction(diffs)
            save_file(
                {f"layer_{layer}": value.contiguous() for layer, value in full.items()},
                str(full_path),
            )
        for rank in (1, 4):
            rank_path = direction_dir / f"{feature}-rank{rank}.safetensors"
            if not rank_path.exists():
                low_rank = runner.low_rank(full, rank)
                save_file(
                    {
                        f"layer_{layer}": value.contiguous()
                        for layer, value in low_rank.items()
                    },
                    str(rank_path),
                )


def run(
    out: Path,
    model_id: str,
    feature: str | None,
    rank: str,
    alpha: float,
    max_new_tokens: int,
) -> None:
    if feature is None:
        raise ValueError("--feature is required for the sequential smoke")
    tokenizer, model = runner.load_model(model_id)
    build_directions(out, model, tokenizer)
    output = out / f"smoke-{feature}-{rank}-alpha={alpha:g}.jsonl"
    done = {row["task_id"] for row in read_jsonl(output)}
    direction = tensor_map(out / "directions" / f"{feature}-{rank}.safetensors")
    for prompt_id, prompt in enumerate(PROMPTS):
        target = runner.prefill(model, tokenizer, prompt)
        before = runner.snapshot_nonrecurrent(target)
        conditions: list[tuple[str, dict[int, torch.Tensor] | None, float]] = [
            ("baseline", None, 0.0),
            (f"{feature}:{rank}:alpha={alpha:g}", direction, alpha),
        ]
        for condition, direction, strength in conditions:
            task_id = f"smoke-{prompt_id:02d}:{condition}"
            if task_id in done:
                continue
            cache = runner.clone_cache(target)
            if direction is not None:
                runner.add_direction(cache, direction, strength)
            runner.assert_nonrecurrent_unchanged(before, cache)
            response = runner.decode(model, tokenizer, cache, max_new_tokens)
            append_jsonl(
                output,
                {
                    "task_id": task_id,
                    "prompt_id": f"smoke-{prompt_id:02d}",
                    "scenario": prompt,
                    "condition": condition,
                    "response": response,
                },
            )
            print(task_id, flush=True)
        del target
        gc.collect()
        torch.cuda.empty_cache()


def has_loop(text: str) -> bool:
    words = re.findall(r"\w+", text.lower())
    chunks = [
        tuple(words[index : index + 6]) for index in range(max(0, len(words) - 5))
    ]
    return len(chunks) != len(set(chunks))


def summarize(out: Path, feature: str | None, rank: str, alpha: float) -> None:
    if feature is None:
        raise ValueError("--feature is required for the sequential smoke")
    rows = read_jsonl(out / f"smoke-{feature}-{rank}-alpha={alpha:g}.jsonl")
    if not rows:
        raise RuntimeError("no smoke generations")
    report = []
    for row in rows:
        text = row["response"]
        report.append(
            {
                **row,
                "numbered_items": len(re.findall(r"(?m)^\s*\d+[.)]\s+", text)),
                "has_cjk": bool(re.search(r"[\u3400-\u9fff]", text)),
                "has_loop": has_loop(text),
                "word_count": len(text.split()),
            }
        )
    write_json(
        out / "smoke_summary.json",
        {
            "complete": len(rows) == len(PROMPTS) * 2,
            "n": len(rows),
            "rows": report,
        },
    )


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "run":
        run(
            args.output_dir,
            args.model,
            args.feature,
            args.rank,
            args.alpha,
            args.max_new_tokens,
        )
    else:
        summarize(args.output_dir, args.feature, args.rank, args.alpha)


if __name__ == "__main__":
    main()
