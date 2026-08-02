"""Generate the registered RSS+clamp composition, resumably and without ablations."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import itertools
import json
from pathlib import Path
from typing import Any

import torch
from composition import (
    clamp_cache,
    clean_response,
    make_clamp_runtime,
    rss_coefficients,
)
from safetensors.torch import load_file

ROOT = Path(__file__).parents[2]
LEGACY_RUNNER = ROOT / "experiments" / "style-singleton-screen" / "run.py"
spec = importlib.util.spec_from_file_location("style_screen_runner", LEGACY_RUNNER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {LEGACY_RUNNER}")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--tag", choices=("dev", "final"), required=True)
    parser.add_argument("--subsets", choices=("screen", "all"), required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--expect-prompts", type=int)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def tensor_map(path: Path, sign: float) -> dict[int, torch.Tensor]:
    return {
        int(key.removeprefix("layer_")): value * sign
        for key, value in load_file(path, device="cpu").items()
    }


def load_selection(path: Path) -> tuple[list[dict[str, Any]], float]:
    value = json.loads(path.read_text(encoding="utf-8"))
    features = value.get("features")
    if not isinstance(features, list) or not 4 <= len(features) <= 5:
        raise ValueError("selection must contain four or five features")
    names = [str(item.get("name", "")) for item in features]
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("selection feature names must be unique and non-empty")
    for item in features:
        required = {"name", "direction", "rank", "alpha", "c"}
        if not required <= set(item):
            raise ValueError(f"selection entry misses {sorted(required - set(item))}")
        if item["rank"] not in {"rank1", "rank4", "full"}:
            raise ValueError(f"unsupported rank for {item['name']}")
        if float(item["alpha"]) <= 0 or not 0 < float(item["c"]) <= 1:
            raise ValueError(f"invalid alpha/c for {item['name']}")
        if not Path(item["direction"]).is_file():
            raise FileNotFoundError(item["direction"])
    clamp_beta = float(value.get("clamp_beta", 1.0))
    if not 0 < clamp_beta <= 1:
        raise ValueError("clamp_beta must be in (0, 1]")
    return features, clamp_beta


def prompt_rows(path: Path, limit: int | None) -> list[dict[str, str]]:
    rows = []
    seen = set()
    for index, row in enumerate(read_jsonl(path)):
        prompt = str(row.get("prompt") or row.get("text") or "").strip()
        prompt_id = str(row.get("id", f"prompt-{index:03d}"))
        if not prompt:
            continue
        if prompt_id in seen:
            raise ValueError(f"duplicate prompt id: {prompt_id}")
        seen.add(prompt_id)
        rows.append({"id": prompt_id, "prompt": prompt})
    rows = rows if limit is None else rows[:limit]
    if not rows:
        raise ValueError("no usable prompts")
    return rows


def condition_sets(names: tuple[str, ...], kind: str) -> list[tuple[str, ...]]:
    sizes = range(1, len(names) + 1) if kind == "all" else (1, 2, len(names))
    return list(
        dict.fromkeys(
            subset for size in sizes for subset in itertools.combinations(names, size)
        )
    )


@torch.inference_mode()
def decode(
    model: Any,
    tokenizer: Any,
    cache: Any,
    max_new_tokens: int,
    clamp: tuple[dict[int, dict[str, Any]], float],
) -> tuple[str, list[dict[str, Any]]]:
    output = None
    trace = []
    for token_id in tokenizer.encode("\n", add_special_tokens=False):
        output = model(
            input_ids=torch.tensor([[token_id]], device="cuda"),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = output.past_key_values
        clamp_cache(cache, *clamp)
    if output is None:
        raise RuntimeError("bridge text produced no tokens")
    logits = output.logits[:, -1, :]
    generated: list[int] = []
    eos = tokenizer.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    checkpoints = {0, max_new_tokens // 2, max_new_tokens - 1}
    for step in range(max_new_tokens):
        token_id = int(logits.argmax(dim=-1).item())
        if token_id in eos_ids:
            break
        generated.append(token_id)
        output = model(
            input_ids=torch.tensor([[token_id]], device="cuda"),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = output.past_key_values
        observed = clamp_cache(cache, *clamp)
        if step in checkpoints:
            trace.append({"token": step + 1, "coefficients": observed})
        logits = output.logits[:, -1, :]
    return clean_response(tokenizer.decode(generated, skip_special_tokens=True)), trace


def generate_condition(
    model: Any,
    tokenizer: Any,
    base_cache: Any,
    directions: dict[str, dict[int, torch.Tensor]],
    selected: list[dict[str, Any]],
    max_new_tokens: int,
    clamp_beta: float,
) -> tuple[str, list[dict[str, Any]]]:
    cache = runner.clone_cache(base_cache)
    before = runner.snapshot_nonrecurrent(cache)
    coefficients = {
        item["name"]: float(item["alpha"]) * float(item["c"]) for item in selected
    }
    active = {item["name"]: directions[item["name"]] for item in selected}
    deltas = rss_coefficients(active, coefficients)
    runtime = make_clamp_runtime(cache, active, deltas)
    clamp_cache(cache, runtime, 1.0)
    runner.assert_nonrecurrent_unchanged(before, cache)
    return decode(model, tokenizer, cache, max_new_tokens, (runtime, clamp_beta))


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected, clamp_beta = load_selection(args.selection)
    rows = prompt_rows(args.prompts, args.limit)
    if args.expect_prompts is not None and len(rows) != args.expect_prompts:
        raise ValueError(f"expected {args.expect_prompts} prompts, got {len(rows)}")
    names = tuple(str(item["name"]) for item in selected)
    conditions = condition_sets(names, args.subsets)
    manifest = {
        "model": args.model,
        "seed": args.seed,
        "prompts": str(args.prompts),
        "selection": selected,
        "subsets": args.subsets,
        "clamp_beta": clamp_beta,
        "conditions": ["+".join(value) for value in conditions],
        "method": "gdn_rss_clamp",
    }
    manifest_path = args.output_dir / f"{args.tag}-selection.json"
    serialized = json.dumps(manifest, ensure_ascii=False, indent=2)
    if (
        manifest_path.exists()
        and manifest_path.read_text(encoding="utf-8") != serialized
    ):
        raise RuntimeError("selection differs from the existing resumable run")
    manifest_path.write_text(
        serialized,
        encoding="utf-8",
    )
    output = args.output_dir / f"{args.tag}-generations.jsonl"
    done = {row["task_id"] for row in read_jsonl(output)} if output.exists() else set()
    directions = {
        item["name"]: tensor_map(Path(item["direction"]), float(item.get("sign", 1)))
        for item in selected
    }
    torch.manual_seed(args.seed)
    tokenizer, model = runner.load_model(args.model)
    for index, row in enumerate(rows, 1):
        base_cache = runner.prefill(model, tokenizer, row["prompt"])
        before = runner.snapshot_nonrecurrent(base_cache)
        for subset in ((), *conditions):
            label = "baseline" if not subset else "+".join(subset)
            task_id = f"{args.tag}:{row['id']}:{label}"
            if task_id in done:
                continue
            if subset:
                chosen = [item for item in selected if item["name"] in subset]
                response, trace = generate_condition(
                    model,
                    tokenizer,
                    base_cache,
                    directions,
                    chosen,
                    args.max_new_tokens,
                    clamp_beta,
                )
            else:
                cache = runner.clone_cache(base_cache)
                response = clean_response(
                    runner.decode(model, tokenizer, cache, args.max_new_tokens)
                )
                trace = []
            runner.assert_nonrecurrent_unchanged(before, base_cache)
            append_jsonl(
                output,
                {
                    "task_id": task_id,
                    "prompt_id": row["id"],
                    "scenario": row["prompt"],
                    "condition": label,
                    "active_features": list(subset),
                    "response": response,
                    "generation": {
                        "method": "gdn_rss_clamp" if subset else "baseline",
                        "coefficient_trace": trace,
                    },
                },
            )
            print(f"{args.tag}: {index}/{len(rows)} {label}", flush=True)
        del base_cache
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run(arguments())
