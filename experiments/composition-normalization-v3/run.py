"""Experiment #3: composition normalization, method, and rank ablations."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).parents[2]
FEATURES = ("joy", "concrete", "optimism", "candor")
RUBRICS = {
    "joy": "joy",
    "concrete": "concrete_language",
    "optimism": "optimism",
    "candor": "principled_candor",
}
ALPHAS = (0.5, 1.0, 2.0, 4.0, 8.0)
ACTIVATION_LAYER = 10
METADATA = {"task_id", "source_id", "scenario", "_generation"}


def module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


BASE = module("exp3_base", ROOT / "experiments/four-axis-night/run.py")
HELPERS = module(
    "exp3_helpers", ROOT / "experiments/composition-generation-queue/run.py"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=(
            "self-test",
            "smoke",
            "dev",
            "prepare-dev-judge",
            "select",
            "gdn",
            "activation",
            "prepare-main-judge",
            "summarize",
        ),
    )
    parser.add_argument("--base-directions-dir", type=Path)
    parser.add_argument("--optimism-direction", type=Path)
    parser.add_argument("--joy-direction", type=Path)
    parser.add_argument("--cached-ranks-dir", type=Path)
    parser.add_argument("--activation-directions", type=Path)
    parser.add_argument("--candor-pairs", type=Path)
    parser.add_argument("--concrete-pairs", type=Path)
    parser.add_argument("--optimism-pairs", type=Path)
    parser.add_argument("--joy-pairs", type=Path)
    parser.add_argument("--dev-prompts", type=Path)
    parser.add_argument("--test-prompts", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--quality-test", type=int, default=32)
    return parser.parse_args()


def load_direction(path: Path) -> dict[int, torch.Tensor]:
    return {
        int(name.removeprefix("layer_")): value.float()
        for name, value in load_file(path, device="cpu").items()
    }


def full_directions(args: argparse.Namespace) -> dict[str, dict[int, torch.Tensor]]:
    return {
        "joy": load_direction(args.joy_direction),
        "concrete": load_direction(args.base_directions_dir / "concrete.safetensors"),
        "optimism": load_direction(args.optimism_direction),
        "candor": load_direction(args.base_directions_dir / "candor.safetensors"),
    }


def ranked_directions(
    args: argparse.Namespace, rank: int
) -> dict[str, dict[int, torch.Tensor]]:
    result = {}
    full = full_directions(args)
    cache = args.output_dir / "directions"
    cache.mkdir(parents=True, exist_ok=True)
    for name in FEATURES:
        existing = args.cached_ranks_dir / f"{name}-rank{rank}.safetensors"
        path = (
            existing if existing.exists() else cache / f"{name}-rank{rank}.safetensors"
        )
        if not path.exists():
            HELPERS.save_direction(path, HELPERS.truncate_direction(full[name], rank))
        result[name] = load_direction(path)
    return result


def donor_rows(args: argparse.Namespace, feature: str) -> list[dict]:
    path = {
        "joy": args.joy_pairs,
        "concrete": args.concrete_pairs,
        "optimism": args.optimism_pairs,
        "candor": args.candor_pairs,
    }[feature]
    rows = BASE.jsonl(path)[:64]
    if len(rows) != 64:
        raise RuntimeError(f"{feature}: need 64 donor pairs")
    if feature == "candor":
        rows = [
            row
            | {
                "positive_text": row["candor_text"],
                "negative_text": row["sycophancy_text"],
            }
            for row in rows
        ]
    return rows


def activation_directions(
    args: argparse.Namespace, model: Any, tokenizer: Any
) -> dict[str, torch.Tensor]:
    cache = args.output_dir / "directions" / "activation-layer10.safetensors"
    if cache.exists():
        return {
            name: value.float()
            for name, value in load_file(cache, device="cpu").items()
        }
    existing = load_file(args.activation_directions, device="cpu")
    result = {
        name: existing[f"{name}_layer_{ACTIVATION_LAYER}"].float()
        for name in ("concrete", "optimism", "candor")
    }
    rows = donor_rows(args, "joy")
    total = None
    for index, row in enumerate(rows, 1):
        negative = HELPERS.residual_for_text(
            model, tokenizer, row["negative_text"], (ACTIVATION_LAYER,)
        )[ACTIVATION_LAYER]
        positive = HELPERS.residual_for_text(
            model, tokenizer, row["positive_text"], (ACTIVATION_LAYER,)
        )[ACTIVATION_LAYER]
        delta = positive - negative
        total = delta if total is None else total + delta
        if index % 8 == 0:
            print(f"activation direction joy: {index}/64", flush=True)
    result["joy"] = (total / len(rows)).squeeze(0).cpu()
    cache.parent.mkdir(parents=True, exist_ok=True)
    save_file({name: value.contiguous() for name, value in result.items()}, str(cache))
    return result


def rss_direction(
    directions: dict[str, dict[int, torch.Tensor]],
    strengths: dict[str, float],
    active: tuple[str, ...],
) -> dict[int, torch.Tensor]:
    return BASE.norm_controlled(directions, strengths, list(active))


def rss_vector(
    directions: dict[str, torch.Tensor],
    strengths: dict[str, float],
    active: tuple[str, ...],
) -> torch.Tensor:
    pieces = [directions[name] * strengths[name] for name in active]
    raw = sum(pieces[1:], pieces[0].clone())
    target = math.sqrt(sum(float(piece.square().sum()) for piece in pieces))
    return raw * (target / max(float(raw.norm()), 1e-12))


@contextmanager
def activation_hook(model: Any, direction: torch.Tensor):
    vector = direction.to("cuda", dtype=torch.float16)

    def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> torch.Tensor:
        return output + vector.to(output).view(1, 1, -1)

    handle = HELPERS.decoder_layers(model)[ACTIVATION_LAYER].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.inference_mode()
def activation_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    direction: torch.Tensor,
    max_new_tokens: int,
) -> str:
    with activation_hook(model, direction):
        cache = BASE.prefill(model, tokenizer, prompt)
        text, _ = BASE.decode(model, tokenizer, cache, max_new_tokens)
    return text


def prompt_rows(path: Path) -> list[dict]:
    rows = BASE.jsonl(path)
    if not rows:
        raise RuntimeError(f"no prompts in {path}")
    return rows


def append(path: Path, row: dict) -> None:
    BASE.append_jsonl(path, row)


def dev_phase(args: argparse.Namespace) -> None:
    rank1 = ranked_directions(args, 1)
    tokenizer, model = BASE.load_model(args.model)
    activation = activation_directions(args, model, tokenizer)
    path = args.output_dir / "dev-generations.jsonl"
    done = {row["task_id"] for row in BASE.jsonl(path)} if path.exists() else set()
    gdn_conditions = {
        f"gdn_{feature}_a{alpha:g}": [(rank1[feature], alpha, None)]
        for feature in FEATURES
        for alpha in ALPHAS
    }
    for index, row in enumerate(prompt_rows(args.dev_prompts), 1):
        task_id = f"exp3:dev:{row['id']}"
        if task_id in done:
            continue
        prompt = BASE.row_prompt(row)
        texts, metadata = BASE.generate(
            model,
            tokenizer,
            prompt,
            {"baseline": [], **gdn_conditions},
            args.max_new_tokens,
        )
        for feature in FEATURES:
            for alpha in ALPHAS:
                texts[f"act_{feature}_a{alpha:g}"] = activation_generate(
                    model,
                    tokenizer,
                    prompt,
                    activation[feature] * alpha,
                    args.max_new_tokens,
                )
        append(
            path,
            {
                "task_id": task_id,
                "source_id": row["id"],
                "scenario": prompt,
                **texts,
                "_generation": metadata,
            },
        )
        print(f"dev: {index}/{len(prompt_rows(args.dev_prompts))}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()


def smoke_phase(args: argparse.Namespace) -> None:
    rank1 = ranked_directions(args, 1)
    tokenizer, model = BASE.load_model(args.model)
    activation = activation_directions(args, model, tokenizer)
    row = prompt_rows(args.dev_prompts)[0]
    prompt = BASE.row_prompt(row)
    texts, _ = BASE.generate(
        model,
        tokenizer,
        prompt,
        {"baseline": [], "gdn_joy": [(rank1["joy"], 1.0, None)]},
        24,
    )
    texts["activation_joy"] = activation_generate(
        model, tokenizer, prompt, activation["joy"], 24
    )
    if any(not text for text in texts.values()):
        raise RuntimeError("smoke generation returned an empty answer")
    BASE.atomic_json(args.output_dir / "smoke.json", texts)
    print("Experiment #3 GPU smoke passed")


def read_generation_rows(args: argparse.Namespace, split: str) -> list[dict]:
    if split == "dev":
        return BASE.jsonl(args.output_dir / "dev-generations.jsonl")
    merged = {}
    for name in ("gdn-generations.jsonl", "activation-generations.jsonl"):
        for row in BASE.jsonl(args.output_dir / name):
            target = merged.setdefault(
                row["source_id"],
                {"source_id": row["source_id"], "scenario": row["scenario"]},
            )
            target.update(
                {key: value for key, value in row.items() if key not in METADATA}
            )
    return list(merged.values())


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def prepare_judge(args: argparse.Namespace, split: str) -> None:
    rows = read_generation_rows(args, split)
    root = args.output_dir / "judge" / split / "inputs"
    for feature in FEATURES:
        prepared = []
        for row in rows:
            answers = [
                {"answer_id": name, "text": text}
                for name, text in row.items()
                if name not in {"source_id", "scenario"}
                and (split == "main" or name == "baseline" or f"_{feature}_" in name)
            ]
            prepared.append(
                {
                    "prompt_id": row["source_id"],
                    "scenario": row["scenario"],
                    "answers": answers,
                    "metadata": {"experiment": "composition-normalization-v3"},
                }
            )
        write_jsonl(root / f"{RUBRICS[feature]}.jsonl", prepared)
    write_jsonl(
        root / "quality.jsonl",
        [
            {
                "prompt_id": row["source_id"],
                "scenario": row["scenario"],
                "answers": [
                    {"answer_id": name, "text": text}
                    for name, text in row.items()
                    if name not in {"source_id", "scenario"}
                ],
                "metadata": {"experiment": "composition-normalization-v3"},
            }
            for row in rows[: args.quality_test]
        ],
    )
    print(f"prepared Judge inputs: split={split} prompts={len(rows)}")


def result_rows(path: Path) -> list[dict]:
    return BASE.jsonl(path)


def select_phase(args: argparse.Namespace) -> None:
    root = args.output_dir / "judge" / "dev" / "results"
    trait = {}
    for feature in FEATURES:
        grouped = defaultdict(list)
        for row in result_rows(root / f"{RUBRICS[feature]}.jsonl"):
            grouped[row["answer_id"]].append(
                row["score_distribution"]["expected_score"]
            )
        trait[feature] = {
            answer: sum(values) / len(values) for answer, values in grouped.items()
        }
    quality_grouped = defaultdict(list)
    for row in result_rows(root / "quality.jsonl"):
        quality_grouped[row["answer_id"]].append(
            min(row["task_fulfillment"], row["coherence"])
        )
    quality = {
        answer: sum(values) / len(values) for answer, values in quality_grouped.items()
    }
    baseline_quality = quality["baseline"]
    selection = {}
    for feature in FEATURES:
        safe_by_method = {}
        for method in ("gdn", "act"):
            candidates = [
                {
                    "alpha": alpha,
                    "trait": trait[feature][f"{method}_{feature}_a{alpha:g}"],
                    "quality": quality[f"{method}_{feature}_a{alpha:g}"],
                }
                for alpha in ALPHAS
            ]
            safe = [
                item
                for item in candidates
                if item["quality"] >= baseline_quality - 0.25
            ]
            safe_by_method[method] = safe or candidates
        target = min(
            max(item["trait"] for item in safe_by_method[method])
            for method in ("gdn", "act")
        )
        selection[feature] = {"matched_target": target}
        for method in ("gdn", "act"):
            eligible = [
                item
                for item in safe_by_method[method]
                if item["trait"] >= target - 0.05
            ]
            chosen = min(
                eligible or safe_by_method[method],
                key=lambda item: (
                    item["alpha"] if eligible else -item["trait"],
                    item["alpha"],
                ),
            )
            selection[feature][method] = {
                "selected_alpha": chosen["alpha"],
                "candidates": safe_by_method[method],
            }
    BASE.atomic_json(args.output_dir / "selection.json", selection)
    print(json.dumps(selection, indent=2))


def strengths(args: argparse.Namespace, method: str) -> dict[str, float]:
    selected = json.loads((args.output_dir / "selection.json").read_text())
    return {
        feature: float(selected[feature][method]["selected_alpha"])
        for feature in FEATURES
    }


def active(mask: int) -> tuple[str, ...]:
    return tuple(
        feature for index, feature in enumerate(FEATURES) if mask & (1 << index)
    )


def gdn_phase(args: argparse.Namespace) -> None:
    rank1, rank4 = ranked_directions(args, 1), ranked_directions(args, 4)
    selected = strengths(args, "gdn")
    conditions: dict[str, list] = {"baseline": []}
    for mask in range(1, 16):
        names = active(mask)
        conditions[f"gdn_raw_r1_{mask:04b}"] = [
            (rank1[name], selected[name], None) for name in names
        ]
        if len(names) >= 2:
            conditions[f"gdn_rss_r1_{mask:04b}"] = [
                (rss_direction(rank1, selected, names), 1.0, None)
            ]
            conditions[f"gdn_rss_r4_{mask:04b}"] = [
                (rss_direction(rank4, selected, names), 1.0, None)
            ]
    tokenizer, model = BASE.load_model(args.model)
    BASE.run_records(
        model=model,
        tokenizer=tokenizer,
        rows=prompt_rows(args.test_prompts),
        conditions=conditions,
        path=args.output_dir / "gdn-generations.jsonl",
        max_new_tokens=args.max_new_tokens,
        prefix="exp3:gdn",
    )


def activation_phase(args: argparse.Namespace) -> None:
    tokenizer, model = BASE.load_model(args.model)
    directions = activation_directions(args, model, tokenizer)
    selected = strengths(args, "act")
    path = args.output_dir / "activation-generations.jsonl"
    done = {row["task_id"] for row in BASE.jsonl(path)} if path.exists() else set()
    rows = prompt_rows(args.test_prompts)
    for index, row in enumerate(rows, 1):
        task_id = f"exp3:activation:{row['id']}"
        if task_id in done:
            continue
        prompt = BASE.row_prompt(row)
        outputs = {}
        for mask in range(1, 16):
            names = active(mask)
            raw = sum(
                (directions[name] * selected[name] for name in names[1:]),
                directions[names[0]] * selected[names[0]],
            )
            outputs[f"act_raw_{mask:04b}"] = activation_generate(
                model, tokenizer, prompt, raw, args.max_new_tokens
            )
            if len(names) >= 2:
                outputs[f"act_rss_{mask:04b}"] = activation_generate(
                    model,
                    tokenizer,
                    prompt,
                    rss_vector(directions, selected, names),
                    args.max_new_tokens,
                )
        append(
            path,
            {
                "task_id": task_id,
                "source_id": row["id"],
                "scenario": prompt,
                **outputs,
            },
        )
        print(f"activation: {index}/{len(rows)}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()


def condition_features(answer_id: str) -> tuple[str, ...]:
    if answer_id == "baseline":
        return FEATURES
    return active(int(answer_id.rsplit("_", 1)[-1], 2))


def summarize_phase(args: argparse.Namespace) -> None:
    root = args.output_dir / "judge" / "main" / "results"
    scores = defaultdict(dict)
    costs = {"input_tokens": 0, "output_tokens": 0}
    for feature in FEATURES:
        for row in result_rows(root / f"{RUBRICS[feature]}.jsonl"):
            scores[(row["prompt_id"], row["answer_id"])][feature] = {
                "hard": row["trait_score"],
                "soft": row["score_distribution"]["expected_score"],
            }
            usage = row["provenance"]["usage"]
            costs["input_tokens"] += usage["input_tokens"]
            costs["output_tokens"] += usage["output_tokens"]
    quality = defaultdict(list)
    for row in result_rows(root / "quality.jsonl"):
        quality[row["answer_id"]].append(min(row["task_fulfillment"], row["coherence"]))
        usage = row["provenance"]["usage"]
        costs["input_tokens"] += usage["input_tokens"]
        costs["output_tokens"] += usage["output_tokens"]
    by_answer = defaultdict(list)
    for (prompt_id, answer_id), values in scores.items():
        if set(values) == set(FEATURES):
            by_answer[answer_id].append((prompt_id, values))
    conditions = []
    for answer_id, rows in sorted(by_answer.items()):
        enabled = condition_features(answer_id)
        minimums = [min(values[name]["soft"] for name in enabled) for _, values in rows]
        hard = [
            all(values[name]["hard"] >= 4 for name in enabled) for _, values in rows
        ]
        inactive = [name for name in FEATURES if name not in enabled]
        conditions.append(
            {
                "condition": answer_id,
                "active_features": list(enabled),
                "n": len(rows),
                "mean_minimum_expected": sum(minimums) / len(minimums),
                "all_active_ge4": sum(hard) / len(hard),
                "feature_expected_means": {
                    name: sum(values[name]["soft"] for _, values in rows) / len(rows)
                    for name in FEATURES
                },
                "inactive_expected_mean": (
                    sum(values[name]["soft"] for _, values in rows for name in inactive)
                    / (len(rows) * len(inactive))
                    if inactive
                    else None
                ),
                "quality_mean": (
                    sum(quality[answer_id]) / len(quality[answer_id])
                    if quality[answer_id]
                    else None
                ),
            }
        )
    costs["estimated_usd"] = (
        costs["input_tokens"] * 0.15 + costs["output_tokens"] * 0.60
    ) / 1_000_000
    BASE.atomic_json(
        args.output_dir / "summary.json",
        {"conditions": conditions, "judge_usage": costs},
    )
    print(f"conditions={len(conditions)} estimated_cost=${costs['estimated_usd']:.3f}")


def self_test() -> None:
    toy = {name: {0: torch.eye(3).reshape(1, 1, 3, 3)} for name in FEATURES}
    selected = {name: 1.0 for name in FEATURES}
    assert active(15) == FEATURES
    assert len(active(7)) == 3
    assert rss_direction(toy, selected, FEATURES)[0].shape == (1, 1, 3, 3)
    vectors = {name: torch.ones(3) for name in FEATURES}
    assert rss_vector(vectors, selected, FEATURES).shape == (3,)
    print("Experiment #3 self-test passed")


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    {
        "self-test": lambda _args: self_test(),
        "smoke": smoke_phase,
        "dev": dev_phase,
        "prepare-dev-judge": lambda value: prepare_judge(value, "dev"),
        "select": select_phase,
        "gdn": gdn_phase,
        "activation": activation_phase,
        "prepare-main-judge": lambda value: prepare_judge(value, "main"),
        "summarize": summarize_phase,
    }[args.phase](args)


if __name__ == "__main__":
    main()
