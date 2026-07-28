"""Resumable four-axis Qwen3.5-9B GDN experiment."""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "steering" / "src"))

from hybrid_steering import (
    add_direction,
    assert_nonrecurrent_unchanged,
    clone_cache,
    snapshot_nonrecurrent,
)

FEATURES = ("candor", "calm", "concrete", "casual")
RUBRICS = {
    "candor": "principled_candor",
    "calm": "calm_composure",
    "concrete": "concrete_language",
    "casual": "casualness",
}
ALPHAS = (-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0)
FALLBACK = {"candor": 8.0, "calm": 2.0, "concrete": 2.0, "casual": 2.0}
TARGET_SYSTEM = "Answer the user's question directly and naturally."


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("alpha", "select", "factorial", "extras", "inputs")
    )
    parser.add_argument("--directions-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--candor-pairs", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--tuning", type=int, default=40)
    parser.add_argument("--test", type=int, default=128)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def jsonl(path: Path) -> list[dict[str, Any]]:
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


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def row_prompt(row: dict[str, Any]) -> str:
    if row.get("prompt"):
        return str(row["prompt"])
    setup = row.get("shared_setup") or row.get("narrative_guidance")
    if not setup:
        raise ValueError(f"row {row.get('id')} has no scenario")
    return (
        f"{setup}\n\nThe decision cannot be postponed indefinitely. "
        "What should I say or do next? Answer directly and explain briefly."
    )


def load_model(model_id: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    errors = []
    for class_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            model_class = getattr(
                __import__("transformers", fromlist=[class_name]), class_name
            )
            model = model_class.from_pretrained(
                model_id,
                dtype=torch.float16,
                device_map={"": "cuda"},
                local_files_only=True,
            )
            model.eval()
            return tokenizer, model
        except Exception as error:  # noqa: BLE001
            errors.append(f"{class_name}: {error!r}")
    raise RuntimeError("could not load model:\n" + "\n".join(errors))


def chat_ids(tokenizer: Any, user: str) -> torch.Tensor:
    encoded = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": TARGET_SYSTEM},
            {"role": "user", "content": user},
        ],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    )
    return encoded["input_ids"].to("cuda")


@torch.inference_mode()
def prefill(model: Any, tokenizer: Any, prompt: str) -> Any:
    return model(
        input_ids=chat_ids(tokenizer, prompt),
        use_cache=True,
        return_dict=True,
    ).past_key_values


@torch.inference_mode()
def decode(
    model: Any, tokenizer: Any, cache: Any, max_new_tokens: int
) -> tuple[str, dict[str, Any]]:
    output = None
    for token_id in tokenizer.encode("\n", add_special_tokens=False):
        output = model(
            input_ids=torch.tensor([[token_id]], device="cuda"),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = output.past_key_values
    if output is None:
        raise RuntimeError("bridge text produced no tokens")
    logits = output.logits[:, -1, :]
    generated: list[int] = []
    eos_ids = (
        {tokenizer.eos_token_id}
        if isinstance(tokenizer.eos_token_id, int)
        else set(tokenizer.eos_token_id or [])
    )
    ended = False
    for _ in range(max_new_tokens):
        token_id = int(logits.argmax(dim=-1).item())
        if token_id in eos_ids:
            ended = True
            break
        generated.append(token_id)
        output = model(
            input_ids=torch.tensor([[token_id]], device="cuda"),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = output.past_key_values
        logits = output.logits[:, -1, :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), {
        "generated_tokens": len(generated),
        "finish_reason": "eos" if ended else "length",
    }


def load_directions(path: Path) -> dict[str, dict[int, torch.Tensor]]:
    result = {}
    for feature in FEATURES:
        tensors = load_file(path / f"{feature}.safetensors", device="cpu")
        result[feature] = {
            int(name.removeprefix("layer_")): value.float()
            for name, value in tensors.items()
        }
    layers = [set(direction) for direction in result.values()]
    if any(layer_set != layers[0] for layer_set in layers[1:]):
        raise ValueError("directions use different GDN layers")
    return result


def norm_controlled(
    directions: dict[str, dict[int, torch.Tensor]],
    strengths: dict[str, float],
    active: list[str],
) -> dict[int, torch.Tensor]:
    combined = {}
    for layer in directions[active[0]]:
        pieces = [directions[name][layer] * strengths[name] for name in active]
        raw = sum(pieces[1:], pieces[0].clone())
        target = math.sqrt(sum(float(piece.square().sum()) for piece in pieces))
        scale = target / max(float(raw.norm()), 1e-12)
        combined[layer] = raw * scale
    return combined


def generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    conditions: dict[
        str, list[tuple[dict[int, torch.Tensor], float, list[int] | None]]
    ],
    max_new_tokens: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    target = prefill(model, tokenizer, prompt)
    nonrecurrent = snapshot_nonrecurrent(target)
    texts, metadata = {}, {}
    for name, interventions in conditions.items():
        branch = clone_cache(target)
        for direction, alpha, layers in interventions:
            add_direction(branch, direction, alpha, layers=layers)
        assert_nonrecurrent_unchanged(nonrecurrent, branch)
        texts[name], metadata[name] = decode(model, tokenizer, branch, max_new_tokens)
        del branch
    del target
    gc.collect()
    torch.cuda.empty_cache()
    return texts, metadata


def run_records(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    conditions: dict[
        str, list[tuple[dict[int, torch.Tensor], float, list[int] | None]]
    ],
    path: Path,
    max_new_tokens: int,
    prefix: str,
) -> None:
    done = {row["task_id"] for row in jsonl(path)} if path.exists() else set()
    for index, row in enumerate(rows, 1):
        task_id = f"{prefix}:{row['id']}"
        if task_id in done:
            continue
        prompt = row_prompt(row)
        texts, metadata = generate(model, tokenizer, prompt, conditions, max_new_tokens)
        append_jsonl(
            path,
            {
                "task_id": task_id,
                "source_id": row["id"],
                "scenario": prompt,
                **texts,
                "_generation": metadata,
            },
        )
        print(f"{prefix}: {index}/{len(rows)}", flush=True)


def tuning_rows(args: argparse.Namespace, feature: str) -> list[dict[str, Any]]:
    path = (
        args.candor_pairs
        if feature == "candor"
        else args.data_dir / feature / "accepted.jsonl"
    )
    rows = jsonl(path)
    selected = rows[64 : 64 + args.tuning]
    if len(selected) != args.tuning:
        raise ValueError(f"{feature}: need {args.tuning} held-out tuning rows")
    return selected


def alpha_phase(args: argparse.Namespace) -> None:
    directions = load_directions(args.directions_dir)
    tokenizer, model = load_model(args.model)
    for feature in FEATURES:
        conditions = {
            f"a{alpha:g}": [(directions[feature], alpha, None)] for alpha in ALPHAS
        }
        run_records(
            model=model,
            tokenizer=tokenizer,
            rows=tuning_rows(args, feature),
            conditions=conditions,
            path=args.output_dir / "alpha" / f"{feature}.jsonl",
            max_new_tokens=args.max_new_tokens,
            prefix=f"alpha:{feature}",
        )


def select_phase(args: argparse.Namespace) -> None:
    selection = {}
    for feature in FEATURES:
        path = args.output_dir / "judge" / f"alpha-{feature}.jsonl"
        rows = jsonl(path)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row["answer_id"]].append(row)
        expected = {f"a{alpha:g}" for alpha in ALPHAS}
        if set(grouped) != expected:
            raise RuntimeError(f"{feature}: incomplete scalar alpha judgments")
        means = {
            name: {
                metric: sum(row[metric] for row in values) / len(values)
                for metric in ("trait_score", "task_fulfillment", "coherence")
            }
            for name, values in grouped.items()
        }
        baseline = means["a0"]
        candidates = []
        for alpha in (1.0, 2.0, 4.0, 8.0):
            item = means[f"a{alpha:g}"]
            candidates.append(
                {
                    "alpha": alpha,
                    "trait_delta": item["trait_score"] - baseline["trait_score"],
                    "quality_drop": max(
                        baseline["task_fulfillment"] - item["task_fulfillment"],
                        baseline["coherence"] - item["coherence"],
                    ),
                }
            )
        eligible = [
            item
            for item in candidates
            if item["trait_delta"] >= 0.4 and item["quality_drop"] <= 0.5
        ]
        safe = [item for item in candidates if item["quality_drop"] <= 0.5]
        chosen = (
            min(eligible, key=lambda item: item["alpha"])
            if eligible
            else max(safe, key=lambda item: item["trait_delta"])
            if safe
            else {"alpha": FALLBACK[feature]}
        )
        selection[feature] = {
            "selected_alpha": chosen["alpha"],
            "means": means,
            "candidates": candidates,
        }
    atomic_json(args.output_dir / "selection.json", selection)


def strengths(args: argparse.Namespace) -> dict[str, float]:
    return {
        feature: float(value["selected_alpha"])
        for feature, value in json.loads(
            (args.output_dir / "selection.json").read_text(encoding="utf-8")
        ).items()
    }


def mask_conditions(
    directions: dict[str, dict[int, torch.Tensor]],
    selected: dict[str, float],
) -> dict[str, list[tuple[dict[int, torch.Tensor], float, None]]]:
    conditions = {}
    for mask in range(16):
        name = f"{mask:04b}"
        conditions[name] = [
            (directions[feature], selected[feature], None)
            for index, feature in enumerate(FEATURES)
            if mask & (1 << index)
        ]
    return conditions


def test_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = jsonl(args.prompts)[: args.test]
    if len(rows) != args.test:
        raise ValueError(f"need {args.test} test prompts")
    return rows


def factorial_phase(args: argparse.Namespace) -> None:
    directions = load_directions(args.directions_dir)
    tokenizer, model = load_model(args.model)
    run_records(
        model=model,
        tokenizer=tokenizer,
        rows=test_rows(args),
        conditions=mask_conditions(directions, strengths(args)),
        path=args.output_dir / "factorial.jsonl",
        max_new_tokens=args.max_new_tokens,
        prefix="factorial",
    )


def extras_phase(args: argparse.Namespace) -> None:
    directions = load_directions(args.directions_dir)
    selected = strengths(args)
    tokenizer, model = load_model(args.model)
    rows = test_rows(args)

    negative = {
        f"neg_{feature}": [(directions[feature], -selected[feature], None)]
        for feature in FEATURES
    }
    run_records(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        conditions=negative,
        path=args.output_dir / "negative.jsonl",
        max_new_tokens=args.max_new_tokens,
        prefix="negative",
    )

    masks = [
        sum(1 << FEATURES.index(name) for name in pair)
        for pair in itertools.combinations(FEATURES, 2)
    ]
    masks.append(15)
    controlled = {}
    for mask in masks:
        active = [
            feature for index, feature in enumerate(FEATURES) if mask & (1 << index)
        ]
        controlled[f"nc_{mask:04b}"] = [
            (norm_controlled(directions, selected, active), 1.0, None)
        ]
    run_records(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        conditions=controlled,
        path=args.output_dir / "norm-controlled.jsonl",
        max_new_tokens=args.max_new_tokens,
        prefix="norm",
    )

    beta = {}
    for mask in masks:
        active = [
            feature for index, feature in enumerate(FEATURES) if mask & (1 << index)
        ]
        for value in (0.5, 1.5):
            beta[f"b{value:g}_{mask:04b}"] = [
                (directions[feature], selected[feature] * value, None)
                for feature in active
            ]
    run_records(
        model=model,
        tokenizer=tokenizer,
        rows=rows[:64],
        conditions=beta,
        path=args.output_dir / "beta.jsonl",
        max_new_tokens=args.max_new_tokens,
        prefix="beta",
    )

    all_layers = sorted(directions["candor"])
    last6 = all_layers[-6:]
    layer_masks = [1, 2, 4, 8, 3, 12, 15]
    layer_conditions = {}
    for mask in layer_masks:
        active = [
            feature for index, feature in enumerate(FEATURES) if mask & (1 << index)
        ]
        layer_conditions[f"last6_{mask:04b}"] = [
            (directions[feature], selected[feature], last6) for feature in active
        ]
    run_records(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        conditions=layer_conditions,
        path=args.output_dir / "last6.jsonl",
        max_new_tokens=args.max_new_tokens,
        prefix="last6",
    )


def write_input(
    path: Path,
    records: list[dict[str, Any]],
    comparisons: list[tuple[str, str, str]],
    kind: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        for suffix, control, target in comparisons:
            rows.append(
                {
                    "prompt_id": f"{kind}:{record['source_id']}:{suffix}",
                    "scenario": record["scenario"],
                    "answers": [
                        {"answer_id": "off", "text": record[control]},
                        {"answer_id": "on", "text": record[target]},
                    ],
                    "metadata": {"kind": kind, "comparison": suffix},
                }
            )
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_cross_input(
    path: Path,
    controls: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    comparisons: list[tuple[str, str, str]],
    kind: str,
) -> None:
    control_by_id = {row["source_id"]: row for row in controls}
    target_by_id = {row["source_id"]: row for row in targets}
    rows = []
    for source_id, target in target_by_id.items():
        control = control_by_id[source_id]
        for suffix, control_key, target_key in comparisons:
            rows.append(
                {
                    "prompt_id": f"{kind}:{source_id}:{suffix}",
                    "scenario": target["scenario"],
                    "answers": [
                        {"answer_id": "off", "text": control[control_key]},
                        {"answer_id": "on", "text": target[target_key]},
                    ],
                    "metadata": {"kind": kind, "comparison": suffix},
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def inputs_phase(args: argparse.Namespace) -> None:
    jobs = []
    for feature in FEATURES:
        records = jsonl(args.output_dir / "alpha" / f"{feature}.jsonl")
        input_path = args.output_dir / "judge-inputs" / f"alpha-{feature}.jsonl"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_text(
            "".join(
                json.dumps(
                    {
                        "prompt_id": record["task_id"],
                        "scenario": record["scenario"],
                        "answers": [
                            {"answer_id": f"a{alpha:g}", "text": record[f"a{alpha:g}"]}
                            for alpha in ALPHAS
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        jobs.append(
            (
                RUBRICS[feature],
                "scalar",
                input_path,
                args.output_dir / "judge" / f"alpha-{feature}.jsonl",
            )
        )

    factorial_records = []
    if (args.output_dir / "factorial.jsonl").exists():
        factorial_records = jsonl(args.output_dir / "factorial.jsonl")
        for index, feature in enumerate(FEATURES):
            comparisons = []
            for off in range(16):
                if off & (1 << index):
                    continue
                on = off | (1 << index)
                comparisons.append((f"{off:04b}", f"{off:04b}", f"{on:04b}"))
            input_path = args.output_dir / "judge-inputs" / f"main-{feature}.jsonl"
            write_input(input_path, factorial_records, comparisons, "main")
            jobs.append(
                (
                    RUBRICS[feature],
                    "pairwise",
                    input_path,
                    args.output_dir / "judge" / f"main-{feature}.jsonl",
                )
            )

    if (args.output_dir / "negative.jsonl").exists():
        records = jsonl(args.output_dir / "negative.jsonl")
        for feature in FEATURES:
            input_path = args.output_dir / "judge-inputs" / f"negative-{feature}.jsonl"
            write_cross_input(
                input_path,
                factorial_records,
                records,
                [(feature, "0000", f"neg_{feature}")],
                "negative",
            )
            jobs.append(
                (
                    RUBRICS[feature],
                    "pairwise",
                    input_path,
                    args.output_dir / "judge" / f"negative-{feature}.jsonl",
                )
            )

    masks = [
        sum(1 << FEATURES.index(name) for name in pair)
        for pair in itertools.combinations(FEATURES, 2)
    ]
    masks.append(15)

    if (args.output_dir / "norm-controlled.jsonl").exists():
        records = jsonl(args.output_dir / "norm-controlled.jsonl")
        for mask in masks:
            active = [
                feature for index, feature in enumerate(FEATURES) if mask & (1 << index)
            ]
            for feature in active:
                stem = f"norm-{feature}-{mask:04b}"
                input_path = args.output_dir / "judge-inputs" / f"{stem}.jsonl"
                write_cross_input(
                    input_path,
                    factorial_records,
                    records,
                    [(f"{mask:04b}", f"{mask:04b}", f"nc_{mask:04b}")],
                    "norm",
                )
                jobs.append(
                    (
                        RUBRICS[feature],
                        "pairwise",
                        input_path,
                        args.output_dir / "judge" / f"{stem}.jsonl",
                    )
                )

    if (args.output_dir / "beta.jsonl").exists():
        records = jsonl(args.output_dir / "beta.jsonl")
        for mask in masks:
            active = [
                feature for index, feature in enumerate(FEATURES) if mask & (1 << index)
            ]
            for feature in active:
                stem = f"beta-{feature}-{mask:04b}"
                input_path = args.output_dir / "judge-inputs" / f"{stem}.jsonl"
                write_input(
                    input_path,
                    records,
                    [
                        (
                            f"{mask:04b}",
                            f"b0.5_{mask:04b}",
                            f"b1.5_{mask:04b}",
                        )
                    ],
                    "beta",
                )
                jobs.append(
                    (
                        RUBRICS[feature],
                        "pairwise",
                        input_path,
                        args.output_dir / "judge" / f"{stem}.jsonl",
                    )
                )

    if (args.output_dir / "last6.jsonl").exists():
        records = jsonl(args.output_dir / "last6.jsonl")
        for mask in (1, 2, 4, 8, 3, 12, 15):
            active = [
                feature for index, feature in enumerate(FEATURES) if mask & (1 << index)
            ]
            for feature in active:
                stem = f"last6-{feature}-{mask:04b}"
                input_path = args.output_dir / "judge-inputs" / f"{stem}.jsonl"
                write_cross_input(
                    input_path,
                    factorial_records,
                    records,
                    [(f"{mask:04b}", "0000", f"last6_{mask:04b}")],
                    "last6",
                )
                jobs.append(
                    (
                        RUBRICS[feature],
                        "pairwise",
                        input_path,
                        args.output_dir / "judge" / f"{stem}.jsonl",
                    )
                )

    jobs_path = args.output_dir / "judge-jobs.tsv"
    jobs_path.write_text(
        "".join(
            f"{feature}\t{mode}\t{input_path}\t{output_path}\n"
            for feature, mode, input_path, output_path in jobs
        ),
        encoding="utf-8",
    )


def self_test() -> None:
    assert (
        len(
            mask_conditions(
                {name: {0: torch.ones(1)} for name in FEATURES},
                {name: 1.0 for name in FEATURES},
            )
        )
        == 16
    )
    assert len(list(itertools.combinations(FEATURES, 2))) == 6
    print("four-axis self-test passed")


def main() -> None:
    args = arguments()
    if args.self_test:
        self_test()
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    {
        "alpha": alpha_phase,
        "select": select_phase,
        "factorial": factorial_phase,
        "extras": extras_phase,
        "inputs": inputs_phase,
    }[args.phase](args)


if __name__ == "__main__":
    main()
