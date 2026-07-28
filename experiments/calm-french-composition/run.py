"""Build calm/French GDN directions and run a 2x2 composition evaluation."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from langdetect import DetectorFactory, detect
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "steering" / "src"))

from hybrid_steering import (
    add_direction,
    assert_nonrecurrent_unchanged,
    clone_cache,
    extract_recurrent,
    snapshot_nonrecurrent,
)

DetectorFactory.seed = 0
DONOR_SYSTEM = (
    "Read the following response in its situation. Preserve its behavioral "
    "stance and language in your internal state."
)
TARGET_SYSTEM = "Answer the user's question directly and naturally."


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--language-pairs", type=Path, required=True)
    parser.add_argument("--calm-pairs", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--donors", type=int, default=64)
    parser.add_argument("--validation", type=int, default=64)
    parser.add_argument("--calm-alpha", type=float, default=2.0)
    parser.add_argument("--french-alpha-grid", default="1,2,4")
    parser.add_argument("--calibration-prompts", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
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


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


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
        except Exception as error:  # noqa: BLE001 - try the compatible auto class
            errors.append(f"{class_name}: {error!r}")
    raise RuntimeError("could not load model:\n" + "\n".join(errors))


def chat_ids(tokenizer: Any, system: str, user: str) -> torch.Tensor:
    encoded = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
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
def prefill(model: Any, tokenizer: Any, system: str, user: str) -> Any:
    output = model(
        input_ids=chat_ids(tokenizer, system, user),
        use_cache=True,
        return_dict=True,
    )
    return output.past_key_values


@torch.inference_mode()
def decode(model: Any, tokenizer: Any, cache: Any, max_new_tokens: int) -> str:
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
    generated = []
    eos = tokenizer.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    for _ in range(max_new_tokens):
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
        logits = output.logits[:, -1, :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def save_direction(path: Path, direction: dict[int, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {f"layer_{layer}": tensor.contiguous() for layer, tensor in direction.items()},
        str(path),
    )


def load_direction(path: Path) -> dict[int, torch.Tensor]:
    return {
        int(name.removeprefix("layer_")): tensor.float()
        for name, tensor in load_file(path, device="cpu").items()
    }


def build_direction(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    name: str,
) -> tuple[dict[int, torch.Tensor], dict[str, dict[int, torch.Tensor]]]:
    sums: dict[int, torch.Tensor] = {}
    source_sums: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)
    source_counts: dict[str, int] = defaultdict(int)
    for index, row in enumerate(rows, 1):
        negative_cache = prefill(model, tokenizer, DONOR_SYSTEM, row["negative_text"])
        positive_cache = prefill(model, tokenizer, DONOR_SYSTEM, row["positive_text"])
        negative = extract_recurrent(negative_cache)
        positive = extract_recurrent(positive_cache)
        source = str(row.get("source_language", "all"))
        source_counts[source] += 1
        for layer in positive:
            difference = positive[layer] - negative[layer]
            sums[layer] = sums.get(layer, torch.zeros_like(difference)) + difference
            source_sums[source][layer] = (
                source_sums[source].get(layer, torch.zeros_like(difference))
                + difference
            )
        del negative_cache, positive_cache, negative, positive
        if index % 8 == 0:
            print(f"direction {name}: {index}/{len(rows)}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    direction = {layer: tensor / len(rows) for layer, tensor in sums.items()}
    by_source = {
        source: {
            layer: tensor / source_counts[source] for layer, tensor in values.items()
        }
        for source, values in source_sums.items()
    }
    return direction, by_source


def ensure_direction(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    name: str,
    path: Path,
) -> tuple[dict[int, torch.Tensor], dict[str, dict[int, torch.Tensor]]]:
    if path.exists():
        print(f"reuse direction {name}: {path}", flush=True)
        return load_direction(path), {}
    direction, by_source = build_direction(model, tokenizer, rows, name)
    save_direction(path, direction)
    return direction, by_source


def flattened_cosine(
    left: dict[int, torch.Tensor], right: dict[int, torch.Tensor]
) -> float:
    a = torch.cat([left[layer].flatten().double() for layer in sorted(left)])
    b = torch.cat([right[layer].flatten().double() for layer in sorted(right)])
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12))


def direction_norm(direction: dict[int, torch.Tensor]) -> float:
    return math.sqrt(sum(float(value.square().sum()) for value in direction.values()))


def generate_conditions(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    conditions: dict[str, list[tuple[dict[int, torch.Tensor], float]]],
    max_new_tokens: int,
) -> dict[str, str]:
    target = prefill(model, tokenizer, TARGET_SYSTEM, prompt)
    nonrecurrent = snapshot_nonrecurrent(target)
    outputs = {}
    for name, interventions in conditions.items():
        branch = clone_cache(target)
        for direction, alpha in interventions:
            add_direction(branch, direction, alpha)
        assert_nonrecurrent_unchanged(nonrecurrent, branch)
        outputs[name] = decode(model, tokenizer, branch, max_new_tokens)
        del branch
    del target
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


def is_french(text: str) -> bool:
    try:
        return len(text.split()) >= 3 and detect(text) == "fr"
    except Exception:  # noqa: BLE001 - language detection is diagnostic only
        return False


def run_records(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    conditions: dict[str, list[tuple[dict[int, torch.Tensor], float]]],
    path: Path,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    done = {row["prompt_id"]: row for row in jsonl(path)} if path.exists() else {}
    for index, row in enumerate(rows, 1):
        prompt_id = str(row["id"])
        if prompt_id in done:
            continue
        result = {
            "prompt_id": prompt_id,
            "scenario": row["prompt"],
            **generate_conditions(
                model=model,
                tokenizer=tokenizer,
                prompt=row["prompt"],
                conditions=conditions,
                max_new_tokens=max_new_tokens,
            ),
        }
        append_jsonl(path, result)
        done[prompt_id] = result
        print(f"{path.stem}: {index}/{len(rows)}", flush=True)
    return [done[str(row["id"])] for row in rows]


def select_french_alpha(
    records: list[dict[str, Any]], alphas: Iterable[float]
) -> tuple[float, dict[str, Any]]:
    metrics = {}
    for alpha in alphas:
        name = f"french_a{alpha:g}"
        texts = [row[name] for row in records]
        metrics[name] = {
            "french_rate": sum(is_french(text) for text in texts) / len(texts),
            "mean_words": sum(len(text.split()) for text in texts) / len(texts),
        }
    eligible = [
        alpha
        for alpha in alphas
        if metrics[f"french_a{alpha:g}"]["french_rate"] >= 0.8
        and metrics[f"french_a{alpha:g}"]["mean_words"] >= 8
    ]
    selected = (
        min(eligible)
        if eligible
        else max(
            alphas,
            key=lambda alpha: (
                metrics[f"french_a{alpha:g}"]["french_rate"],
                metrics[f"french_a{alpha:g}"]["mean_words"],
            ),
        )
    )
    return selected, metrics


def write_judge_inputs(records: list[dict[str, Any]], output_dir: Path) -> None:
    comparisons = {
        "calm_single": ("baseline", "calm"),
        "calm_with_french": ("french", "calm_french"),
        "french_single": ("baseline", "french"),
        "french_with_calm": ("calm", "calm_french"),
    }
    for name, (control, target) in comparisons.items():
        path = output_dir / "judge-inputs" / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "prompt_id": row["prompt_id"],
                        "scenario": row["scenario"],
                        "answers": [
                            {"answer_id": "control", "text": row[control]},
                            {"answer_id": "target", "text": row[target]},
                        ],
                        "metadata": {"comparison": name},
                    },
                    ensure_ascii=False,
                )
                + "\n"
                for row in records
            ),
            encoding="utf-8",
        )


def self_test() -> None:
    records = [
        {
            "french_a1": "This is an English response.",
            "french_a2": "Voici une réponse entièrement rédigée en français.",
        }
    ]
    alpha, _ = select_french_alpha(records, [1.0, 2.0])
    assert alpha == 2.0
    print("run self-test passed")


def main() -> None:
    args = arguments()
    if args.self_test:
        self_test()
        return
    language_rows = jsonl(args.language_pairs)
    calm_rows = jsonl(args.calm_pairs)
    prompt_rows = jsonl(args.prompts)[: args.validation]
    if len(language_rows) < args.donors or len(calm_rows) < args.donors:
        raise RuntimeError("need 64 donor pairs for both directions")
    if len(prompt_rows) != args.validation:
        raise RuntimeError("incomplete validation prompt split")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_model(args.model)
    french, by_source = ensure_direction(
        model=model,
        tokenizer=tokenizer,
        rows=language_rows[: args.donors],
        name="french",
        path=args.output_dir / "directions" / "french.safetensors",
    )
    calm, _ = ensure_direction(
        model=model,
        tokenizer=tokenizer,
        rows=calm_rows[: args.donors],
        name="calm",
        path=args.output_dir / "directions" / "calm.safetensors",
    )
    diagnostics = {
        "donors": args.donors,
        "layers": sorted(french),
        "french_norm": direction_norm(french),
        "calm_norm": direction_norm(calm),
        "calm_french_cosine": flattened_cosine(calm, french),
        "french_source_cosines": {
            source: flattened_cosine(direction, french)
            for source, direction in by_source.items()
        },
    }
    atomic_json(args.output_dir / "direction_diagnostics.json", diagnostics)

    alphas = sorted(
        {float(value.strip()) for value in args.french_alpha_grid.split(",")}
    )
    calibration_conditions = {"baseline": []}
    calibration_conditions.update(
        {f"french_a{alpha:g}": [(french, alpha)] for alpha in alphas}
    )
    calibration = run_records(
        model=model,
        tokenizer=tokenizer,
        rows=prompt_rows[: args.calibration_prompts],
        conditions=calibration_conditions,
        path=args.output_dir / "calibration.jsonl",
        max_new_tokens=args.max_new_tokens,
    )
    french_alpha, calibration_metrics = select_french_alpha(calibration, alphas)
    atomic_json(
        args.output_dir / "selection.json",
        {
            "calm_alpha": args.calm_alpha,
            "french_alpha": french_alpha,
            "french_calibration": calibration_metrics,
        },
    )

    conditions = {
        "baseline": [],
        "calm": [(calm, args.calm_alpha)],
        "french": [(french, french_alpha)],
        "calm_french": [(calm, args.calm_alpha), (french, french_alpha)],
    }
    generations = run_records(
        model=model,
        tokenizer=tokenizer,
        rows=prompt_rows,
        conditions=conditions,
        path=args.output_dir / "generations.jsonl",
        max_new_tokens=args.max_new_tokens,
    )
    write_judge_inputs(generations, args.output_dir)
    atomic_json(
        args.output_dir / "language_summary.json",
        {
            condition: {
                "n": len(generations),
                "french_rate": sum(is_french(row[condition]) for row in generations)
                / len(generations),
                "mean_words": sum(len(row[condition].split()) for row in generations)
                / len(generations),
            }
            for condition in conditions
        },
    )
    atomic_json(
        args.output_dir / "run_manifest.json",
        {
            "model": args.model,
            "donors_per_direction": args.donors,
            "validation_prompts": args.validation,
            "conditions": list(conditions),
            "calm_alpha": args.calm_alpha,
            "french_alpha": french_alpha,
            "max_new_tokens": args.max_new_tokens,
            "judge_version": "v2",
        },
    )
    print("generation complete", flush=True)


if __name__ == "__main__":
    main()
