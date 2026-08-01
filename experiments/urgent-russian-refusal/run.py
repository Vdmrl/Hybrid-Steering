from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hybrid_steering import (
    add_direction,
    clone_cache,
)
from hybrid_steering.cache import recurrent_tensor
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

FEATURES = (
    "russian_language",
    "optimism",
    "casualness",
    "refusal",
)
DEFAULT_STRENGTHS = {
    "russian_language": 2.0344,
    "optimism": 4.0,
    "casualness": 2.3566,
    "refusal": 1.0,
}
DIRECTION_SIGNS = {
    "russian_language": -1.0,  # source stores English - Russian
    "optimism": 1.0,
    "casualness": 1.0,
    "refusal": 1.0,
}
GDN_DECODER_LAYERS = (
    0,
    1,
    2,
    4,
    5,
    6,
    8,
    9,
    10,
    12,
    13,
    14,
    16,
    17,
    18,
    20,
    21,
    22,
    24,
    25,
    26,
    28,
    29,
    30,
)
TARGET_SYSTEM = "Answer the user's question directly and naturally."


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=("self-test", "smoke", "main", "factuality"),
    )
    parser.add_argument("--directions-dir", type=Path, required=True)
    parser.add_argument("--dev-prompts", type=Path)
    parser.add_argument("--test-prompts", type=Path)
    parser.add_argument("--factual-prompts", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--test", type=int, default=128)
    parser.add_argument("--ridge", type=float, default=1e-6)
    return parser.parse_args()


def jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def row_prompt(row: dict[str, Any]) -> str:
    if row.get("prompt") or row.get("text"):
        return str(row.get("prompt") or row["text"])
    setup = row.get("shared_setup") or row.get("narrative_guidance")
    if not setup:
        raise ValueError(f"row {row.get('id')} has no scenario")
    return (
        f"{setup}\n\nThe decision cannot be postponed indefinitely. "
        "What should I say or do next? Answer directly and explain briefly."
    )


def chat_ids(tokenizer: Any, system: str, user: str) -> torch.Tensor:
    encoded = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    )
    return encoded["input_ids"].to("cuda")


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


def save_direction(path: Path, direction: dict[int, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {f"layer_{layer}": tensor.contiguous() for layer, tensor in direction.items()},
        str(path),
    )


def load_direction(path: Path) -> dict[int, torch.Tensor]:
    if path.suffix == ".npz":
        value = np.load(path)["deltaS"]
        if value.ndim != 4:
            raise ValueError(f"expected [layer, head, 128, 128], got {value.shape}")
        if value.shape[0] != len(GDN_DECODER_LAYERS):
            raise ValueError(
                f"expected {len(GDN_DECODER_LAYERS)} GDN layers, got {value.shape[0]}"
            )
        return {
            decoder_layer: torch.from_numpy(value[ordinal]).unsqueeze(0).float()
            for ordinal, decoder_layer in enumerate(GDN_DECODER_LAYERS)
        }
    return {
        int(name.removeprefix("layer_")): tensor.float()
        for name, tensor in load_file(path, device="cpu").items()
    }


def truncate_tensor(value: torch.Tensor, rank: int) -> torch.Tensor:
    matrices = value.squeeze(0).to("cuda", dtype=torch.float32)
    u, s, vh = torch.linalg.svd(matrices, full_matrices=False)
    result = (u[..., :rank] * s[..., :rank].unsqueeze(-2)) @ vh[..., :rank, :]
    return result.unsqueeze(0).cpu()


def truncate_direction(
    direction: dict[int, torch.Tensor], rank: int
) -> dict[int, torch.Tensor]:
    return {layer: truncate_tensor(value, rank) for layer, value in direction.items()}


def load_directions(args: argparse.Namespace, rank: int) -> dict[str, dict]:
    result = {}
    for feature in FEATURES:
        candidates = [
            args.directions_dir / f"{feature}.safetensors",
            args.directions_dir / f"{feature}.npz",
        ]
        source = next((path for path in candidates if path.exists()), None)
        if source is None:
            raise FileNotFoundError(f"missing direction for {feature}: {candidates}")
        cache = args.output_dir / "directions" / f"{feature}-rank{rank}.safetensors"
        if not cache.exists():
            direction = {
                layer: value * DIRECTION_SIGNS[feature]
                for layer, value in load_direction(source).items()
            }
            save_direction(cache, truncate_direction(direction, rank))
        result[feature] = load_direction(cache)
    layer_sets = [set(value) for value in result.values()]
    if any(layers != layer_sets[0] for layers in layer_sets[1:]):
        raise RuntimeError("directions use different GDN layers")
    return result


def rss_coefficients(
    directions: dict[str, dict[int, torch.Tensor]],
    coefficients: dict[str, float],
) -> dict[int, dict[str, float]]:
    active = tuple(coefficients)
    result = {}
    for layer in directions[active[0]]:
        pieces = [directions[name][layer] * coefficients[name] for name in active]
        raw = sum(pieces[1:], pieces[0].clone())
        target = math.sqrt(sum(float(piece.square().sum()) for piece in pieces))
        scale = target / max(float(raw.norm()), 1e-12)
        result[layer] = {name: coefficients[name] * scale for name in active}
    return result


def gram_inverse(flat_basis: torch.Tensor, ridge: float) -> torch.Tensor:
    gram = flat_basis @ flat_basis.T
    scale = torch.diag(gram).mean().clamp_min(1e-12)
    gram = gram + torch.eye(len(flat_basis), device=gram.device) * ridge * scale
    return torch.linalg.inv(gram)


def gram_coefficients(
    state: torch.Tensor,
    basis: list[torch.Tensor],
    ridge: float,
) -> torch.Tensor:
    flat_basis = torch.stack([value.float().flatten() for value in basis])
    return gram_inverse(flat_basis, ridge) @ (flat_basis @ state.float().flatten())


def make_clamp_runtime(
    cache: Any,
    directions: dict[str, dict[int, torch.Tensor]],
    deltas: dict[int, dict[str, float]],
    ridge: float,
) -> dict[int, dict[str, Any]]:
    result = {}
    for layer, by_feature in deltas.items():
        state = recurrent_tensor(cache, layer)
        names = tuple(by_feature)
        flat_basis = torch.stack(
            [directions[name][layer].to(state).float().flatten() for name in names]
        )
        inverse = gram_inverse(flat_basis, ridge)
        initial = inverse @ (flat_basis @ state.float().flatten())
        result[layer] = {
            "names": names,
            "flat_basis": flat_basis,
            "inverse": inverse,
            "target": initial
            + torch.tensor([by_feature[name] for name in names], device=state.device),
        }
    return result


@torch.no_grad()
def clamp_cache(
    cache: Any,
    runtime: dict[int, dict[str, Any]],
    beta: float,
) -> dict[int, dict[str, float]]:
    observed = {}
    for layer, values in runtime.items():
        state = recurrent_tensor(cache, layer)
        flat_basis = values["flat_basis"]
        current = values["inverse"] @ (flat_basis @ state.float().flatten())
        correction = values["target"] - current
        state.add_((correction @ flat_basis).reshape(state.shape).to(state), alpha=beta)
        final = current + beta * correction
        observed[layer] = {
            name: float(final[index]) for index, name in enumerate(values["names"])
        }
    return observed


@torch.inference_mode()
def prefill(model: Any, tokenizer: Any, prompt: str) -> Any:
    return model(
        input_ids=chat_ids(tokenizer, TARGET_SYSTEM, prompt),
        use_cache=True,
        return_dict=True,
    ).past_key_values


@torch.inference_mode()
def decode(
    model: Any,
    tokenizer: Any,
    cache: Any,
    max_new_tokens: int,
    clamp: tuple[dict, float] | None = None,
) -> tuple[str, dict[str, Any]]:
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
        if clamp:
            clamp_cache(cache, *clamp)
    if output is None:
        raise RuntimeError("bridge text produced no tokens")
    logits = output.logits[:, -1, :]
    generated = []
    eos = {tokenizer.eos_token_id}
    checkpoints = {0, max_new_tokens // 4, max_new_tokens // 2, 3 * max_new_tokens // 4}
    for step in range(max_new_tokens):
        token_id = int(logits.argmax(dim=-1).item())
        if token_id in eos:
            break
        generated.append(token_id)
        output = model(
            input_ids=torch.tensor([[token_id]], device="cuda"),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = output.past_key_values
        if clamp:
            observed = clamp_cache(cache, *clamp)
            if step in checkpoints:
                trace.append({"token": step + 1, "coefficients": observed})
        logits = output.logits[:, -1, :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), {
        "generated_tokens": len(generated),
        "finish_reason": "eos" if len(generated) < max_new_tokens else "length",
        "coefficient_trace": trace,
    }


def condition_plan(rank1: dict, beta: float = 1.0) -> dict:
    strengths = dict(DEFAULT_STRENGTHS)
    plans: dict[str, dict[str, Any]] = {"baseline": {"kind": "add", "items": []}}
    labels = {1: "singleton", 2: "pair", 3: "triple", 4: "full"}
    for size in range(1, len(FEATURES) + 1):
        for names in itertools.combinations(FEATURES, size):
            coefficients = {name: strengths[name] for name in names}
            suffix = "+".join(names)
            plans[f"{labels[size]}_{suffix}"] = {
                "kind": "clamp",
                "directions": rank1,
                "deltas": rss_coefficients(rank1, coefficients),
                "beta": beta,
            }
    return plans


def generate_condition(
    model: Any,
    tokenizer: Any,
    base_cache: Any,
    plan: dict[str, Any],
    max_new_tokens: int,
    ridge: float,
) -> tuple[str, dict]:
    cache = clone_cache(base_cache)
    if plan["kind"] == "add":
        for direction, alpha in plan["items"]:
            add_direction(cache, direction, alpha)
        clamp = None
    elif plan["kind"] == "add_coefficients":
        for layer, coefficients in plan["coefficients"].items():
            for name, alpha in coefficients.items():
                add_direction(
                    cache,
                    {layer: plan["directions"][name][layer]},
                    alpha,
                    layers=[layer],
                )
        clamp = None
    else:
        runtime = make_clamp_runtime(cache, plan["directions"], plan["deltas"], ridge)
        clamp_cache(cache, runtime, 1.0)
        clamp = (runtime, plan["beta"])
    return decode(model, tokenizer, cache, max_new_tokens, clamp)


def run_rows(
    args: argparse.Namespace, rows: list[dict], plans: dict, path: Path, prefix: str
) -> None:
    tokenizer, model = load_model(args.model)
    done = {row["task_id"] for row in jsonl(path)}
    for row_index, row in enumerate(rows, 1):
        prompt = row_prompt(row)
        base_cache = prefill(model, tokenizer, prompt)
        for name, plan in plans.items():
            task_id = f"{prefix}:{row['id']}:{name}"
            if task_id in done:
                continue
            text, metadata = generate_condition(
                model, tokenizer, base_cache, plan, args.max_new_tokens, args.ridge
            )
            append_jsonl(
                path,
                {
                    "task_id": task_id,
                    "source_id": row["id"],
                    "scenario": prompt,
                    "condition": name,
                    "response": text,
                    "generation": metadata,
                },
            )
            print(
                f"{prefix}: prompt {row_index}/{len(rows)} condition={name}", flush=True
            )
        del base_cache
        gc.collect()
        torch.cuda.empty_cache()


def smoke_phase(args: argparse.Namespace) -> None:
    rank1 = load_directions(args, 1)
    plans = condition_plan(rank1)
    keep = {
        name: plan
        for name, plan in plans.items()
        if name == "baseline"
        or name.startswith("singleton_")
        or name.startswith("full_")
    }
    row = jsonl(args.dev_prompts)[0]
    original = args.max_new_tokens
    args.max_new_tokens = min(16, original)
    run_rows(args, [row], keep, args.output_dir / "smoke.jsonl", "exp5:smoke")


def main_phase(args: argparse.Namespace) -> None:
    rank1 = load_directions(args, 1)
    plans = condition_plan(rank1)
    rows = jsonl(args.test_prompts)[: args.test]
    if len(rows) != args.test:
        raise RuntimeError(f"need {args.test} test prompts, found {len(rows)}")
    run_rows(
        args,
        rows,
        plans,
        args.output_dir / "main-generations.jsonl",
        "urgent4:main",
    )


def factuality_phase(args: argparse.Namespace) -> None:
    rank1 = load_directions(args, 1)
    all_plans = condition_plan(rank1)
    tag = "triple_russian_language+optimism+casualness"
    plans = {"baseline": all_plans["baseline"], tag: all_plans[tag]}
    run_rows(
        args,
        jsonl(args.factual_prompts)[:30],
        plans,
        args.output_dir / "factuality-generations.jsonl",
        "urgent4:factuality",
    )


def self_test() -> None:
    eye = torch.eye(3)
    state = 2 * eye[0] + 3 * eye[1]
    values = gram_coefficients(state, [eye[0], eye[1]], 1e-8)
    assert torch.allclose(values, torch.tensor([2.0, 3.0]), atol=1e-5)
    toy = {
        name: {0: torch.eye(3).reshape(1, 1, 3, 3) * (index + 1)}
        for index, name in enumerate(FEATURES)
    }
    plans = condition_plan(toy)
    assert len(plans) == 16
    assert sum(name.startswith("pair_") for name in plans) == 6
    assert sum(name.startswith("triple_") for name in plans) == 4
    print("urgent four-axis clamp self-test passed")


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    {
        "self-test": lambda _: self_test(),
        "smoke": smoke_phase,
        "main": main_phase,
        "factuality": factuality_phase,
    }[args.phase](args)


if __name__ == "__main__":
    main()
