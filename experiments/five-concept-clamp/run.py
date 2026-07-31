from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
from pathlib import Path
from typing import Any

import torch
from hybrid_steering import (
    add_direction,
    clone_cache,
    extract_recurrent,
    mean_direction,
    subtract_states,
)
from hybrid_steering.cache import recurrent_tensor
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

FEATURES = (
    "french_language",
    "concrete_language",
    "principled_candor",
    "optimism",
    "first_person_voice",
)
DEFAULT_STRENGTHS = {
    "french_language": 4.0,
    "concrete_language": 4.0,
    "principled_candor": 8.0,
    "optimism": 8.0,
    "first_person_voice": 4.0,
}
BETAS = (0.2, 0.5, 1.0)
TARGET_SYSTEM = "Answer the user's question directly and naturally."
DONOR_SYSTEM = (
    "Read the following response in its situation. Preserve its behavioral "
    "stance and language in your internal state."
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=("self-test", "direction", "smoke", "dev", "main", "extension"),
    )
    parser.add_argument("--directions-dir", type=Path)
    parser.add_argument("--first-person-pairs", type=Path)
    parser.add_argument("--dev-prompts", type=Path)
    parser.add_argument("--test-prompts", type=Path)
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
    if row.get("prompt"):
        return str(row["prompt"])
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


@torch.inference_mode()
def state_for_text(model: Any, tokenizer: Any, text: str) -> dict[int, torch.Tensor]:
    output = model(
        input_ids=chat_ids(tokenizer, DONOR_SYSTEM, text),
        use_cache=True,
        return_dict=True,
    )
    return extract_recurrent(output.past_key_values)


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


def direction_phase(args: argparse.Namespace) -> None:
    path = args.output_dir / "directions" / "first_person_voice.safetensors"
    if path.exists():
        print(f"direction already exists: {path}")
        return
    rows = jsonl(args.first_person_pairs)
    if len(rows) != 64:
        raise RuntimeError(f"need 64 first-person pairs, found {len(rows)}")
    tokenizer, model = load_model(args.model)
    differences = []
    for index, row in enumerate(rows, 1):
        negative = state_for_text(model, tokenizer, row["negative_text"])
        positive = state_for_text(model, tokenizer, row["positive_text"])
        differences.append(subtract_states(positive, negative))
        if index % 8 == 0:
            print(f"first-person direction: {index}/64", flush=True)
    save_direction(path, mean_direction(differences))


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
        source = (
            args.output_dir / "directions" / "first_person_voice.safetensors"
            if feature == "first_person_voice"
            else args.directions_dir / f"{feature}.safetensors"
        )
        cache = args.output_dir / "directions" / f"{feature}-rank{rank}.safetensors"
        if not cache.exists():
            save_direction(cache, truncate_direction(load_direction(source), rank))
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


def condition_plan(rank1: dict, rank4: dict, scale: float, beta: float) -> dict:
    strengths = {name: DEFAULT_STRENGTHS[name] * scale for name in FEATURES}
    plans: dict[str, dict[str, Any]] = {"baseline": {"kind": "add", "items": []}}
    for name in FEATURES:
        plans[f"singleton_{name}"] = {
            "kind": "add",
            "items": [(rank1[name], strengths[name])],
        }
    full = dict(strengths)
    plans.update(
        {
            "full_add_raw_r1": {
                "kind": "add",
                "items": [(rank1[name], full[name]) for name in FEATURES],
            },
            "full_add_rss_r1": {
                "kind": "add_coefficients",
                "directions": rank1,
                "coefficients": rss_coefficients(rank1, full),
            },
            "full_add_rss_r4": {
                "kind": "add_coefficients",
                "directions": rank4,
                "coefficients": rss_coefficients(rank4, full),
            },
            "full_clamp_raw_r1": {
                "kind": "clamp",
                "directions": rank1,
                "deltas": {layer: dict(full) for layer in next(iter(rank1.values()))},
                "beta": beta,
            },
            "full_clamp_rss_r1": {
                "kind": "clamp",
                "directions": rank1,
                "deltas": rss_coefficients(rank1, full),
                "beta": beta,
            },
        }
    )
    for omitted in FEATURES:
        coefficients = {name: value for name, value in full.items() if name != omitted}
        deltas = rss_coefficients(rank1, coefficients)
        plans[f"loo_add_{omitted}"] = {
            "kind": "add_coefficients",
            "directions": rank1,
            "coefficients": deltas,
        }
        plans[f"loo_clamp_{omitted}"] = {
            "kind": "clamp",
            "directions": rank1,
            "deltas": deltas,
            "beta": beta,
        }
    for flipped in FEATURES:
        coefficients = {
            name: (-value if name == flipped else value) for name, value in full.items()
        }
        plans[f"flip_{flipped}"] = {
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


def selected(args: argparse.Namespace) -> tuple[float, float]:
    path = args.output_dir / "selection.json"
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        return float(value["scale"]), float(value["beta"])
    return 0.5, 0.5


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
    rank4 = load_directions(args, 4)
    plans = condition_plan(rank1, rank4, 0.5, 0.5)
    keep = {
        name: plans[name]
        for name in ("baseline", "full_add_rss_r1", "full_clamp_rss_r1")
    }
    row = jsonl(args.dev_prompts)[0]
    original = args.max_new_tokens
    args.max_new_tokens = min(16, original)
    run_rows(args, [row], keep, args.output_dir / "smoke.jsonl", "exp5:smoke")


def dev_phase(args: argparse.Namespace) -> None:
    rank1, rank4 = load_directions(args, 1), load_directions(args, 4)
    plans = {}
    for scale in (0.5, 0.75, 1.0):
        for beta in BETAS:
            candidate = condition_plan(rank1, rank4, scale, beta)
            name = f"s{scale:g}_b{beta:g}"
            plans[name] = candidate["full_clamp_rss_r1"]
    run_rows(
        args,
        jsonl(args.dev_prompts)[:32],
        plans,
        args.output_dir / "dev-generations.jsonl",
        "exp5:dev",
    )


def main_phase(args: argparse.Namespace) -> None:
    rank1, rank4 = load_directions(args, 1), load_directions(args, 4)
    scale, beta = selected(args)
    plans = condition_plan(rank1, rank4, scale, beta)
    rows = jsonl(args.test_prompts)[: args.test]
    if len(rows) != args.test:
        raise RuntimeError(f"need {args.test} test prompts, found {len(rows)}")
    run_rows(
        args,
        rows,
        plans,
        args.output_dir / "main-generations.jsonl",
        "exp5:main",
    )


def extension_phase(args: argparse.Namespace) -> None:
    rank1 = load_directions(args, 1)
    scale, beta = selected(args)
    strengths = {name: DEFAULT_STRENGTHS[name] * scale for name in FEATURES}
    plans = {}
    for size in (2, 3):
        for names in itertools.combinations(FEATURES, size):
            coefficients = {name: strengths[name] for name in names}
            deltas = rss_coefficients(rank1, coefficients)
            tag = "+".join(names)
            plans[f"add_{tag}"] = {
                "kind": "add_coefficients",
                "directions": rank1,
                "coefficients": deltas,
            }
            plans[f"clamp_{tag}"] = {
                "kind": "clamp",
                "directions": rank1,
                "deltas": deltas,
                "beta": beta,
            }
    run_rows(
        args,
        jsonl(args.test_prompts)[: args.test],
        plans,
        args.output_dir / "extension-generations.jsonl",
        "exp5:extension",
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
    assert len(condition_plan(toy, toy, 0.5, 0.5)) == 26
    print("five-concept clamp self-test passed")


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    {
        "self-test": lambda _: self_test(),
        "direction": direction_phase,
        "smoke": smoke_phase,
        "dev": dev_phase,
        "main": main_phase,
        "extension": extension_phase,
    }[args.phase](args)


if __name__ == "__main__":
    main()
