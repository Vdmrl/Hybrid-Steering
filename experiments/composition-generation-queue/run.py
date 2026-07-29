"""Resumable generation-only ablations for GDN composition."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import itertools
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).parents[2]
FEATURES = ("candor", "concrete", "casual", "optimism")
STRENGTHS = {"candor": 8.0, "concrete": 4.0, "casual": 1.0, "optimism": 2.0}
ACTIVATION_LAYERS = (10, 20, 30)
ACTIVATION_ALPHAS = (0.5, 1.0, 2.0, 4.0)
TARGET_SYSTEM = "Answer the user's question directly and naturally."
DONOR_SYSTEM = (
    "Read the following response in its situation. Preserve its behavioral "
    "stance and language in your internal state."
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module("composition_base", ROOT / "experiments/four-axis-night/run.py")
DIRECTION = load_module(
    "composition_direction", ROOT / "experiments/calm-french-composition/run.py"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("smoke", "svd", "activation", "joy", "norm"))
    parser.add_argument("--base-directions-dir", type=Path, required=True)
    parser.add_argument("--optimism-direction", type=Path, required=True)
    parser.add_argument("--candor-pairs", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--optimism-pairs", type=Path, required=True)
    parser.add_argument("--joy-pairs", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--test", type=int, default=128)
    parser.add_argument("--activation-test", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def load_direction(path: Path) -> dict[int, torch.Tensor]:
    return {
        int(name.removeprefix("layer_")): tensor.float()
        for name, tensor in load_file(path, device="cpu").items()
    }


def save_direction(path: Path, direction: dict[int, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {f"layer_{layer}": value.contiguous() for layer, value in direction.items()},
        str(path),
    )


def gdn_directions(args: argparse.Namespace) -> dict[str, dict[int, torch.Tensor]]:
    directions = {
        name: load_direction(args.base_directions_dir / f"{name}.safetensors")
        for name in FEATURES[:3]
    }
    directions["optimism"] = load_direction(args.optimism_direction)
    layer_sets = [set(value) for value in directions.values()]
    if any(layers != layer_sets[0] for layers in layer_sets[1:]):
        raise RuntimeError("GDN directions have different recurrent layers")
    return directions


def test_rows(args: argparse.Namespace, count: int | None = None) -> list[dict]:
    expected = count or args.test
    rows = BASE.jsonl(args.prompts)[:expected]
    if len(rows) != expected:
        raise RuntimeError(f"need {expected} prompts, found {len(rows)}")
    return rows


def truncate_tensor(value: torch.Tensor, rank: int) -> torch.Tensor:
    original_device = value.device
    matrices = value.squeeze(0).to("cuda", dtype=torch.float32)
    u, s, vh = torch.linalg.svd(matrices, full_matrices=False)
    result = (u[..., :rank] * s[..., :rank].unsqueeze(-2)) @ vh[..., :rank, :]
    return result.unsqueeze(0).to(original_device, dtype=torch.float32).cpu()


def truncate_direction(
    direction: dict[int, torch.Tensor], rank: int
) -> dict[int, torch.Tensor]:
    result = {}
    for index, layer in enumerate(sorted(direction), 1):
        result[layer] = truncate_tensor(direction[layer], rank)
        print(f"SVD rank={rank}: layer {index}/{len(direction)}", flush=True)
    return result


def energy_ratio(
    original: dict[int, torch.Tensor], approximation: dict[int, torch.Tensor]
) -> float:
    kept = sum(float(value.square().sum()) for value in approximation.values())
    total = sum(float(value.square().sum()) for value in original.values())
    return kept / max(total, 1e-12)


def scaled_sum(
    directions: dict[str, dict[int, torch.Tensor]], active: tuple[str, ...]
) -> dict[int, torch.Tensor]:
    return {
        layer: sum(
            (directions[name][layer] * STRENGTHS[name] for name in active[1:]),
            directions[active[0]][layer] * STRENGTHS[active[0]],
        )
        for layer in directions[active[0]]
    }


def cached_truncation(
    path: Path, direction: dict[int, torch.Tensor], rank: int
) -> dict[int, torch.Tensor]:
    if path.exists():
        return load_direction(path)
    result = truncate_direction(direction, rank)
    save_direction(path, result)
    return result


def svd_phase(args: argparse.Namespace) -> None:
    directions = gdn_directions(args)
    artifacts = args.output_dir / "directions"
    diagnostics: dict[str, Any] = {}
    low_rank: dict[int, dict[str, dict[int, torch.Tensor]]] = {1: {}, 4: {}}
    for rank in (1, 4):
        for name in FEATURES:
            path = artifacts / f"{name}-rank{rank}.safetensors"
            low_rank[rank][name] = cached_truncation(path, directions[name], rank)
            diagnostics[f"{name}_rank{rank}_energy"] = energy_ratio(
                directions[name], low_rank[rank][name]
            )

    all_sum = scaled_sum(directions, FEATURES)
    post = {}
    for rank in (1, 4):
        path = artifacts / f"all4-postsum-rank{rank}.safetensors"
        post[rank] = cached_truncation(path, all_sum, rank)
        diagnostics[f"all4_postsum_rank{rank}_energy"] = energy_ratio(
            all_sum, post[rank]
        )
    BASE.atomic_json(args.output_dir / "svd-diagnostics.json", diagnostics)

    conditions = {}
    for mask in range(1, 16):
        active = [name for index, name in enumerate(FEATURES) if mask & (1 << index)]
        conditions[f"per_r1_{mask:04b}"] = [
            (low_rank[1][name], STRENGTHS[name], None) for name in active
        ]
    for mask in (1, 2, 4, 8, 15):
        active = [name for index, name in enumerate(FEATURES) if mask & (1 << index)]
        conditions[f"per_r4_{mask:04b}"] = [
            (low_rank[4][name], STRENGTHS[name], None) for name in active
        ]
    conditions["post_r1_1111"] = [(post[1], 1.0, None)]
    conditions["post_r4_1111"] = [(post[4], 1.0, None)]

    tokenizer, model = BASE.load_model(args.model)
    BASE.run_records(
        model=model,
        tokenizer=tokenizer,
        rows=test_rows(args),
        conditions=conditions,
        path=args.output_dir / "svd.jsonl",
        max_new_tokens=args.max_new_tokens,
        prefix="svd",
    )


def decoder_layers(model: Any) -> Any:
    for path in (
        ("model", "language_model", "layers"),
        ("model", "layers"),
        ("language_model", "layers"),
    ):
        value = model
        try:
            for part in path:
                value = getattr(value, part)
            return value
        except AttributeError:
            pass
    raise RuntimeError("cannot locate decoder layers")


@torch.inference_mode()
def residual_for_text(
    model: Any, tokenizer: Any, text: str, layer_ids: tuple[int, ...]
) -> dict[int, torch.Tensor]:
    captured = {}
    layers = decoder_layers(model)
    handles = []
    for layer_id in layer_ids:

        def hook(_module: Any, _inputs: Any, output: torch.Tensor, i: int = layer_id):
            captured[i] = output[:, -1, :].detach().float().cpu()

        handles.append(layers[layer_id].register_forward_hook(hook))
    try:
        model(
            input_ids=DIRECTION.chat_ids(tokenizer, DONOR_SYSTEM, text),
            use_cache=False,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(layer_ids):
        raise RuntimeError("not all residual layers were captured")
    return captured


def pair_rows(args: argparse.Namespace, name: str) -> list[dict]:
    paths = {
        "candor": args.candor_pairs,
        "concrete": args.data_dir / "concrete" / "accepted.jsonl",
        "casual": args.data_dir / "casual" / "accepted.jsonl",
        "optimism": args.optimism_pairs,
    }
    rows = BASE.jsonl(paths[name])[:64]
    if len(rows) != 64:
        raise RuntimeError(f"{name}: need 64 donor pairs")
    if name == "candor":
        rows = [
            row
            | {
                "positive_text": row["candor_text"],
                "negative_text": row["sycophancy_text"],
            }
            for row in rows
        ]
    return rows


def build_residual_directions(
    args: argparse.Namespace, model: Any, tokenizer: Any
) -> dict[str, dict[int, torch.Tensor]]:
    path = args.output_dir / "directions" / "activation.safetensors"
    if path.exists():
        tensors = load_file(path, device="cpu")
        return {
            name: {
                layer: tensors[f"{name}_layer_{layer}"].float()
                for layer in ACTIVATION_LAYERS
            }
            for name in FEATURES
        }

    result = {}
    for name in FEATURES:
        sums: dict[int, torch.Tensor] = {}
        rows = pair_rows(args, name)
        for index, row in enumerate(rows, 1):
            negative = residual_for_text(
                model, tokenizer, row["negative_text"], ACTIVATION_LAYERS
            )
            positive = residual_for_text(
                model, tokenizer, row["positive_text"], ACTIVATION_LAYERS
            )
            for layer in ACTIVATION_LAYERS:
                delta = positive[layer] - negative[layer]
                sums[layer] = sums.get(layer, torch.zeros_like(delta)) + delta
            if index % 8 == 0:
                print(f"activation direction {name}: {index}/64", flush=True)
        result[name] = {layer: value.squeeze(0) / 64 for layer, value in sums.items()}

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            f"{name}_layer_{layer}": value.contiguous()
            for name, by_layer in result.items()
            for layer, value in by_layer.items()
        },
        str(path),
    )
    return result


@contextmanager
def activation_hook(model: Any, layer: int, direction: torch.Tensor, alpha: float):
    vector = direction.to("cuda", dtype=torch.float16)

    def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> torch.Tensor:
        return output + vector.to(output).view(1, 1, -1) * alpha

    handle = decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.inference_mode()
def activation_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    layer: int | None,
    direction: torch.Tensor | None,
    alpha: float,
    max_new_tokens: int,
) -> str:
    input_ids = BASE.chat_ids(tokenizer, prompt)
    context = (
        activation_hook(model, layer, direction, alpha)
        if layer is not None and direction is not None
        else nullcontext()
    )
    with context:
        output = model.generate(
            input_ids=input_ids,
            do_sample=False,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0, input_ids.shape[1] :], skip_special_tokens=True
    ).strip()


def activation_phase(args: argparse.Namespace) -> None:
    tokenizer, model = BASE.load_model(args.model)
    directions = build_residual_directions(args, model, tokenizer)
    rows = test_rows(args, args.activation_test)
    path = args.output_dir / "activation.jsonl"
    done = {row["task_id"] for row in BASE.jsonl(path)} if path.exists() else set()

    for index, row in enumerate(rows, 1):
        prompt = BASE.row_prompt(row)
        baseline_id = f"activation:{row['id']}:baseline"
        if baseline_id not in done:
            BASE.append_jsonl(
                path,
                {
                    "task_id": baseline_id,
                    "source_id": row["id"],
                    "scenario": prompt,
                    "baseline": activation_generate(
                        model, tokenizer, prompt, None, None, 0.0, args.max_new_tokens
                    ),
                },
            )
        for layer in ACTIVATION_LAYERS:
            for alpha in ACTIVATION_ALPHAS:
                task_id = f"activation:{row['id']}:l{layer}:a{alpha:g}"
                if task_id in done:
                    continue
                outputs = {}
                for name in FEATURES:
                    outputs[name] = activation_generate(
                        model,
                        tokenizer,
                        prompt,
                        layer,
                        directions[name][layer],
                        alpha,
                        args.max_new_tokens,
                    )
                all_four = sum(
                    (directions[name][layer] for name in FEATURES[1:]),
                    directions[FEATURES[0]][layer].clone(),
                )
                outputs["all4"] = activation_generate(
                    model,
                    tokenizer,
                    prompt,
                    layer,
                    all_four,
                    alpha,
                    args.max_new_tokens,
                )
                BASE.append_jsonl(
                    path,
                    {
                        "task_id": task_id,
                        "source_id": row["id"],
                        "scenario": prompt,
                        "layer": layer,
                        "alpha": alpha,
                        **outputs,
                    },
                )
                print(
                    f"activation: prompt {index}/{len(rows)} layer={layer} "
                    f"alpha={alpha:g}",
                    flush=True,
                )
                gc.collect()
                torch.cuda.empty_cache()


def joy_phase(args: argparse.Namespace) -> None:
    rows = BASE.jsonl(args.joy_pairs)
    if len(rows) < 64:
        raise RuntimeError("need 64 joy donor pairs")
    tokenizer, model = BASE.load_model(args.model)
    joy, _ = DIRECTION.ensure_direction(
        model=model,
        tokenizer=tokenizer,
        rows=rows[:64],
        name="joy",
        path=args.output_dir / "directions" / "joy.safetensors",
    )
    optimism = load_direction(args.optimism_direction)
    conditions = {}
    for alpha in (1.0, 2.0, 4.0, 8.0):
        conditions[f"joy_a{alpha:g}"] = [(joy, alpha, None)]
        conditions[f"joy_a{alpha:g}_optimism"] = [
            (joy, alpha, None),
            (optimism, STRENGTHS["optimism"], None),
        ]
    BASE.run_records(
        model=model,
        tokenizer=tokenizer,
        rows=test_rows(args),
        conditions=conditions,
        path=args.output_dir / "joy-optimism.jsonl",
        max_new_tokens=args.max_new_tokens,
        prefix="joy",
    )


def norm_phase(args: argparse.Namespace) -> None:
    directions = gdn_directions(args)
    masks = [
        sum(1 << FEATURES.index(name) for name in names)
        for names in itertools.combinations(FEATURES, 2)
    ]
    masks.append(15)
    conditions = {}
    for mask in masks:
        active = [name for index, name in enumerate(FEATURES) if mask & (1 << index)]
        conditions[f"norm_{mask:04b}"] = [
            (BASE.norm_controlled(directions, STRENGTHS, active), 1.0, None)
        ]
    tokenizer, model = BASE.load_model(args.model)
    BASE.run_records(
        model=model,
        tokenizer=tokenizer,
        rows=test_rows(args),
        conditions=conditions,
        path=args.output_dir / "norm-controlled.jsonl",
        max_new_tokens=args.max_new_tokens,
        prefix="norm",
    )


def smoke_phase(args: argparse.Namespace) -> None:
    directions = gdn_directions(args)
    first_layer = min(directions["candor"])
    rank_one = truncate_tensor(directions["candor"][first_layer], 1)
    if rank_one.shape != directions["candor"][first_layer].shape:
        raise RuntimeError("SVD changed the GDN tensor shape")

    tokenizer, model = BASE.load_model(args.model)
    prompt_row = test_rows(args, 1)[0]
    prompt = BASE.row_prompt(prompt_row)
    gdn_texts, _ = BASE.generate(
        model,
        tokenizer,
        prompt,
        {"gdn": [(directions["candor"], STRENGTHS["candor"], None)]},
        min(args.max_new_tokens, 16),
    )
    donor = pair_rows(args, "candor")[0]
    positive = residual_for_text(
        model, tokenizer, donor["positive_text"], (ACTIVATION_LAYERS[1],)
    )
    negative = residual_for_text(
        model, tokenizer, donor["negative_text"], (ACTIVATION_LAYERS[1],)
    )
    residual = positive[ACTIVATION_LAYERS[1]] - negative[ACTIVATION_LAYERS[1]]
    activation_text = activation_generate(
        model,
        tokenizer,
        prompt,
        ACTIVATION_LAYERS[1],
        residual.squeeze(0),
        1.0,
        min(args.max_new_tokens, 16),
    )
    BASE.atomic_json(
        args.output_dir / "smoke.json",
        {
            "source_id": prompt_row["id"],
            "gdn_text": gdn_texts["gdn"],
            "activation_text": activation_text,
            "rank_one_shape": list(rank_one.shape),
        },
    )
    print("GPU smoke passed", flush=True)


def self_test() -> None:
    assert len(list(itertools.combinations(FEATURES, 2))) == 6
    assert len(ACTIVATION_LAYERS) * len(ACTIVATION_ALPHAS) == 12
    toy = {name: {0: torch.eye(4).reshape(1, 1, 4, 4)} for name in FEATURES}
    assert scaled_sum(toy, FEATURES)[0].shape == (1, 1, 4, 4)
    print("composition generation queue self-test passed")


def main() -> None:
    args = arguments()
    if args.self_test:
        self_test()
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    {
        "smoke": smoke_phase,
        "svd": svd_phase,
        "activation": activation_phase,
        "joy": joy_phase,
        "norm": norm_phase,
    }[args.phase](args)


if __name__ == "__main__":
    main()
