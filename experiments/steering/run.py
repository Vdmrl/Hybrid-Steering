"""Extract a target-minus-source direction and generate with it.

Smoke uses the random tiny model and two hand-written pairs. A real run passes
``--model Qwen/Qwen3.5-9B`` and a larger ``--pairs`` file is not required:
pass ``--target-texts`` and ``--source-texts`` as repeated arguments, or rely
on the built-in pairs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hybrid_steering import (
    DirectionManifest,
    FrobeniusDelta,
    Runner,
    collect_direction,
    concept_detector,
    gdn_layers,
    load_direction,
    load_runtime,
    save_direction,
    truncate_direction,
)

PAIRS = (
    ("Это русский текст про дождь и город.", "This is English text about rain and a city."),
    ("Завтра будет холодно и ветрено у моря.", "Tomorrow will be cold and windy by the sea."),
)
PROMPTS = (
    "Describe the weather tomorrow.",
    "What happens if the last bus is missed?",
)


def extract(args: argparse.Namespace, output: Path) -> None:
    model, tokenizer = load_runtime(args.model)
    layers = gdn_layers(model)
    runner = Runner(model, tokenizer, layers, normalize=False)
    collected = collect_direction(
        runner, PAIRS[: args.pairs], batch_size=args.batch_size, metrics=(FrobeniusDelta(),)
    )
    manifest = DirectionManifest(
        model_id=args.model,
        target=args.target,
        source=args.source,
        example_ids=[f"pair-{index}" for index in range(collected.pairs)],
        decoder_layer_indices=sorted(collected.delta),
        state_shapes={layer: list(tensor.shape) for layer, tensor in collected.delta.items()},
    )
    save_direction(
        output,
        collected.delta,
        manifest,
        mean_target=collected.mean_target,
        mean_source=collected.mean_source,
    )
    norms = {
        str(layer): float(torch.linalg.matrix_norm(tensor.float(), dim=(-2, -1)).mean())
        for layer, tensor in collected.delta.items()
    }
    (output / "summary.json").write_text(
        json.dumps({"pairs": collected.pairs, "frobenius": norms}, indent=2) + "\n"
    )


def generate(args: argparse.Namespace, direction_dir: Path, output: Path) -> None:
    direction, manifest, _, _ = load_direction(direction_dir)
    model, tokenizer = load_runtime(args.model)
    device = next(model.parameters()).device
    deltas = truncate_direction(direction, args.rank)
    deltas = {layer: tensor.to(device) for layer, tensor in deltas.items()}
    runner = Runner(model, tokenizer, sorted(deltas), deltas, normalize=args.normalize)
    detector = concept_detector(args.language)
    rows = []
    for scale in args.scales:
        tokens = runner.generate(
            list(PROMPTS[: args.questions]),
            scale=scale,
            prompt_position=args.position,
            max_new_tokens=args.max_new_tokens,
        )
        for index, row_tokens in enumerate(tokens):
            text = tokenizer.decode(row_tokens, skip_special_tokens=True)
            rows.append(
                {
                    "scale": scale,
                    "prompt": PROMPTS[index],
                    "response": text,
                    "language": detector.label(text),
                    "target_language": detector.detects(text),
                    "direction": f"{manifest.target} - {manifest.source}",
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    (output / "generations.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--target", default="russian")
    parser.add_argument("--source", default="english")
    parser.add_argument("--language", default="ru")
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--questions", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.model = "tiny"
        args.pairs = min(args.pairs, len(PAIRS))
        args.questions = 1
        args.scales = [0.0, 0.5]
        args.max_new_tokens = 2
        args.rank = 1
    if not 1 <= args.pairs <= len(PAIRS):
        parser.error(f"--pairs must be in [1, {len(PAIRS)}]")
    args.output.mkdir(parents=True, exist_ok=True)
    direction_dir = args.output / "direction"
    extract(args, direction_dir)
    generate(args, direction_dir, args.output / "generate")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
