"""Prepare directions and run the four-axis singleton screen."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).parents[2]
STYLE_RUNNER = ROOT / "experiments" / "style-singleton-screen" / "run.py"
FINAL_SCREEN = ROOT / "experiments" / "final-feature-screen" / "screen.py"
FINAL_DIR = FINAL_SCREEN.parent
GDN_LAYERS = (
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


def load_module(name: str, path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_passive(style_root: Path, output: Path, count: int) -> None:
    target = style_root / "ATP" / "train.tsv"
    if not target.exists():
        subprocess.run(
            [sys.executable, str(style_root / "single_transform_checkout.py"), "ATP"],
            cwd=style_root,
            check=True,
        )
    runner = load_module("style_runner", STYLE_RUNNER)
    pairs = runner.load_style_pairs(style_root.parent, "ATP", count)
    write_json(
        output / "data" / "passive_voice_pairs.json",
        {
            "feature": "passive_voice",
            "source": "StylePTB ATP",
            "pairs": pairs,
        },
    )


def low_rank(direction: dict[int, torch.Tensor], rank: int) -> dict[int, torch.Tensor]:
    result = {}
    for layer, value in direction.items():
        matrix = value.squeeze(0).float()
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        k = min(rank, len(s))
        result[layer] = (
            (u[..., :k] * s[..., :k].unsqueeze(-2)) @ vh[..., :k, :]
        ).unsqueeze(0)
    return result


def prepare_russian(source: Path, output: Path) -> None:
    with np.load(source) as archive:
        raw = archive["deltaS"]
    if raw.shape[0] != len(GDN_LAYERS):
        raise ValueError(f"expected {len(GDN_LAYERS)} layers, got {raw.shape}")
    direction = {
        layer: torch.from_numpy(raw[index]).float()
        for index, layer in enumerate(GDN_LAYERS)
    }
    if next(iter(direction.values())).ndim == 2:
        direction = {layer: value.unsqueeze(0) for layer, value in direction.items()}
    directions = output / "directions"
    directions.mkdir(parents=True, exist_ok=True)
    for name, values in (("full", direction), ("rank4", low_rank(direction, 4))):
        save_file(
            {f"layer_{layer}": value.contiguous() for layer, value in values.items()},
            directions / f"russian_language-{name}.safetensors",
        )


def run_screen(args: argparse.Namespace) -> None:
    screen = load_module("final_screen", FINAL_SCREEN)
    screen.run(
        out=args.output,
        model_id=args.model,
        feature=args.feature,
        rank=args.rank,
        alpha=args.alpha,
        max_new_tokens=args.max_new_tokens,
        prompts_file=args.prompts,
        limit=args.limit,
        tag=args.tag,
        clamp_beta=args.clamp_beta,
        direction_path=args.direction,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    passive = sub.add_parser("prepare-passive")
    passive.add_argument("--style-root", type=Path, required=True)
    passive.add_argument("--output", type=Path, required=True)
    passive.add_argument("--count", type=int, default=128)

    russian = sub.add_parser("prepare-russian")
    russian.add_argument("--source", type=Path, required=True)
    russian.add_argument("--output", type=Path, required=True)

    screen = sub.add_parser("screen")
    screen.add_argument("--output", type=Path, required=True)
    screen.add_argument("--feature", required=True)
    screen.add_argument("--rank", choices=("full", "rank1", "rank4"), required=True)
    screen.add_argument("--alpha", type=float, required=True)
    screen.add_argument("--direction", type=Path)
    screen.add_argument("--prompts", type=Path)
    screen.add_argument("--limit", type=int, default=4)
    screen.add_argument("--tag", default="singleton")
    screen.add_argument("--clamp-beta", type=float, default=1.0)
    screen.add_argument("--max-new-tokens", type=int, default=128)
    screen.add_argument("--model", default="Qwen/Qwen3.5-9B")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare-passive":
        prepare_passive(args.style_root, args.output, args.count)
    elif args.command == "prepare-russian":
        prepare_russian(args.source, args.output)
    else:
        run_screen(args)


if __name__ == "__main__":
    main()
