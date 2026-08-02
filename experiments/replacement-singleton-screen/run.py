"""Batched singleton screen for persuasive, confidence, and past tense."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).parents[2]
STYLE_RUNNER = ROOT / "experiments" / "style-singleton-screen" / "run.py"
FINAL_SCREEN = ROOT / "experiments" / "final-feature-screen" / "screen.py"


def module(name: str, path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


runner = module("style_runner", STYLE_RUNNER)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def batched_chat(tokenizer: Any, prompts: list[str], system: str) -> dict[str, Any]:
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        for p in prompts
    ]
    tokenizer.padding_side = "left"
    return {
        key: value.to("cuda")
        for key, value in tokenizer(
            rendered, padding=True, return_tensors="pt", add_special_tokens=False
        ).items()
    }


@torch.inference_mode()
def rewrite(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    style: str,
    batch_size: int,
) -> list[str]:
    result = []
    system = (
        f"Rewrite the text in a {style} style. Preserve every fact, meaning, "
        "approximate length, language, and formatting. Return only the rewrite."
    )
    for start in range(0, len(texts), batch_size):
        inputs = batched_chat(tokenizer, texts[start : start + batch_size], system)
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=160,
            pad_token_id=tokenizer.eos_token_id,
        )
        width = inputs["input_ids"].shape[1]
        result.extend(
            tokenizer.batch_decode(outputs[:, width:], skip_special_tokens=True)
        )
        print(f"rewrite {style}: {min(start + batch_size, len(texts))}/{len(texts)}")
    return [text.strip() for text in result]


def prepare_confidence(args: argparse.Namespace) -> None:
    frame = pd.read_parquet(args.source)
    texts = [
        str(text).strip()
        for text in frame["input_text"].dropna().drop_duplicates()
        if 35 <= len(str(text).split()) <= 140
    ][: args.count]
    if len(texts) < args.count:
        raise RuntimeError(f"only {len(texts)} source texts")
    tokenizer, model = runner.load_model(args.model)
    confident = rewrite(model, tokenizer, texts, "clear and confident", args.batch)
    hedged = rewrite(
        model,
        tokenizer,
        texts,
        "cautious and explicitly hedged, without refusing to answer",
        args.batch,
    )
    pairs = [
        {
            "source_id": f"synthetic:{index}",
            "positive_text": positive,
            "negative_text": negative,
        }
        for index, (positive, negative) in enumerate(
            zip(confident, hedged, strict=True)
        )
        if positive and negative and positive != negative
    ]
    write_json(
        args.output / "data" / "confident_pairs.json",
        {"feature": "confident", "pairs": pairs},
    )


def repair_confidence(args: argparse.Namespace) -> None:
    old = json.loads(args.source_pairs.read_text(encoding="utf-8"))["pairs"]
    positive = [row["positive_text"] for row in old[: args.candidates]]
    tokenizer, model = runner.load_model(args.model)
    negative = rewrite(
        model,
        tokenizer,
        positive,
        (
            "cautious and hedged using may, might, likely, perhaps, or appears. "
            "Make minimal word-level edits only; preserve sentence count, length, "
            "formality, emotion, voice, vocabulary, punctuation, and formatting"
        ),
        args.batch,
    )
    pairs = []
    for index, (target, opposite) in enumerate(zip(positive, negative, strict=True)):
        ratio = len(opposite.split()) / max(len(target.split()), 1)
        hedge = re.search(
            r"\b(may|might|likely|perhaps|possibly|appears?|seems?)\b",
            opposite,
            re.IGNORECASE,
        )
        if not 0.9 <= ratio <= 1.1 or not hedge:
            continue
        pairs.append(
            {
                "source_id": f"minimal:{index}",
                "positive_text": target,
                "negative_text": opposite,
                "length_ratio": round(ratio, 3),
            }
        )
        if len(pairs) == args.count:
            break
    if len(pairs) < args.count:
        raise RuntimeError(f"only {len(pairs)}/{args.count} clean pairs")
    write_json(
        args.output / "data" / "confident_pairs.json",
        {"feature": "confident", "pair_version": "minimal-v1", "pairs": pairs},
    )


def prepare_past(args: argparse.Namespace) -> None:
    target = args.style_root / "TPA" / "train.tsv"
    if not target.exists():
        subprocess.run(
            [
                sys.executable,
                str(args.style_root / "single_transform_checkout.py"),
                "TPA",
            ],
            cwd=args.style_root,
            check=True,
        )
    pairs = runner.load_style_pairs(args.style_root.parent, "TPA", args.count)
    write_json(
        args.output / "data" / "past_tense_pairs.json",
        {"feature": "past_tense", "source": "StylePTB TPA", "pairs": pairs},
    )


@torch.inference_mode()
def build_direction(args: argparse.Namespace) -> None:
    payload = json.loads(
        (args.output / "data" / f"{args.feature}_pairs.json").read_text()
    )
    pairs = payload["pairs"]
    tokenizer, model = runner.load_model(args.model)
    sums: dict[int, torch.Tensor] = {}
    count = 0
    system = "Preserve this response's style in your internal state."
    for start in range(0, len(pairs), args.batch):
        batch = pairs[start : start + args.batch]
        texts = [p["positive_text"] for p in batch] + [
            p["negative_text"] for p in batch
        ]
        cache = model(
            **batched_chat(tokenizer, texts, system),
            use_cache=True,
            return_dict=True,
        ).past_key_values
        states = runner.extract_recurrent(cache)
        size = len(batch)
        for layer, state in states.items():
            if state.shape[0] != size * 2:
                raise RuntimeError(f"unexpected batch shape {state.shape}")
            delta = (state[:size] - state[size:]).sum(dim=0, keepdim=True)
            sums[layer] = sums.get(layer, torch.zeros_like(delta)) + delta
        count += size
        print(f"direction {args.feature}: {count}/{len(pairs)}", flush=True)
    full = {layer: value / count for layer, value in sums.items()}
    direction_dir = args.output / "directions"
    direction_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (
        ("full", full),
        ("rank1", runner.low_rank(full, 1)),
        ("rank4", runner.low_rank(full, 4)),
    ):
        save_file(
            {f"layer_{layer}": value.contiguous() for layer, value in values.items()},
            direction_dir / f"{args.feature}-{name}.safetensors",
        )


def screen(args: argparse.Namespace) -> None:
    screen_module = module("final_screen", FINAL_SCREEN)
    tokenizer, model = runner.load_model(args.model)
    screen_module.runner.load_model = lambda _model: (tokenizer, model)
    for item in args.spec:
        rank, alpha_text = item.split(":", 1)
        alpha = float(alpha_text)
        direction = args.direction or (
            args.output / "directions" / f"{args.feature}-{rank}.safetensors"
        )
        screen_module.run(
            out=args.output,
            model_id=args.model,
            feature=args.feature,
            rank=rank,
            alpha=alpha,
            max_new_tokens=128,
            prompts_file=args.prompts,
            limit=args.limit,
            tag=args.tag,
            clamp_beta=1.0,
            direction_path=direction,
        )


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (
        "prepare-confidence",
        "repair-confidence",
        "prepare-past",
        "build-direction",
        "screen",
    ):
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--model", default="Qwen/Qwen3.5-9B")
        command.add_argument("--count", type=int, default=128)
        command.add_argument("--batch", type=int, default=8)
        if name == "prepare-confidence":
            command.add_argument("--source", type=Path, required=True)
        elif name == "repair-confidence":
            command.add_argument("--source-pairs", type=Path, required=True)
            command.add_argument("--candidates", type=int, default=48)
        elif name == "prepare-past":
            command.add_argument("--style-root", type=Path, required=True)
        elif name == "build-direction":
            command.add_argument("--feature", required=True)
        else:
            command.add_argument("--feature", required=True)
            command.add_argument("--spec", action="append", required=True)
            command.add_argument("--direction", type=Path)
            command.add_argument("--prompts", type=Path, required=True)
            command.add_argument("--limit", type=int, default=4)
            command.add_argument("--tag", default="screen")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.command == "prepare-confidence":
        prepare_confidence(args)
    elif args.command == "repair-confidence":
        repair_confidence(args)
    elif args.command == "prepare-past":
        prepare_past(args)
    elif args.command == "build-direction":
        build_direction(args)
    else:
        screen(args)


if __name__ == "__main__":
    main()
