"""Real two-GPU smoke for Theistic direction extraction and steering."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

from hybrid_steering.direction import collect_from_pairs, read_pairs, target_and_source
from hybrid_steering.mamba import MambaRunner
from hybrid_steering.runner import Runner
from hybrid_steering.runtime import chat_prompts, load_runtime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/pipeline"))
from residual import ResidualRunner, collect_residual  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--backend", choices=("tp", "layers"), required=True)
    args = parser.parse_args()
    os.environ["HYBRID_PARALLEL_BACKEND"] = args.backend
    started = time.monotonic()
    model, tokenizer = load_runtime(args.model)
    examples = [target_and_source(row) for row in read_pairs(args.pairs)[:2]]
    direction = collect_from_pairs(model, tokenizer, examples, batch_size=1)
    text = chat_prompts(tokenizer, ["What is two plus two?"])
    if args.backend == "tp":
        runner = Runner.from_direction(
            model, tokenizer, direction.delta, rank=1, normalize=False, intervention="clamp"
        )
    else:
        runner = MambaRunner(
            model,
            tokenizer,
            direction.delta,
            rank=1,
            normalize=False,
            intervention="clamp",
            mean_target=direction.mean_target,
        )
    tokens = runner.generate(text, scale=0.75, max_new_tokens=16)
    if tokens.shape != (1, 16) or not tokenizer.decode(tokens[0], skip_special_tokens=True).strip():
        raise RuntimeError("recurrent steering smoke produced no answer")
    residual = collect_residual(model, tokenizer, examples[:1], batch_size=1)
    residual_tokens = ResidualRunner(model, tokenizer, residual[16], 16).generate(
        text, scale=0.1, max_new_tokens=16
    )
    if residual_tokens.shape != (1, 16):
        raise RuntimeError("residual steering smoke returned a wrong shape")
    digest = hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest()
    if dist.is_initialized():
        digests = [None] * dist.get_world_size()
        dist.all_gather_object(digests, digest)
        if len(set(digests)) != 1:
            raise RuntimeError("TP ranks disagree on smoke output")
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(
            json.dumps(
                {
                    "backend": args.backend,
                    "model": args.model,
                    "direction_layers": len(direction.delta),
                    "seconds": round(time.monotonic() - started, 2),
                    "peak_mib": round(torch.cuda.max_memory_reserved() / 1024**2),
                    "response": tokenizer.decode(tokens[0], skip_special_tokens=True)[:120],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
