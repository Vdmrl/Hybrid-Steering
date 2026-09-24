"""Smoke-test and benchmark generation with and without tracing on a real model.

Run: uv run python -m benchmarks.gdn_steering
"""

import argparse
import statistics
import time
from collections.abc import Callable
from typing import Any, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedTokenizer

from gdn_interp import GDNTrace, GDNRunner, gdn_layers

PROMPTS = (
    "Explain why leaves change color in autumn.",
    "Give a concise recipe for vegetable soup.",
    "What makes a bicycle remain balanced while moving?",
)


def prompts(batch_size: int) -> list[str]:
    return [f"{PROMPTS[index % len(PROMPTS)]} {'Additional context. ' * (index % 8)}" for index in range(batch_size)]


def measure(fn: Callable[[], object], warmup: int, repeats: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def unit_deltas(model: Any, layers: list[int]) -> dict[int, torch.Tensor]:
    deltas = {}
    for layer in layers:
        module = model.model.layers[layer].linear_attn
        delta = torch.randn(module.num_v_heads, module.head_k_dim, module.head_v_dim, device=next(module.parameters()).device)
        deltas[layer] = delta / delta.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return deltas


def mean_delta_output_norm(baseline: Any, steered: Any) -> float:
    norms = [
        torch.linalg.vector_norm(steered.outputs[position][layer] - baseline.outputs[position][layer], dim=-1).float().mean()
        for position in baseline.outputs.keys() & steered.outputs.keys()
        if torch.equal(baseline.indices[position], steered.indices[position])
        for layer in baseline.outputs[position].keys() & steered.outputs[position].keys()
    ]
    return torch.stack(norms).mean().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--modes", choices=("exact", "chunk"), nargs="+", default=("exact", "chunk"))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    tokenizer = cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(args.model))

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    layers = gdn_layers(model)[: args.layers]

    deltas = unit_deltas(model, layers)
    runner = GDNRunner(model, tokenizer, layers, deltas, normalize=False)
    batch = prompts(args.batch_size)

    with torch.inference_mode():
        baseline = runner.forward(batch, scale=0, prompt_steer_position=0, prompt_steer_mode="exact")
        steered = runner.forward(batch, prompt_steer_position=0, prompt_steer_mode="exact")

        mean_norm = mean_delta_output_norm(baseline, steered)

        for mode in args.modes:
            runner_tokens = runner.generate(batch, prompt_steer_mode=mode, max_new_tokens=args.max_new_tokens)
            trace = GDNTrace()
            traced_tokens = runner.generate(batch, prompt_steer_mode=mode, max_new_tokens=args.max_new_tokens, trace=trace)
            torch.testing.assert_close(runner_tokens, traced_tokens, atol=0, rtol=0)

            assert runner_tokens.shape == (len(batch), args.max_new_tokens)
            assert traced_tokens.shape == (len(batch), args.max_new_tokens)
    assert trace.outputs and trace.states and mean_norm > 0

    print(f"model={args.model} batch={len(batch)} layers={layers} generated={args.max_new_tokens}")
    print(f"mean ||delta_o||={mean_norm:.6g}")

    for mode in args.modes:
        runner_ms = measure(
            lambda: runner.generate(batch, prompt_steer_mode=mode, max_new_tokens=args.max_new_tokens),
            args.warmup,
            args.repeats,
        )
        traced_ms = measure(
            lambda: runner.generate(batch, prompt_steer_mode=mode, max_new_tokens=args.max_new_tokens, trace=GDNTrace()),
            args.warmup,
            args.repeats,
        )
        print(f"mode={mode}: runner={runner_ms:.2f} ms traced={traced_ms:.2f} ms ({traced_ms / runner_ms:.2f}x)")


if __name__ == "__main__":
    main()
