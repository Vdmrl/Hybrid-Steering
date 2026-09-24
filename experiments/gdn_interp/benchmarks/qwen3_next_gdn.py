#!/usr/bin/env python
"""Benchmark Qwen3-Next Gated Delta Rule reference PyTorch vs FLA kernels.

Run: uv run python benchmarks/qwen3_next_gdn.py
"""

import argparse
import statistics
import time

import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from transformers.models.qwen3_next.modeling_qwen3_next import (
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)


def measure(name, fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return name, statistics.median(samples)


def run(fn, q, k, v, g, beta):
    return fn(q, k, v, g, beta, initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--hub", action="store_true", help="also load kernels-community/fla via the kernels package")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires CUDA.")

    torch.manual_seed(0)
    shape = (1, args.seq_len, args.heads, args.head_dim)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(shape[:-1], device="cuda", dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(shape[:-1], device="cuda", dtype=torch.float32))

    cases = (
        ("prefill / reference torch", torch_chunk_gated_delta_rule, (q, k, v, g, beta)),
        ("prefill / FLA chunk", chunk_gated_delta_rule, (q, k, v, g, beta)),
        ("decode T=1 / reference torch", torch_recurrent_gated_delta_rule, (q[:, :1], k[:, :1], v[:, :1], g[:, :1], beta[:, :1])),
        ("decode T=1 / FLA fused recurrent", fused_recurrent_gated_delta_rule, (q[:, :1], k[:, :1], v[:, :1], g[:, :1], beta[:, :1])),
    )
    # The model uses this mode: normalization happens inside the selected implementation.
    for _, reference, optimized, inputs in (
        ("prefill", torch_chunk_gated_delta_rule, chunk_gated_delta_rule, (q, k, v, g, beta)),
        ("decode", torch_recurrent_gated_delta_rule, fused_recurrent_gated_delta_rule, (q[:, :1], k[:, :1], v[:, :1], g[:, :1], beta[:, :1])),
    ):
        torch.testing.assert_close(run(reference, *inputs)[0], run(optimized, *inputs)[0], rtol=0.2, atol=0.03)
    if args.hub:
        try:
            from kernels import get_kernel
        except ImportError as error:
            raise SystemExit("Install temporarily with: uv run --with kernels python benchmarks/qwen3_next_gdn.py --hub") from error
        hub = get_kernel("kernels-community/fla", version=1)
        hub_cases = (
            ("prefill / Hub FLA", hub.chunk_gated_delta_rule, (q, k, v, g, beta)),
            ("decode T=1 / Hub FLA fused recurrent", hub.fused_recurrent_gated_delta_rule, (q[:, :1], k[:, :1], v[:, :1], g[:, :1], beta[:, :1])),
        )
        for _, implementation, inputs in hub_cases:
            reference = torch_recurrent_gated_delta_rule if inputs[0].shape[1] == 1 else torch_chunk_gated_delta_rule
            torch.testing.assert_close(run(reference, *inputs)[0], run(implementation, *inputs)[0], rtol=0.2, atol=0.03)
        cases += hub_cases
    results = [measure(name, lambda fn=fn, x=x: run(fn, *x), args.warmup, args.repeats) for name, fn, x in cases]
    baseline = dict(results)
    print(f"GPU: {torch.cuda.get_device_name()} | shape: {shape} | bf16 | median of {args.repeats}")
    for name, ms in results:
        reference = baseline["prefill / reference torch"] if name.startswith("prefill") else baseline["decode T=1 / reference torch"]
        print(f"{name:38} {ms:9.3f} ms  {reference / ms:6.2f}x vs reference")


if __name__ == "__main__":
    main()
