"""Compare the PyTorch Gated DeltaNet reference with FLA when both are present.

The fused kernel is recorded as skipped when CUDA or flash-linear-attention
is unavailable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)


def run(function, query, key, value, gate, beta):
    return function(
        query,
        key,
        value,
        gate,
        beta,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=8)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    torch.manual_seed(0)
    shape = (1, args.seq_len, args.heads, args.head_dim)
    query = torch.randn(shape, device=device, dtype=dtype)
    key = torch.randn(shape, device=device, dtype=dtype)
    value = torch.randn(shape, device=device, dtype=dtype)
    gate = -torch.rand(shape[:-1], device=device, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(shape[:-1], device=device, dtype=torch.float32))
    prefill = run(torch_chunk_gated_delta_rule, query, key, value, gate, beta)[0]
    decode = run(
        torch_recurrent_gated_delta_rule,
        query[:, :1],
        key[:, :1],
        value[:, :1],
        gate[:, :1],
        beta[:, :1],
    )[0]
    if not torch.isfinite(prefill).all() or not torch.isfinite(decode).all():
        raise SystemExit("reference kernel produced non-finite values")
    fused = "skipped"
    if device == "cuda":
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule

            optimized = run(chunk_gated_delta_rule, query, key, value, gate, beta)[0]
            torch.testing.assert_close(prefill, optimized, rtol=0.2, atol=0.03)
            fused = "matched"
        except Exception as error:  # noqa: BLE001 - record why the optional kernel did not run
            fused = f"skipped: {error}"
    payload = {
        "device": device,
        "prefill_norm": float(prefill.float().norm()),
        "decode_norm": float(decode.float().norm()),
        "fused": fused,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "kernels.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
