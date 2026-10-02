"""How the direction's rank and orientation settle as pairs are added.

Every text's final recurrent state is computed once. Then, for each random
permutation of the pool, the target mean, the source mean and their difference
are taken over the first N texts, for N on a fixed checkpoint grid, and two
things are measured at each N:

- rank of each 128x128 head matrix, averaged over all GDN layers and heads:
  ``stable`` (sum sigma^2 / sigma_1^2), ``effective`` (exp of the entropy of the
  energy shares sigma_i^2 / sum sigma^2), ``rank90`` and ``rank95`` (singular
  values needed for 90 / 95 percent of the energy). ``effective`` here is the
  energy version; ``hybrid_steering.state.effective_rank`` uses sigma itself, so
  the two are not interchangeable.
- ``cosine_to_full``: per head, the cosine between the flattened matrix at N and
  the same permutation's matrix at the largest N, averaged over heads.

Target and source texts are permuted independently, as in the runs behind
``results/``: the pairs there are two independently generated pools, so a pair
index carries no alignment. Rows are written to ``rows.jsonl``; per-text states
go to ``states_target.npy`` and ``states_source.npy`` (float16, large: about
19 GB per pole for 750 texts on Qwen3.5-9B) so a rerun with other
permutations does not repeat the forward passes.

    uv run python experiments/pair-count/run.py --concept theism \\
        --source atheism --target theism --output runs/pair-count/theism
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from hybrid_steering import Runner, final_states, gdn_layers, load_runtime
from hybrid_steering.direction import (
    concept_sides,
    load_concept_pairs,
    read_pairs,
    target_and_source,
)
from hybrid_steering.runtime import batched, write_jsonl

CHECKPOINTS = (1, 2, 3, 5, 8, 12, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 750)
RANKS = ("stable", "effective", "rank90", "rank95")


def spectrum_metrics(heads: torch.Tensor) -> dict[str, float]:
    """Rank metrics of ``[n, key, value]`` head matrices, averaged over heads."""
    energy = torch.linalg.svdvals(heads.float()).pow(2)
    total = energy.sum(-1).clamp_min(torch.finfo(energy.dtype).tiny)
    share = energy / total[:, None]
    cumulative = share.cumsum(-1)
    return {
        "stable": float((total / energy[:, 0].clamp_min(torch.finfo(energy.dtype).tiny)).mean()),
        "effective": float(torch.exp(-torch.special.xlogy(share, share).sum(-1)).mean()),
        "rank90": float(((cumulative < 0.90).sum(-1) + 1).float().mean()),
        "rank95": float(((cumulative < 0.95).sum(-1) + 1).float().mean()),
    }


def cosine_to(heads: torch.Tensor, reference: torch.Tensor) -> float:
    """Per-head cosine of flattened matrices, averaged over heads."""
    return float(
        torch.nn.functional.cosine_similarity(
            heads.flatten(1).float(), reference.flatten(1).float(), dim=1
        ).mean()
    )


def store_states(runner: Runner, texts: list[str], batch_size: int, path: Path) -> np.ndarray:
    """Final state of every text as ``[text, layer, head, key, value]`` float16; NaN rows stay NaN."""
    layers = runner.layers
    array = None
    for start, chunk in zip(
        range(0, len(texts), batch_size), batched(texts, batch_size), strict=True
    ):
        states = final_states(runner, chunk)
        stacked = torch.stack([states[layer] for layer in layers], dim=1).float().cpu()
        if array is None:
            array = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float16, shape=(len(texts), *stacked.shape[1:])
            )
        array[start : start + len(chunk)] = stacked.numpy().astype(np.float16)
    array.flush()
    return array


def sweep(target: np.ndarray, source: np.ndarray, checkpoints: list[int], permutations: int):
    valid_target = np.where(np.isfinite(target).reshape(len(target), -1).all(1))[0]
    valid_source = np.where(np.isfinite(source).reshape(len(source), -1).all(1))[0]
    pool = min(len(valid_target), len(valid_source))
    grid = [n for n in checkpoints if n <= pool]
    rows = []
    for permutation in range(permutations):
        rng = np.random.default_rng(permutation)
        order_target = rng.permutation(valid_target)
        order_source = rng.permutation(valid_source)
        running_target = torch.zeros(target.shape[1:], dtype=torch.float32)
        running_source = torch.zeros(source.shape[1:], dtype=torch.float32)
        matrices: dict[str, list[torch.Tensor]] = {"target": [], "source": [], "delta": []}
        for count in range(1, grid[-1] + 1):
            running_target += torch.from_numpy(target[order_target[count - 1]].astype(np.float32))
            running_source += torch.from_numpy(source[order_source[count - 1]].astype(np.float32))
            if count in grid:
                mean_target, mean_source = running_target / count, running_source / count
                for name, matrix in (
                    ("target", mean_target),
                    ("source", mean_source),
                    ("delta", mean_target - mean_source),
                ):
                    matrices[name].append(matrix.reshape(-1, *matrix.shape[-2:]).clone())
        for name, series in matrices.items():
            reference = series[-1]
            for pairs, heads in zip(grid, series, strict=True):
                rows.append(
                    {
                        "permutation": permutation,
                        "pairs": pairs,
                        "matrix": name,
                        **spectrum_metrics(heads),
                        "cosine_to_full": cosine_to(heads, reference),
                    }
                )
        print(f"permutation {permutation + 1}/{permutations}", flush=True)
    return rows, len(valid_target), len(valid_source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--concept", required=True, help="Hub dataset directory")
    parser.add_argument("--jsonl", type=Path, help="local pairs; skips the Hub download")
    parser.add_argument("--source", help="source name when the concept slug has no hyphen")
    parser.add_argument("--target", help="target name when the concept slug has no hyphen")
    parser.add_argument("--pairs", type=int, default=750, help="size of the pool")
    parser.add_argument("--checkpoints", type=int, nargs="+", default=list(CHECKPOINTS))
    parser.add_argument("--permutations", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    source_name, target_name = concept_sides(args.concept, args.source, args.target)
    rows = read_pairs(args.jsonl) if args.jsonl else load_concept_pairs(args.concept)
    pairs = [target_and_source(row) for row in rows[: args.pairs]]
    model, tokenizer = load_runtime(args.model)
    runner = Runner(model, tokenizer, gdn_layers(model), normalize=False)
    args.output.mkdir(parents=True, exist_ok=True)
    target = store_states(
        runner, [t for t, _ in pairs], args.batch_size, args.output / "states_target.npy"
    )
    source = store_states(
        runner, [s for _, s in pairs], args.batch_size, args.output / "states_source.npy"
    )
    measured, n_target, n_source = sweep(target, source, args.checkpoints, args.permutations)
    for row in measured:
        row.update(concept=args.concept, target=target_name, source=source_name)
    write_jsonl(args.output / "rows.jsonl", measured)
    skipped = (len(pairs) - n_target, len(pairs) - n_source)
    print(f"wrote {len(measured)} rows; non-finite states skipped: {skipped}", flush=True)


if __name__ == "__main__":
    main()
