"""Rank of the target-minus-source direction across many concepts.

For each concept, the direction is built with ``collect_direction`` from its
first ``--pairs`` pairs, and each 128x128 head matrix of the target mean, the
source mean and the difference gets four rank measures, averaged over all GDN
layers and heads: ``stable`` (sum sigma^2 / sigma_1^2), ``effective`` (exp of
the entropy of the energy shares sigma_i^2 / sum sigma^2; not the same as
``hybrid_steering.state.effective_rank``, which uses sigma), ``rank90`` and
``rank95`` (singular values needed for 90 / 95 percent of the energy).

Concepts come from a manifest, one JSON object per line with ``concept`` (the
Hub directory) and optional ``class_name``, ``target``, ``source``;
``results/concepts.jsonl`` is the list behind the committed results. Rows are
appended to ``rows.jsonl`` one concept at a time, and a rerun skips concepts
already there, so a long run can be stopped and resumed.

    uv run python experiments/concept-ranks/run.py \\
        --manifest experiments/concept-ranks/results/concepts.jsonl --output runs/concept-ranks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hybrid_steering import Runner, collect_direction, gdn_layers, load_runtime
from hybrid_steering.direction import load_concept_pairs, read_pairs, target_and_source
from hybrid_steering.runtime import read_jsonl


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


def stacked(by_layer: dict[int, torch.Tensor]) -> torch.Tensor:
    """All layers' head matrices as one ``[layer * head, key, value]`` tensor."""
    return torch.cat([by_layer[layer] for layer in sorted(by_layer)], dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--pairs-dir",
        type=Path,
        help="local copy of the Hub layout, <dir>/<concept>/data/pairs.jsonl; skips the Hub",
    )
    args = parser.parse_args()
    concepts = read_jsonl(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / "rows.jsonl"
    done = (
        {row["concept"] for row in read_jsonl(out)}
        if out.exists() and out.stat().st_size
        else set()
    )
    model, tokenizer = load_runtime(args.model)
    runner = Runner(model, tokenizer, gdn_layers(model), normalize=False)
    with out.open("a", encoding="utf-8") as handle:
        for index, entry in enumerate(concepts):
            name = entry["concept"]
            if name in done:
                continue
            rows = (
                read_pairs(args.pairs_dir / name / "data" / "pairs.jsonl")
                if args.pairs_dir
                else load_concept_pairs(name)
            )
            pairs = [target_and_source(row) for row in rows[: args.pairs]]
            collected = collect_direction(runner, pairs, batch_size=args.batch_size)
            for matrix, by_layer in (
                ("target", collected.mean_target),
                ("source", collected.mean_source),
                ("delta", collected.delta),
            ):
                measured = spectrum_metrics(stacked(by_layer))
                row = {
                    "concept": name,
                    "class_name": entry.get("class_name"),
                    "target": entry.get("target"),
                    "source": entry.get("source"),
                    "pairs": len(pairs),
                    "matrix": matrix,
                    **measured,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{index + 1}/{len(concepts)}] {name}: delta effective {measured['effective']:.2f}",
                flush=True,
            )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
