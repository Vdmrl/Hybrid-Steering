"""Agreement of two half-pool directions, and how much of each sits in rank 1 and 2.

Halves are split by question, so the same question never feeds both. Per
head: cosine between the flattened half directions, and the energy share
sigma_1^2 / sum sigma^2 (and top two) of the full direction. Heads are
weighted by their energy, since a near-zero head carries no steering.

    uv run python experiments/direction-stability/run.py --directions runs/directions \\
        --output runs/direction-stability
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hybrid_steering import load_direction
from hybrid_steering.report import write_page


def measure(full: dict, first: dict, second: dict) -> dict:
    cosines, energies, top1, top2 = [], [], [], []
    for layer, tensor in full.items():
        a = first[layer].flatten(1)
        b = second[layer].flatten(1)
        cosines.append(torch.nn.functional.cosine_similarity(a, b, dim=1))
        singular = torch.linalg.svdvals(tensor.float())
        energy = singular.pow(2)
        energies.append(energy.sum(-1))
        top1.append(energy[:, 0] / energy.sum(-1))
        top2.append(energy[:, :2].sum(-1) / energy.sum(-1))
    cosine, energy = torch.cat(cosines), torch.cat(energies)
    weights = energy / energy.sum()
    return {
        "heads": len(cosine),
        "half cosine (energy-weighted)": float((weights * cosine).sum()),
        "half cosine (median head)": float(cosine.median()),
        "rank-1 energy share": float((weights * torch.cat(top1)).sum()),
        "rank-2 energy share": float((weights * torch.cat(top2)).sum()),
        "direction norm": float(energy.sum().sqrt()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for half0 in sorted(args.directions.glob("*-half0")):
        slug = half0.name.removesuffix("-half0")
        paths = [
            args.directions / name / "direction"
            for name in (slug, f"{slug}-half0", f"{slug}-half1")
        ]
        if not all(path.exists() for path in paths):
            continue
        full, first, second = (load_direction(path)[0] for path in paths)
        rows.append({"concept": slug, **measure(full, first, second)})
        print(rows[-1], flush=True)
    columns = list(rows[0])
    head = "".join(f"<th>{c}</th>" for c in columns)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{r[c]:.3f}</td>" if isinstance(r[c], float) else f"<td>{r[c]}</td>"
            for c in columns
        )
        + "</tr>"
        for r in rows
    )
    write_page(
        args.output / "report.html",
        "Direction stability",
        [
            f"<h2>Half-pool agreement and rank concentration</h2><table><tr>{head}</tr>{body}</table>"
        ],
    )


if __name__ == "__main__":
    main()
