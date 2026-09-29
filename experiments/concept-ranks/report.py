"""Distribution of direction rank over concepts."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from hybrid_steering.report import summary_section, write_page
from hybrid_steering.runtime import read_jsonl

WIDTH = {"effective": 0.5, "stable": 0.25, "rank90": 2.0, "rank95": 2.0}


def histogram(rows: list[dict], metric: str) -> list[dict]:
    """Concept counts per bin of ``metric`` for the difference matrix."""
    width = WIDTH[metric]
    counts: dict[float, int] = {}
    for row in rows:
        if row["matrix"] == "delta":
            low = math.floor(row[metric] / width) * width
            counts[low] = counts.get(low, 0) + 1
    return [
        {metric: f"{low:g}-{low + width:g}", "concepts": count}
        for low, count in sorted(counts.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.rows)
    sections = [
        summary_section(
            histogram(rows, metric),
            metric,
            "concepts",
            title=f"Concepts by {metric} rank of the direction",
        )
        for metric in ("effective", "stable", "rank90")
    ]
    sections.append(
        summary_section(rows, "matrix", "effective", title="Mean effective rank by matrix")
    )
    write_page(args.output, "Direction rank across concepts", sections)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
