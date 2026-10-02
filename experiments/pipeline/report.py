"""Join independently cached Judge and benchmark results and plot their intervals."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from html import escape
from pathlib import Path

from intervals import bootstrap_ratio


def mean_interval(values: list[float], *, draws: int = 2000, seed: int = 42) -> dict:
    """Prompt bootstrap for a mean Judge score; deterministic for a frozen set."""
    result = bootstrap_ratio([(value, 1) for value in values], draws=draws, seed=seed)
    return {
        "mean": result["estimate"],
        "low": result["low"],
        "high": result["high"],
        "n": result["n_prompts"],
    }


def build_report(judge_dir: Path, benchmark_dir: Path, output: Path) -> Path | None:
    judge_file = judge_dir / "judge_scores.jsonl"
    benchmark_file = benchmark_dir / "benchmark_scores.json"
    if not judge_file.exists() or not benchmark_file.exists():
        return None
    judge = [json.loads(line) for line in judge_file.read_text().splitlines() if line]
    benchmark = json.loads(benchmark_file.read_text())
    expected_judge = json.loads((judge_dir / "manifest.json").read_text())["n_tasks"]
    grouped = defaultdict(list)
    for row in judge:
        grouped[(row["condition"], float(row["scale"]))].append(row)
    if len({(row["condition"], float(row["scale"]), row["key"]) for row in judge}) != len(judge):
        raise ValueError("duplicate Judge score keys")
    baselines = [items for items in grouped.values() if items[0]["steering_method"] == "baseline"]
    baseline = {row["key"]: row for row in baselines[0]} if baselines else None
    rows = []
    for cell, metrics in benchmark.items():
        condition, scale_text = cell.rsplit(":", 1)
        scale = float(scale_text)
        answers = grouped[(condition, scale)]
        if len(answers) != expected_judge or any(
            row.get("concept_score") is None for row in answers
        ):
            raise ValueError(f"Judge scores incomplete for {cell}")
        judge_ci = mean_interval([float(row["concept_score"]) for row in answers])
        paired = None
        if baseline is not None:
            if {row["key"] for row in answers} != baseline.keys() or any(
                row["prompt"] != baseline[row["key"]]["prompt"] for row in answers
            ):
                raise ValueError(f"Judge prompts do not match baseline for {cell}")
            paired = bootstrap_ratio(
                [
                    (float(row["concept_score"] - baseline[row["key"]]["concept_score"]), 1)
                    for row in answers
                ]
            )
        primary = "pass_at_1" if "pass_at_1" in metrics else "prompt_strict"
        benchmark_ci = metrics["confidence_intervals"][primary]
        rows.append(
            {
                "condition": condition,
                "scale": scale,
                "judge_mean": judge_ci["mean"],
                "judge_low": judge_ci["low"],
                "judge_high": judge_ci["high"],
                "judge_n": judge_ci["n"],
                **(
                    {
                        "judge_delta": paired["estimate"],
                        "judge_delta_low": paired["low"],
                        "judge_delta_high": paired["high"],
                    }
                    if paired
                    else {}
                ),
                "benchmark_metric": primary,
                "benchmark_score": metrics[primary],
                "benchmark_low": benchmark_ci["low"],
                "benchmark_high": benchmark_ci["high"],
                "benchmark_n": benchmark_ci["n"],
            }
        )
    report_dir = output / "reports" / f"{judge_dir.name}-{benchmark_dir.name}"
    report_dir.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: (row["condition"], row["scale"]))
    with (report_dir / "rates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (report_dir / "report.json").write_text(
        json.dumps(
            {
                "judge_manifest": json.loads((judge_dir / "manifest.json").read_text()),
                "benchmark_manifest": json.loads((benchmark_dir / "manifest.json").read_text()),
                "intervals": "95% prompt bootstrap for Judge mean and paired delta; Wilson for binary benchmark rate",
                "report_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    (report_dir / "comparison.svg").write_text(_svg(rows), encoding="utf-8")
    return report_dir


def _svg(rows: list[dict]) -> str:
    """Draw measured points and 95% bars without a plotting dependency."""
    colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2")
    scales = sorted({row["scale"] for row in rows})
    minimum, maximum = scales[0], scales[-1]
    if minimum == maximum:
        minimum, maximum = minimum - 0.5, maximum + 0.5
    canvas_height = max(470, 435 + 17 * len({row["condition"] for row in rows}))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{canvas_height}" viewBox="0 0 1120 {canvas_height}">',
        f'<rect width="1120" height="{canvas_height}" fill="white"/>',
        "<style>text{font:13px sans-serif;fill:#263238}.axis{stroke:#667085;stroke-width:1}.bar{stroke-width:2}</style>",
    ]
    for panel, (stem, ymax, label) in enumerate(
        (
            ("judge", 5, "Judge concept score (mean, 95% CI)"),
            ("benchmark", 1, f"{rows[0]['benchmark_metric']} (rate, 95% CI)"),
        )
    ):
        left, top, width, height = 70 + panel * 555, 65, 440, 300

        def xcoord(value):
            return left + (value - minimum) / (maximum - minimum) * width

        def ycoord(value):
            return top + height - max(0, min(ymax, value)) / ymax * height

        parts += [
            f'<text x="{left}" y="30" font-size="17">{escape(label)}</text>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + height}"/>',
            f'<line class="axis" x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}"/>',
        ]
        for tick in range(6):
            y = ycoord(ymax * tick / 5)
            parts.append(
                f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end">{ymax * tick / 5:.1f}</text>'
            )
        for scale in scales:
            x = xcoord(scale)
            parts.append(
                f'<text x="{x:.1f}" y="{top + height + 22}" text-anchor="middle">{scale:g}</text>'
            )
        parts.append(
            f'<text x="{left + width / 2}" y="{top + height + 42}" text-anchor="middle">Steering scale</text>'
        )
        for index, condition in enumerate(sorted({row["condition"] for row in rows})):
            cells = [row for row in rows if row["condition"] == condition]
            color = colors[index % len(colors)]
            field = "judge_mean" if stem == "judge" else "benchmark_score"
            points = " ".join(
                f"{xcoord(row['scale']):.1f},{ycoord(row[field]):.1f}" for row in cells
            )
            parts.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
            )
            for row in cells:
                x, y = xcoord(row["scale"]), ycoord(row[field])
                low, high = ycoord(row[f"{stem}_low"]), ycoord(row[f"{stem}_high"])
                parts += [
                    f'<line class="bar" x1="{x:.1f}" y1="{high:.1f}" x2="{x:.1f}" y2="{low:.1f}" stroke="{color}"/>',
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"><title>{escape(condition)}; scale {row["scale"]:g}; n={row[f"{stem}_n"]}</title></circle>',
                ]
            parts.append(
                f'<text x="{left}" y="{top + height + 55 + index * 17}" style="fill:{color}">{escape(condition)}</text>'
            )
    return "\n".join(parts + ["</svg>"])
