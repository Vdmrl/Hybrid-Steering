"""Render the compact factual report required by the final feature screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def fmt(value: dict[str, Any] | None) -> str:
    if not value or value["mean"] is None:
        return "missing"
    low, high = value["ci95"]
    return f"{value['mean']:+.3f} [{low:+.3f}, {high:+.3f}] (n={value['n']})"


def render(summary: dict[str, Any]) -> str:
    features = summary["features"]
    selection = summary["selection"]
    lines = [
        "# Final feature composition",
        "",
        "This is a factual experiment artifact. Judge outputs are blind and exploratory.",
        "",
        "## Frozen configuration",
        "",
        "| feature | rank | alpha | c |",
        "| --- | --- | ---: | ---: |",
    ]
    for item in selection["features"]:
        lines.append(
            f"| {item['name']} | {item['rank']} | {item['alpha']} | {item['c']} |"
        )
    lines.extend(["", f"RSS + clamp beta: {selection['clamp_beta']}", ""])
    lines.extend(["## Conditions", ""])
    for name, value in summary["conditions"].items():
        active = "+".join(value["active_features"]) or "baseline"
        joint = fmt(value["joint_success"])
        quality = fmt(value["quality_delta_vs_baseline"])
        lines.append(
            f"- `{name}` ({active}): joint success {joint}; quality delta {quality}"
        )
    lines.extend(["", "## Own-trait deltas against baseline", ""])
    for value in summary["conditions"].values():
        active = value["active_features"]
        if len(active) != 1:
            continue
        feature = active[0]
        delta = value["delta_vs_baseline"].get(feature, {}).get("expected_score")
        lines.append(f"- `{feature}`: expected-score delta {fmt(delta)}")
    lines.extend(["", "## Retention in full composition", ""])
    for feature, value in summary["singleton_retention_expected_score"].items():
        lines.append(f"- `{feature}` full minus singleton: {fmt(value)}")
    lines.extend(["", "## Leakage matrix", ""])
    lines.append(
        "Entries are average expected-score changes when the row feature is added."
    )
    lines.append("")
    lines.append("| added \\ evaluated | " + " | ".join(features) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in features) + " |")
    for added in features:
        values = summary["leakage_matrix_expected_score"].get(added, {})
        lines.append(
            "| "
            + added
            + " | "
            + " | ".join(fmt(values.get(feature)) for feature in features)
            + " |"
        )
    if summary["missing_judge_features"]:
        lines.extend(
            ["", "## Missing", "", ", ".join(summary["missing_judge_features"])]
        )
    lines.extend(["", "## Judge usage", "", json.dumps(summary["judge_usage"]), ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        render(json.loads(args.summary.read_text(encoding="utf-8"))), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
