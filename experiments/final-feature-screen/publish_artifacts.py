"""Publish only compact, portable final artifacts to the experiment branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def portable_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Remove machine-specific paths while retaining frozen intervention choices."""
    result = json.loads(json.dumps(summary))
    for feature in result.get("selection", {}).get("features", []):
        if "direction" in feature:
            feature["direction"] = Path(feature["direction"]).name
    return result


def manifest(summary: dict[str, Any]) -> dict[str, Any]:
    selection = summary["selection"]
    return {
        "schema_version": "final-feature-screen-manifest-1",
        "generation_model": "Qwen/Qwen3.5-9B",
        "prompt_split": {"kind": "held_out", "n_prompts": summary["n_prompts"]},
        "intervention": {
            "composition": "RSS-normalized direction sum",
            "clamp": {"enabled": True, "beta": selection["clamp_beta"]},
            "features": [
                {
                    key: feature[key]
                    for key in ("name", "rank", "alpha", "c", "sign")
                    if key in feature
                }
                for feature in selection["features"]
            ],
        },
        "evaluation": {
            "judge": "standard blind Judge v3",
            "endpoints": [
                "trait_score",
                "score_distribution",
                "expected_score",
                "p_ge4",
                "answer_quality",
            ],
            "bootstrap": {"samples": 2000, "confidence": 0.95},
        },
    }


def publish(summary_path: Path, report_path: Path, output_dir: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    compact = portable_summary(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest(compact), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        report_path.read_text(encoding="utf-8"), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    publish(args.summary, args.report, args.output_dir)


if __name__ == "__main__":
    main()
