from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    records = []
    for path in sorted((out / "judge-results").glob("*.jsonl")):
        feature = path.stem
        input_path = out / "judge-inputs" / f"{feature}.jsonl"
        inputs = {
            row["prompt_id"]: row
            for row in (
                json.loads(line)
                for line in input_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            result = json.loads(line)
            metadata = inputs.get(result["prompt_id"], {}).get("metadata", {})
            distribution = result.get("score_distribution", {})
            records.append(
                {
                    "feature": feature,
                    "prompt_id": result["prompt_id"],
                    "condition": metadata.get("condition", "unknown"),
                    "expected_score": distribution.get(
                        "expected_score", result.get("trait_score")
                    ),
                    "trait_score": result.get("trait_score"),
                }
            )
    (out / "compact-results.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n",
        encoding="utf-8",
    )
    print(f"compacted {len(records)} Judge rows")


if __name__ == "__main__":
    main()
