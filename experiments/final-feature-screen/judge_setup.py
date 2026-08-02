"""Create external exploratory rubrics and blind Judge inputs for this run."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

EXPLORATORY = {
    "russian_language": (
        "Russian language",
        "non-Russian language",
        "The substantive answer is written in natural Russian.",
        {
            1: "The substantive answer is consistently non-Russian.",
            2: "The answer is mostly non-Russian with only short Russian fragments.",
            3: "The language is genuinely mixed, unclear, or too short to judge.",
            4: "The answer is predominantly natural Russian with minor non-Russian material.",
            5: "The substantive answer is consistently and fluently written in Russian.",
        },
    ),
    "technical": (
        "technical vocabulary",
        "basic vocabulary",
        "Technical vocabulary uses appropriate domain-specific terms and precise mechanisms.",
        {
            1: "Uses only everyday wording and avoids relevant technical mechanisms.",
            2: "Is mostly basic vocabulary with at most one incidental technical term.",
            3: "Is mixed, neutral, or the scenario does not make the vocabulary contrast judgeable.",
            4: "Clearly uses appropriate domain-specific terms or precise mechanisms.",
            5: "Uses several accurate, appropriate specialized terms and mechanisms throughout the answer.",
        },
    ),
    "numbered_list": (
        "numbered-list structure",
        "continuous prose",
        "Numbered-list structure presents distinct content in explicitly ordered numbered items.",
        {
            1: "Continuous prose with no ordered list structure.",
            2: "Has an isolated ordinal, heading, or unordered bullets but no clear ordered list.",
            3: "Mixed or ambiguous structure, or only a weak partial numbered list.",
            4: "Has at least two distinct, explicitly numbered items.",
            5: "Has three or more meaningful, explicitly numbered items forming a clear ordered structure.",
        },
    ),
    "persuasive": (
        "persuasive style",
        "informative style",
        "Persuasive style tries to move the reader toward a position or action using reasons.",
        {
            1: "Purely informative or descriptive, without an advocated position or action.",
            2: "Has only a weak preference or recommendation without supporting reasons.",
            3: "Neutral, mixed, absent, or balanced without a clear attempt to persuade.",
            4: "Clearly recommends a position or action and gives supporting reasons.",
            5: "Sustained evidence-based advocacy with clear reasons and a concrete call to action.",
        },
    ),
}


def feature(
    target: str, opposite: str, definition: str, anchors: dict[int, str]
) -> dict:
    return {
        "target": target,
        "opposite": opposite,
        "definition": definition,
        "exclusions": [
            "Do not reward answer quality, verbosity, politeness, or another target trait by itself.",
            "Judge only observable answer text; the intervention is unknown.",
        ],
        "anchors": anchors,
    }


def config_features(features: list[str]) -> dict[str, dict]:
    definitions = {}
    for name in features:
        if name == "optimism":
            definitions[name] = feature(
                "optimism",
                "pessimism",
                "Optimism gives appropriate weight to plausible favorable outcomes while acknowledging uncertainty and risk.",
                {
                    1: "Treats failure as overwhelmingly likely and improvement as implausible.",
                    2: "Leans pessimistic beyond the evidence while allowing limited improvement.",
                    3: "Balanced, mixed, absent, or appropriately uncertain outlook.",
                    4: "Clearly gives weight to plausible improvement while acknowledging risk.",
                    5: "Consistently emphasizes realistic favorable possibilities without denying uncertainty or risk.",
                },
            )
        elif name == "answer_quality":
            definitions[name] = feature(
                "high answer quality",
                "poor answer quality",
                "High quality correctly and coherently fulfills the scenario without material errors or omissions.",
                {
                    1: "Fails the request, is materially wrong, or is largely incoherent or unusable.",
                    2: "Addresses the request weakly with substantial errors, omissions, or coherence problems.",
                    3: "Partially fulfills the request but has notable limitations or missing information.",
                    4: "Correctly and coherently fulfills the request with only minor defects.",
                    5: "Fully, correctly, clearly, and coherently fulfills the request without material defects.",
                },
            )
        elif name in EXPLORATORY:
            definitions[name] = feature(*EXPLORATORY[name])
        else:
            raise ValueError(f"no rubric for {name}")
    return definitions


def write_config(output_dir: Path, repo_root: Path, features: list[str]) -> Path:
    root = output_dir / "judge-config"
    concepts = root.parent / "concepts"
    for folder in (concepts, root / "config", root / "prompts"):
        folder.mkdir(parents=True, exist_ok=True)
    definitions = config_features([*features, "answer_quality"])
    lines = ["rubric_version: exploratory-final-feature-screen-1\nfeatures:\n"]
    for name, value in definitions.items():
        lines.extend(
            [
                f"  {name}:\n",
                f"    target: {json.dumps(value['target'])}\n",
                f"    opposite: {json.dumps(value['opposite'])}\n",
                f"    definition: {json.dumps(value['definition'])}\n",
                "    exclusions:\n",
                *[f"      - {json.dumps(item)}\n" for item in value["exclusions"]],
                "    anchors:\n",
                *[
                    f"      {score}: {json.dumps(anchor)}\n"
                    for score, anchor in value["anchors"].items()
                ],
            ]
        )
    (concepts / "features.yaml").write_text("".join(lines), encoding="utf-8")
    shutil.copy2(repo_root / "judge" / "config" / "judge.yaml", root / "config")
    shutil.copy2(repo_root / "judge" / "prompts" / "judge_v3.txt", root / "prompts")
    return root


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_inputs(output_dir: Path, generations: Path, features: list[str]) -> None:
    records = read_jsonl(generations)
    destination = output_dir / "judge-inputs"
    destination.mkdir(parents=True, exist_ok=True)
    for feature_name in [*features, "answer_quality"]:
        path = destination / f"{feature_name}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in records:
                item = {
                    "prompt_id": f"{row['task_id']}:{feature_name}",
                    "scenario": row["scenario"],
                    "answers": [{"answer_id": "answer_0", "text": row["response"]}],
                    "metadata": {"condition": row["condition"]},
                }
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("config", "inputs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--generations", type=Path)
    args = parser.parse_args()
    if args.phase == "config":
        write_config(args.output_dir, args.repo_root, args.features)
    elif args.generations is None:
        raise ValueError("--generations is required for inputs")
    else:
        write_inputs(args.output_dir, args.generations, args.features)


if __name__ == "__main__":
    main()
