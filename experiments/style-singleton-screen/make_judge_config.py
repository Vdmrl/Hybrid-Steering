from __future__ import annotations

import json
import shutil
from pathlib import Path

AXES = {
    "humorous": (
        "humorous",
        "factual/serious",
        "Humor uses a clearly comedic framing, incongruity, punchline, or playful reinterpretation rather than merely positive sentiment.",
    ),
    "adjective_emphasis": (
        "adjective emphasis",
        "neutral adjective use",
        "Adjective emphasis makes descriptive qualities unusually salient through deliberate, expressive adjective choices.",
    ),
    "action_emphasis": (
        "verb/action emphasis",
        "neutral action description",
        "Verb/action emphasis makes events and actions unusually salient through vivid or strongly foregrounded action wording.",
    ),
    "technical": (
        "technical vocabulary",
        "basic vocabulary",
        "Technical vocabulary uses domain-specific terms and precise mechanisms when they are relevant.",
    ),
    "persuasive": (
        "persuasive style",
        "informative style",
        "Persuasive style explicitly tries to move the reader toward a position or action using reasons and calls to action.",
    ),
    "narrative": (
        "narrative style",
        "analytical style",
        "Narrative style presents events as a temporally connected account with actors, actions, and progression.",
    ),
    "answer_quality": (
        "high answer quality",
        "poor answer quality",
        "High quality means correctly and coherently fulfilling the scenario without material errors or omissions.",
    ),
}


def rubric(target: str, opposite: str, definition: str) -> dict:
    return {
        "target": target,
        "opposite": opposite,
        "definition": definition,
        "exclusions": [
            "Do not reward answer quality, verbosity, politeness, or unusual vocabulary by themselves.",
            "Judge the target style in the answer text, not the method or intervention name.",
        ],
        "anchors": {
            1: f"Strong and consistent expression of {opposite}.",
            2: f"Mostly {opposite}, with only weak target evidence.",
            3: "Neutral, mixed, absent, ambiguous, or balanced.",
            4: f"Clear {target} expression with minor inconsistency.",
            5: f"Strong and sustained {target} expression throughout the answer.",
        },
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir / "judge-config"
    for folder in (root / "concepts", root / "config", root / "prompts"):
        folder.mkdir(parents=True, exist_ok=True)
    definitions = {name: rubric(*values) for name, values in AXES.items()}
    chunks = ["rubric_version: exploratory-style-screen-1\nfeatures:\n"]
    for name, value in definitions.items():
        chunks.extend(
            [
                f"  {name}:\n",
                f"    target: {json.dumps(value['target'], ensure_ascii=False)}\n",
                f"    opposite: {json.dumps(value['opposite'], ensure_ascii=False)}\n",
                f"    definition: {json.dumps(value['definition'], ensure_ascii=False)}\n",
                "    exclusions:\n",
                *[
                    f"      - {json.dumps(item, ensure_ascii=False)}\n"
                    for item in value["exclusions"]
                ],
                "    anchors:\n",
                *[
                    f"      {score}: {json.dumps(text, ensure_ascii=False)}\n"
                    for score, text in value["anchors"].items()
                ],
            ]
        )
    (root / "concepts" / "features.yaml").write_text("".join(chunks), encoding="utf-8")
    shutil.copy2(
        args.repo_root / "judge" / "config" / "judge.yaml",
        root / "config" / "judge.yaml",
    )
    shutil.copy2(
        args.repo_root / "judge" / "prompts" / "judge_v3.txt",
        root / "prompts" / "judge_v3.txt",
    )


if __name__ == "__main__":
    main()
