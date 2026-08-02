"""Write the fixed neutral development split used for feature gates."""

from __future__ import annotations

import json
from pathlib import Path

DEV_PROMPTS = (
    "How should a team decide whether to postpone a software release?",
    "Explain how a household can reduce its electricity use.",
    "What should a student do when two sources disagree?",
    "How can a small shop improve its inventory planning?",
    "What is a sensible way to compare two job offers?",
    "How should a city prepare for a heat wave?",
    "How can a researcher reduce measurement error?",
    "What should a driver check before a long trip?",
    "How should an organization respond to a recurring data-security incident?",
    "How can a school make a new policy understandable to parents and students?",
    "What should a project manager consider before handing work to another team?",
    "How can a community plan a small public garden with limited resources?",
    "What is a practical way to investigate a recurring customer complaint?",
    "How should a hospital improve the scheduling of routine appointments?",
    "What should a family consider before moving to a new neighborhood?",
    "How can a team decide which maintenance work to prioritize?",
)


def write_split(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({"id": f"final-dev-{index:02d}", "prompt": prompt}) + "\n"
            for index, prompt in enumerate(DEV_PROMPTS)
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    write_split(parser.parse_args().output)
