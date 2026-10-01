"""Concept score by steering mode and scale."""

from hybrid_steering.report import report_main


def main() -> None:
    report_main(
        "Context length",
        [
            ("scale", "concept_score", "mode", "Concept score"),
            ("scale", "content_quality", "mode", "Content quality"),
        ],
    )


if __name__ == "__main__":
    main()
