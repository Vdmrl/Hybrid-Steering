"""Concept score by scale for a steering run."""

from hybrid_steering.report import report_main


def main() -> None:
    report_main(
        "Steering",
        [
            ("scale", "concept_score", None, "Concept score by scale"),
            ("scale", "content_quality", None, "Content quality by scale"),
        ],
    )


if __name__ == "__main__":
    main()
