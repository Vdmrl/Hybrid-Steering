"""Concept score by filler length and scale."""

from hybrid_steering.report import report_main


def main() -> None:
    report_main(
        "Forgetting",
        [("prefix_length", "concept_score", "scale", "Concept score")],
    )


if __name__ == "__main__":
    main()
