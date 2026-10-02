"""Language rate and answer equivalence by filler length and scale."""

from hybrid_steering.report import report_main


def main() -> None:
    report_main(
        "SQuAD retention",
        [
            ("prefix_length", "target_language", "scale", "Target language"),
            ("prefix_length", "equivalent", "scale", "Answer equivalence"),
        ],
    )


if __name__ == "__main__":
    main()
