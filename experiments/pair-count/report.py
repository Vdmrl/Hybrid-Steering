"""Rank and cosine-to-full-pool by number of pairs."""

from hybrid_steering.report import report_main


def main() -> None:
    report_main(
        "Direction by number of pairs",
        [
            ("pairs", "effective", "matrix", "Effective rank (energy) by pairs"),
            ("pairs", "stable", "matrix", "Stable rank by pairs"),
            ("pairs", "rank90", "matrix", "Singular values for 90% of the energy by pairs"),
            ("pairs", "cosine_to_full", "matrix", "Cosine to the full-pool matrix by pairs"),
        ],
    )


if __name__ == "__main__":
    main()
