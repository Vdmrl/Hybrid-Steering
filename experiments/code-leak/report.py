"""Target language in the prose and in the code, by method and scale."""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_steering.report import summary_section, write_page
from hybrid_steering.runtime import read_jsonl

SECTIONS = [
    ("prose_share", "Target language share of the prose"),
    ("code_leak_any", "Answers with the target language in code proper"),
    ("code_leak_share", "Target-script share of code-proper letters"),
    ("has_code", "Answers with a code block"),
    ("code_parses", "Answers whose code parses"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.rows)
    # answers without a code block have no code score; they count only in has_code
    sections = [
        summary_section(
            [row for row in rows if row.get(value) is not None], "scale", value, "method", title
        )
        for value, title in SECTIONS
    ]
    write_page(args.output, "Language steering on code tasks", sections)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
