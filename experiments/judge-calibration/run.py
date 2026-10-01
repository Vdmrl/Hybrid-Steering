"""Check the concept judge on the pairs that define each direction.

For every concept, a random sample of pairs is scored on both sides with the
concept's own guide: the positive text should score high, the plain text low.
Positives of every other concept are also scored with this guide, so a judge
that rewards any unusual style shows up off the diagonal.

Reported per concept: mean score per side, AUC of positive over negative with
a bootstrap 95% interval, and the true and false positive rates at score >= 2.

    uv run python experiments/judge-calibration/run.py --pairs runs/pairs-v2 \\
        --output runs/judge-calibration
"""

from __future__ import annotations

import argparse
import random
from html import escape
from pathlib import Path

from hybrid_steering.direction import load_concept_pairs, read_pairs
from hybrid_steering.judge import score_steering
from hybrid_steering.report import write_page
from hybrid_steering.runtime import write_jsonl

FEATURES = {
    "plain-theistic_framing": "theistic_framing",
    "plain-probabilistic_framing": "probabilistic_framing",
    "plain-fictional_narrative": "fictional_narrative",
    "plain-comparative_framing": "comparative_framing",
    "technical": "technical_language",
}
THRESHOLD = 2


def load(slug: str, pairs: Path) -> list[dict]:
    local = pairs / slug / "pairs.jsonl"
    if local.exists():
        return read_pairs(local)
    return load_concept_pairs(slug, "pairs_n218.jsonl" if slug == "technical" else "pairs.jsonl")


def auc(positive: list[float], negative: list[float]) -> float:
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def bootstrap(positive: list[float], negative: list[float], seed: int, rounds: int = 2000):
    rng = random.Random(seed)
    values = sorted(
        auc(rng.choices(positive, k=len(positive)), rng.choices(negative, k=len(negative)))
        for _ in range(rounds)
    )
    return values[int(0.025 * rounds)], values[int(0.975 * rounds) - 1]


def rate(scores: list[int]) -> float:
    return sum(score >= THRESHOLD for score in scores) / len(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--cross", type=int, default=20)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--thinking", action=argparse.BooleanOptionalAction, help="override config/judge.yaml"
    )
    args = parser.parse_args()

    pools = {slug: load(slug, args.pairs) for slug in FEATURES}
    picked = {
        slug: random.Random(args.seed).sample(rows, min(args.samples, len(rows)))
        for slug, rows in pools.items()
    }
    jobs = []
    for slug, feature in FEATURES.items():
        for row in picked[slug]:
            question = row.get("source_question") or ""
            jobs.append((feature, slug, "positive", question, row["positive_text"]))
            jobs.append((feature, slug, "negative", question, row["negative_text"]))
        for other in FEATURES:
            if other == slug:
                continue
            for row in picked[other][: args.cross]:
                question = row.get("source_question") or ""
                jobs.append((feature, other, "positive", question, row["positive_text"]))

    rows = []
    for feature in FEATURES.values():
        mine = [job for job in jobs if job[0] == feature]
        judgments = score_steering(
            [(question, text) for *_, question, text in mine],
            feature,
            batch_size=args.batch_size,
            thinking=args.thinking,
        )
        for (_, slug, side, question, text), judgment in zip(mine, judgments, strict=True):
            rows.append(
                {
                    "feature": feature,
                    "text_concept": slug,
                    "side": side,
                    "question": question,
                    "text": text,
                    "concept_score": judgment.concept_score if judgment else None,
                    "content_quality": judgment.content_quality if judgment else None,
                    "reason": judgment.reason if judgment else None,
                }
            )
    write_jsonl(args.output / "rows.jsonl", rows)

    summary = []
    for slug, feature in FEATURES.items():
        own = [row for row in rows if row["feature"] == feature and row["text_concept"] == slug]
        positive = [
            r["concept_score"]
            for r in own
            if r["side"] == "positive" and r["concept_score"] is not None
        ]
        negative = [
            r["concept_score"]
            for r in own
            if r["side"] == "negative" and r["concept_score"] is not None
        ]
        low, high = bootstrap(positive, negative, args.seed)
        summary.append(
            {
                "concept": slug,
                "n": f"{len(positive)}/{len(negative)}",
                "positive mean": sum(positive) / len(positive),
                "negative mean": sum(negative) / len(negative),
                "AUC": auc(positive, negative),
                "AUC 95% CI": f"{low:.2f}–{high:.2f}",
                f"TPR (>= {THRESHOLD})": rate(positive),
                f"FPR (>= {THRESHOLD})": rate(negative),
                "quality pos/neg": "{:.2f} / {:.2f}".format(
                    *(
                        sum(
                            r["content_quality"]
                            for r in own
                            if r["side"] == side and r["content_quality"] is not None
                        )
                        / max(
                            1,
                            sum(
                                r["side"] == side and r["content_quality"] is not None for r in own
                            ),
                        )
                        for side in ("positive", "negative")
                    )
                ),
            }
        )

    def cell(value) -> str:
        return f"{value:.2f}" if isinstance(value, float) else escape(str(value))

    head = "".join(f"<th>{escape(key)}</th>" for key in summary[0])
    body = "".join(
        "<tr>" + "".join(f"<td>{cell(value)}</td>" for value in item.values()) + "</tr>"
        for item in summary
    )
    slugs = list(FEATURES)
    matrix = (
        "<tr><th>guide \\ positive texts of</th>"
        + "".join(f"<th>{escape(slug)}</th>" for slug in slugs)
        + "</tr>"
    )
    for slug, feature in FEATURES.items():
        matrix += f"<tr><th>{escape(feature)}</th>"
        for other in slugs:
            scores = [
                r["concept_score"]
                for r in rows
                if r["feature"] == feature
                and r["text_concept"] == other
                and r["side"] == "positive"
                and r["concept_score"] is not None
            ][: args.cross]
            matrix += f"<td>{sum(scores) / max(1, len(scores)):.2f}</td>"
        matrix += "</tr>"
    misses = [
        r
        for r in rows
        if r["text_concept"] == next(s for s, f in FEATURES.items() if f == r["feature"])
        and (
            (r["side"] == "positive" and (r["concept_score"] or 0) < THRESHOLD)
            or (r["side"] == "negative" and (r["concept_score"] or 0) >= THRESHOLD)
        )
    ]
    examples = "".join(
        f"<tr><td>{escape(r['feature'])}</td><td>{escape(r['side'])}</td>"
        f"<td>{r['concept_score']}</td><td style='text-align:left'>{escape(r['text'])}</td>"
        f"<td style='text-align:left'>{escape(r['reason'] or '')}</td></tr>"
        for r in misses[:60]
    )
    write_page(
        args.output / "report.html",
        "Judge calibration on concept pairs",
        [
            f"<h2>Own guide, both sides</h2><table><tr>{head}</tr>{body}</table>",
            "<h2>Mean score of each concept's positive texts under each guide</h2>"
            f"<table>{matrix}</table>",
            f"<h2>Disagreements ({len(misses)})</h2><table><tr><th>guide</th><th>side</th>"
            f"<th>score</th><th>text</th><th>judge reason</th></tr>{examples}</table>",
        ],
    )
    print(f"wrote {args.output / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
