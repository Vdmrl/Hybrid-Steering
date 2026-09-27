"""Build a pole-labeled Judge benchmark from existing concept corpora.

The hand-written calibration cases are few and their labels come from the same
assistant that writes the prompts. Pole-labeled corpora give a much larger,
independently-produced signal: each text is already known to have been written
to express one side of a concept axis, so the Judge can be scored on whether it
separates the sides -- no per-text human score needed.

Two source formats are supported:

  pairs   aligned rewrites, one row per pair with both sides. This is the
          PLABA-MU layout used by the `technical` concept in the
          hybrid-steering-concepts dataset (`negative_text`/`positive_text`),
          and it enables the paired statistics in pole_metrics.py.
          --source pairs --positive-field positive_text --negative-field negative_text

  poles   one row per text with a pole marker, the layout of the GDN project's
          generated stimuli (`key`, `pole` in {concept, antagonist}, `text`).
          --source poles --concept-key russian_vs_english

Outputs two files that line up by `prompt_id`:
  <out>.input.jsonl  JudgeInput rows, ready for the standard runner
  <out>.gold.jsonl   {prompt_id, pole, pair_id, concept} sidecar for pole_metrics

The scenario field carries the eliciting prompt when the corpus has one, else a
neutral placeholder: the Judge scores the answer text, and a scenario that
announced the pole would leak the label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

NEUTRAL_SCENARIO = "An assistant was asked to write a short passage on the topic below."
TARGET, OPPOSITE = "target", "opposite"


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def from_pairs(
    rows: list[dict[str, Any]],
    *,
    feature: str,
    positive_field: str,
    negative_field: str,
    id_field: str,
    scenario_field: str | None,
    limit: int | None,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    for index, row in enumerate(rows[:limit] if limit else rows):
        pair_id = str(row.get(id_field, index))
        scenario = row.get(scenario_field) if scenario_field else None
        for pole, field in ((TARGET, positive_field), (OPPOSITE, negative_field)):
            text = row.get(field)
            if not text:
                continue
            prompt_id = f"{feature}:{pair_id}:{pole}"
            yield (
                {
                    "prompt_id": prompt_id,
                    "scenario": scenario or NEUTRAL_SCENARIO,
                    "answers": [{"answer_id": "answer_0", "text": text}],
                    "metadata": {"concept": feature, "source": "pairs"},
                },
                {"prompt_id": prompt_id, "pole": pole, "pair_id": f"{feature}:{pair_id}", "concept": feature},
            )


def from_poles(
    rows: list[dict[str, Any]],
    *,
    feature: str,
    concept_key: str,
    text_field: str,
    pole_field: str,
    key_field: str,
    positive_pole: str,
    scenario_field: str | None,
    limit: int | None,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    selected = [row for row in rows if row.get(key_field) == concept_key]
    per_pole: dict[str, int] = {}
    for index, row in enumerate(selected):
        pole = TARGET if row.get(pole_field) == positive_pole else OPPOSITE
        seen = per_pole.get(pole, 0)
        if limit and seen >= limit:
            continue
        per_pole[pole] = seen + 1
        text = row.get(text_field)
        if not text:
            continue
        prompt_id = f"{feature}:{index}:{pole}"
        yield (
            {
                "prompt_id": prompt_id,
                "scenario": row.get(scenario_field) or NEUTRAL_SCENARIO
                if scenario_field
                else NEUTRAL_SCENARIO,
                "answers": [{"answer_id": "answer_0", "text": text}],
                "metadata": {"concept": feature, "source": "poles"},
            },
            {"prompt_id": prompt_id, "pole": pole, "pair_id": None, "concept": feature},
        )


def leaking_scenarios(
    inputs: list[dict[str, Any]], feature: str, concept_key: str | None
) -> int:
    """Count scenarios that name the concept. A stimulus corpus is often built
    by telling the model which pole to express ("write this in a way that
    clearly reflects X"); reusing that instruction as the scenario would let the
    Judge read the label instead of the text."""
    words = {
        word
        for source in (feature, concept_key or "")
        for word in source.replace("_", " ").split()
        if len(word) > 3
    }
    return sum(
        any(word in row["scenario"].casefold() for word in words) for row in inputs
    )


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="output stem")
    parser.add_argument("--feature", required=True, help="feature name in concepts/features.yaml")
    parser.add_argument("--source", choices=("pairs", "poles"), required=True)
    parser.add_argument("--limit", type=int, help="cap rows (pairs) or rows per pole (poles)")
    parser.add_argument("--scenario-field")
    parser.add_argument("--positive-field", default="positive_text")
    parser.add_argument("--negative-field", default="negative_text")
    parser.add_argument("--id-field", default="pair_id")
    parser.add_argument("--concept-key")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--pole-field", default="pole")
    parser.add_argument("--key-field", default="key")
    parser.add_argument("--positive-pole", default="concept")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    rows = list(read_jsonl(args.corpus))
    if args.source == "pairs":
        built = from_pairs(
            rows,
            feature=args.feature,
            positive_field=args.positive_field,
            negative_field=args.negative_field,
            id_field=args.id_field,
            scenario_field=args.scenario_field,
            limit=args.limit,
        )
    else:
        if not args.concept_key:
            raise SystemExit("--concept-key is required for --source poles")
        built = from_poles(
            rows,
            feature=args.feature,
            concept_key=args.concept_key,
            text_field=args.text_field,
            pole_field=args.pole_field,
            key_field=args.key_field,
            positive_pole=args.positive_pole,
            scenario_field=args.scenario_field,
            limit=args.limit,
        )

    inputs, gold = [], []
    for judge_input, gold_row in built:
        inputs.append(judge_input)
        gold.append(gold_row)
    if not inputs:
        raise SystemExit("no rows produced; check --source, --concept-key and field names")

    leaked = leaking_scenarios(inputs, args.feature, args.concept_key)
    if leaked:
        raise SystemExit(
            f"{leaked}/{len(inputs)} scenarios name the concept itself, which hands the "
            "Judge the label. Elicitation prompts of the form 'write this in a way that "
            "reflects X' must not be reused as scenarios: drop --scenario-field to fall "
            "back to the neutral placeholder."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix, payload in ((".input.jsonl", inputs), (".gold.jsonl", gold)):
        path = args.out.with_suffix("")
        path = path.with_name(path.name + suffix)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in payload),
            encoding="utf-8",
        )
        print(f"wrote {path} ({len(payload)} rows)")
    poles = {TARGET: 0, OPPOSITE: 0}
    for row in gold:
        poles[row["pole"]] += 1
    print(f"feature={args.feature} target={poles[TARGET]} opposite={poles[OPPOSITE]}")


if __name__ == "__main__":
    main()
