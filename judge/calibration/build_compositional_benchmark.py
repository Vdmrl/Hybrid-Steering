"""Build a small, natural compositional calibration set from Exp5 answers.

The source files are local copies of the already generated Exp5 calibration
answers.  Labels are fixed here before any new Judge calls.  The generated
JSONL is a compact fixture; the large generation files remain outside Git.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FEATURES = (
    "concrete_language",
    "optimism",
    "first_person_voice",
    "bulleted_layout",
)

LANGUAGE_FEATURE = "french_language"
LANGUAGE_SANITY_IDS = {"baseline", "full_add_rss_r1"}

SCENARIOS = {
    "presentation": (
        (
            "deepseek_deepseek_v4_flash__affective_polarity_emotional_opposites__"
            "relief__anxiety__case_study__english__5"
        ),
        "development",
    ),
    "allocation": (
        (
            "deepseek_deepseek_v4_flash__boundaries_consent__negotiated_agreement__"
            "unilateral_imposition__case_study__english__8"
        ),
        "development",
    ),
    "campaign": (
        (
            "deepseek_deepseek_v4_flash__big_five_trait_poles__ambition__"
            "complacency__news__english__4"
        ),
        "validation",
    ),
    "zoning": (
        (
            "deepseek_deepseek_v4_flash__agent_self_descriptions__uses_first_person__"
            "uses_third_person__news__english__0"
        ),
        "validation",
    ),
}

EXTRA_SOURCE_IDS = {
    "development": (0, 1),
    "validation": (2, 3),
}

ANSWER_IDS = (
    "baseline",
    "singleton_optimism",
    "full_add_rss_r1",
    "full_clamp_rss_r1",
    "flip_concrete_language",
    "flip_optimism",
    "flip_bulleted_layout",
    "flip_french_language",
)


def labels(scenario: str, answer_id: str) -> dict[str, int]:
    """Return hand-fixed scores for one real answer and all five traits."""

    french = (
        1
        if answer_id in {"baseline", "singleton_optimism", "flip_french_language"}
        else 5
    )
    concrete = {
        "baseline": 5,
        "singleton_optimism": 4,
        "full_add_rss_r1": 4,
        "full_clamp_rss_r1": 4,
        "flip_concrete_language": 5,
        "flip_optimism": 4,
        "flip_bulleted_layout": 4,
        "flip_french_language": 5,
    }[answer_id]

    # A concrete plan is not optimism by itself. Scores above neutral require
    # an explicit favorable outlook, improvement, or positive expectation.
    optimism = {
        "presentation": {
            "baseline": 3,
            "singleton_optimism": 4,
            "full_add_rss_r1": 3,
            "full_clamp_rss_r1": 4,
            "flip_concrete_language": 4,
            "flip_optimism": 2,
            "flip_bulleted_layout": 4,
            "flip_french_language": 4,
        },
        "allocation": {
            "baseline": 3,
            "singleton_optimism": 4,
            "full_add_rss_r1": 4,
            "full_clamp_rss_r1": 4,
            "flip_concrete_language": 4,
            "flip_optimism": 1,
            "flip_bulleted_layout": 4,
            "flip_french_language": 4,
        },
        "campaign": {
            "baseline": 3,
            "singleton_optimism": 4,
            "full_add_rss_r1": 4,
            "full_clamp_rss_r1": 4,
            "flip_concrete_language": 4,
            "flip_optimism": 3,
            "flip_bulleted_layout": 3,
            "flip_french_language": 3,
        },
        "zoning": {
            "baseline": 3,
            "singleton_optimism": 3,
            "full_add_rss_r1": 3,
            "full_clamp_rss_r1": 3,
            "flip_concrete_language": 3,
            "flip_optimism": 1,
            "flip_bulleted_layout": 3,
            "flip_french_language": 3,
        },
    }[scenario][answer_id]

    # The presentation answers use first person only inside a short quote;
    # the rubric's quoted-fragment exclusion makes those genuinely mixed.
    first_person = (
        3
        if scenario == "presentation"
        and answer_id
        in {
            "full_add_rss_r1",
            "full_clamp_rss_r1",
            "flip_concrete_language",
            "flip_bulleted_layout",
        }
        else 1
    )

    # The campaign baseline and its English flip contain a substantive list.
    bulleted = (
        5
        if (
            scenario == "campaign" and answer_id in {"baseline", "flip_french_language"}
        )
        or (scenario == "zoning" and answer_id == "flip_french_language")
        else 1
    )
    return {
        "french_language": french,
        "concrete_language": concrete,
        "optimism": optimism,
        "first_person_voice": first_person,
        "bulleted_layout": bulleted,
    }


def source_rows(source_dir: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for path in sorted(source_dir.glob("real-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for answer in row["answers"]:
                rows.setdefault(row["prompt_id"], {})[answer["answer_id"]] = {
                    "scenario": row["scenario"],
                    "answer": answer["text"],
                }
    return rows


def extra_labels(variant: str) -> dict[str, int]:
    return {
        "concrete_language": 5,
        "optimism": 3,
        "first_person_voice": 1 if variant == "first_person_negative" else 5,
        "bulleted_layout": 5 if variant == "bullets_positive" else 1,
    }


def extra_rows(source_dir: Path) -> list[dict[str, object]]:
    first_person = {
        json.loads(line)["id"]: json.loads(line)
        for line in (source_dir / "first-person.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    }
    bullets = {
        json.loads(line)["source_id"]: json.loads(line)
        for line in (source_dir / "bullets.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    }
    rows: list[dict[str, object]] = []
    for split, indexes in EXTRA_SOURCE_IDS.items():
        for index in indexes:
            source_id = next(key for key in first_person if key.endswith(f"__{index}"))
            fp = first_person[source_id]
            bullet = bullets[source_id]
            for variant, answer in (
                ("first_person_positive", fp["positive_text"]),
                ("first_person_negative", fp["negative_text"]),
                ("bullets_positive", bullet["positive_text"]),
                ("bullets_negative", bullet["negative_text"]),
            ):
                scores = extra_labels(variant)
                active = [feature for feature, score in scores.items() if score >= 4]
                for feature in FEATURES:
                    rows.append(
                        {
                            "prompt_id": f"compositional-real-{index}-{variant}-{feature}",
                            "feature": feature,
                            "scenario": "A generated response describes a concrete technical case and next steps.",
                            "answer": answer,
                            "expected_score": scores[feature],
                            "note": (
                                f"Real Exp5 {variant}; active traits: "
                                f"{', '.join(active) or 'none'}."
                            ),
                            "source": "exp5_real_first_person_bullets",
                            "split": split,
                            "case_group": f"real_source_{index}",
                            "answer_variant": variant,
                            "active_features": active,
                        }
                    )
    return rows


def build(source_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    available = source_rows(source_dir)
    output: dict[str, list[dict[str, object]]] = {"development": [], "validation": []}
    for scenario_name, (prompt_id, split) in SCENARIOS.items():
        if prompt_id not in available:
            raise SystemExit(f"missing source scenario: {prompt_id}")
        for answer_id in ANSWER_IDS:
            source = available[prompt_id].get(answer_id)
            if source is None:
                raise SystemExit(f"missing answer {answer_id} for {scenario_name}")
            score_by_feature = labels(scenario_name, answer_id)
            active = [
                feature for feature, score in score_by_feature.items() if score >= 4
            ]
            features = list(FEATURES)
            if answer_id in LANGUAGE_SANITY_IDS:
                features.append(LANGUAGE_FEATURE)
            for feature in features:
                expected = score_by_feature[feature]
                output[split].append(
                    {
                        "prompt_id": f"compositional-{scenario_name}-{answer_id}-{feature}",
                        "feature": feature,
                        "scenario": source["scenario"],
                        "answer": source["answer"],
                        "expected_score": expected,
                        "note": (
                            f"Exp5 natural case {scenario_name}/{answer_id}; "
                            f"active traits in this answer: {', '.join(active) or 'none'}."
                        ),
                        "source": "exp5_real_compositional",
                        "split": split,
                        "case_group": scenario_name,
                        "answer_variant": answer_id,
                        "active_features": active,
                    }
                )
    for row in extra_rows(source_dir):
        output[row["split"]].append(row)
    return output["development"], output["validation"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(__file__).with_name("run_results"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "benchmark",
    )
    args = parser.parse_args()
    development, validation = build(args.source_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("development_compositional_v3.jsonl", development),
        ("validation_compositional_v3.jsonl", validation),
    ):
        (args.output_dir / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    manifest = {
        "source": "Exp5 real calibration outputs",
        "features": list(FEATURES),
        "sanity_features": {LANGUAGE_FEATURE: sorted(LANGUAGE_SANITY_IDS)},
        "development": len(development),
        "validation": len(validation),
        "scenario_groups": sorted(
            {row["case_group"] for row in development + validation}
        ),
        "coexisting_source_pairs": "Exp5 first-person and bullet positive/negative pairs",
        "policy": (
            "Split by complete scenario group. V3 corrects deterministic labels for "
            "transformed first-person and numbered-list cases; no Judge prediction "
            "is used as a label."
        ),
    }
    (args.output_dir / "compositional_manifest_v3.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
