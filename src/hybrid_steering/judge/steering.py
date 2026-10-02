"""One steering judge: concept presence 0–4 and content quality 0–4."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from hybrid_steering.detect import concept_detector, is_language

from .client import complete_batch
from .config import repo_root

FLAGS = {
    "empty",
    "repetition",
    "incoherent",
    "off_topic",
    "unsupported_claim",
    "apparent_truncation",
    "quotation_only",
    "prompt_injection_attempt",
}
FIELDS = {"evaluable", "concept_score", "content_quality", "flags", "reason"}


@dataclass(frozen=True)
class Judgment:
    """One validated judge label. A missing request is not a judgment."""

    evaluable: bool
    concept_score: int | None
    content_quality: int
    flags: tuple[str, ...]
    reason: str


def load_guides(root: Path | None = None) -> dict[str, str]:
    """Concept paragraphs inserted into the shared rubric. Languages are absent."""
    root = root or repo_root()
    raw = yaml.safe_load((root / "concepts" / "features.yaml").read_text(encoding="utf-8"))
    guides = {}
    for name, item in raw["features"].items():
        guide = str(item["guide"]).strip()
        if not guide:
            raise ValueError(f"{name} has an empty guide")
        guides[name] = guide
    return guides


def render(feature: str, *, root: Path | None = None) -> str:
    """Shared rubric with one concept guide filled in."""
    root = root or repo_root()
    guides = load_guides(root)
    try:
        guide = guides[feature]
    except KeyError:
        known = ", ".join(sorted(guides))
        raise ValueError(f"unknown feature {feature!r}; choose from {known}") from None
    template = (root / "prompts" / "steering_judge.txt").read_text(encoding="utf-8")
    if "{{concept}}" not in template:
        raise ValueError("steering rubric is missing {{concept}}")
    return template.replace("{{concept}}", guide)


def parse_judgment(raw: str) -> Judgment:
    """Accept the shared JSON schema. Anything else is not a score."""
    label = json.loads(raw)
    if not isinstance(label, dict) or set(label) != FIELDS:
        raise ValueError("judge JSON fields do not match the rubric")
    evaluable = label["evaluable"]
    concept_score = label["concept_score"]
    content_quality = label["content_quality"]
    flags = label["flags"]
    reason = label["reason"]
    if type(evaluable) is not bool:
        raise ValueError("evaluable must be a boolean")
    if evaluable:
        if type(concept_score) is not int or not 0 <= concept_score <= 4:
            raise ValueError("concept_score must be an integer 0–4")
    elif concept_score is not None:
        raise ValueError("an unevaluable response has concept_score null")
    if type(content_quality) is not int or not 0 <= content_quality <= 4:
        raise ValueError("content_quality must be an integer 0–4")
    if (
        not isinstance(flags, list)
        or not all(isinstance(flag, str) for flag in flags)
        or len(set(flags)) != len(flags)
        or set(flags) - FLAGS
    ):
        raise ValueError("flags must be unique names from the rubric")
    if not isinstance(reason, str) or not 0 < len(reason.split()) <= 35:
        raise ValueError("reason must be 1–35 words")
    return Judgment(evaluable, concept_score, content_quality, tuple(flags), reason)


def score_steering(
    pairs: list[tuple[str, str]],
    feature: str,
    *,
    batch_size: int = 8,
    thinking: bool | None = None,
) -> list[Judgment | None]:
    """Score ``(prompt, response)`` pairs for one concept.

    Requests go out together. ``None`` is a failed call, not a zero.
    ``thinking`` overrides ``config/judge.yaml``.
    """
    if not pairs:
        return []
    system = render(feature)
    message_lists = [
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False),
            },
        ]
        for prompt, response in pairs
    ]
    raws = complete_batch(message_lists, json_object=True, batch_size=batch_size, thinking=thinking)
    judgments: list[Judgment | None] = []
    invalid = 0
    for raw in raws:
        if raw is None:
            judgments.append(None)
            continue
        try:
            judgments.append(parse_judgment(raw))
        except ValueError as error:
            invalid += 1
            if invalid == 1:
                print(f"judge skipped a response: {error}", flush=True)
            judgments.append(None)
    if invalid:
        print(f"judge rejected {invalid} labels", flush=True)
    if all(item is None for item in judgments):
        raise RuntimeError("judge returned no scores")
    return judgments


def score_rows(
    rows: list[dict],
    feature: str,
    *,
    prompt_field: str = "prompt",
    response_field: str = "response",
    batch_size: int = 8,
) -> None:
    """Write scores onto generation rows. Languages stay on Lingua."""
    if not rows:
        return
    if is_language(feature):
        detector = concept_detector(feature)
        for row in rows:
            text = row[response_field]
            row["label"] = detector.label(text)
            row["concept_score"] = int(detector.detects(text))
        return
    judgments = score_steering(
        [(row[prompt_field], row[response_field]) for row in rows],
        feature,
        batch_size=batch_size,
    )
    for row, judgment in zip(rows, judgments, strict=True):
        if judgment is None:
            row["concept_score"] = None
            row["content_quality"] = None
            row["evaluable"] = None
            row["flags"] = []
            continue
        row["concept_score"] = judgment.concept_score
        row["content_quality"] = judgment.content_quality
        row["evaluable"] = judgment.evaluable
        row["flags"] = list(judgment.flags)
        row["reason"] = judgment.reason
