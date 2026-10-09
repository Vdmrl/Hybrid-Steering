"""Freeze neutral and concept-stress prompts for blind style evaluation.

This module prepares data and a benchmark plan; it never loads a model or calls
an API. Generate answers with the shared hybrid_steering CLI.
"""

import argparse
import json
from pathlib import Path

from hybrid_steering.judge.io import digest, read_jsonl, write
from hybrid_steering.extract import ROOT


PROMPT_ROOT = ROOT / "prompts/judge/v1"
CATEGORIES = {"open_question", "explanation", "commented_code", "mathematics"}


def _source(path: Path, suite: str) -> list[dict]:
    rows = read_jsonl(path)
    expected = 32 if suite == "shared" else 8
    if len(rows) != expected:
        raise ValueError(f"{path} must contain {expected} prompts")
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {"id", "category", "prompt"}
                or not isinstance(row["id"], str) or not row["id"]
                or row["category"] not in CATEGORIES
                or not isinstance(row["prompt"], str) or not row["prompt"].strip()):
            raise ValueError(f"Invalid prompt row in {path}: {row}")
    return [{"key": f"{suite}:{row['id']}", "prompt": row["prompt"],
             "category": row["category"], "suite": suite} for row in rows]


def tasks(concept: str) -> list[dict]:
    """Common prompts first, then target-specific stress prompts."""
    concept_path = PROMPT_ROOT / f"{concept}.jsonl"
    if concept not in {"fairytale", "french", "pirate", "positive", "russian"}:
        raise ValueError(f"No frozen v1 prompt set for concept: {concept}")
    rows = _source(PROMPT_ROOT / "common.jsonl", "shared")
    rows += _source(concept_path, "concept")
    if (len({row["key"] for row in rows}) != len(rows)
            or len({row["prompt"].casefold() for row in rows}) != len(rows)):
        raise ValueError("Duplicate Judge prompt key or text")
    return rows


def prepare(concept: str, data_root: Path, campaign_config: Path) -> tuple[Path, Path]:
    """Write a frozen prompt snapshot and plan for the shared generator."""
    from hybrid_steering.plans import conditions, write_unchanged as _write_unchanged

    data_root = data_root.resolve()
    config = json.loads(campaign_config.read_text())
    if concept not in config["concepts"]:
        raise ValueError(f"Concept is not in campaign: {concept}")
    train = (data_root / config["training_prompts"] / "train.jsonl").resolve()
    if not train.is_relative_to(data_root):
        raise ValueError("Training prompts escape data root")
    if digest(train) != config["train_prompts_sha256"]:
        raise ValueError("Direction-training prompts changed")
    rows = tasks(concept)
    training = {row["prompt"].casefold() for row in read_jsonl(train)}
    if any(row["prompt"].casefold() in training for row in rows):
        raise ValueError("Judge prompt overlaps direction-training prompts")

    relative = Path("cache/judge_prompts/v1") / f"{concept}.jsonl"
    dataset = data_root / relative
    if dataset.exists():
        if read_jsonl(dataset) != rows:
            raise ValueError(f"Frozen Judge dataset differs: {dataset}")
    else:
        write(dataset, rows)
    plan = {
        "benchmark": "judge_prompts", "concept": concept,
        "model": config["model"], "model_revision": config["model_revision"],
        "placement_config": config["placement_config"],
        "dataset": str(relative), "dataset_sha256": digest(dataset),
        "prompt_sources_sha256": {
            name: digest(PROMPT_ROOT / name)
            for name in ("common.jsonl", f"{concept}.jsonl")
        },
        "expected_n": len(rows), "batch_size": 16, "max_new_tokens": 1024,
        "enable_thinking": False, "conditions": conditions(config["conditions"]),
    }
    plan_path = data_root / "plans/judge-prompts-v1" / concept / "judge_prompts.json"
    _write_unchanged(plan_path, plan)
    return dataset, plan_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concept", required=True, choices=("fairytale", "french", "pirate", "positive"))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--campaign-config", type=Path, default=ROOT / "experiments/campaign/config.json")
    args = parser.parse_args()
    dataset, plan = prepare(args.concept, args.data_root, args.campaign_config)
    print(f"Dataset: {dataset}\nPlan: {plan}")


if __name__ == "__main__":
    main()
