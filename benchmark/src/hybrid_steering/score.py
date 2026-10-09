"""Score complete saved HumanEval answers on a host with a working sandbox.

No model is loaded and no generated code is executed without bubblewrap.
The copied source manifest and each result manifest must match the pinned plan.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from hybrid_steering.artifacts import location
from hybrid_steering.benchmarks import _identity, _saved_for_condition, _score, load_tasks, read_plan
from hybrid_steering.extract import ROOT
from hybrid_steering.steering import Directions, SteeringConfig


def score_saved(plan_path: Path, source_manifest: Path, data_root: Path) -> None:
    from hybrid_steering.humaneval_score import check_sandbox

    check_sandbox()
    plan = read_plan(plan_path)
    if plan["benchmark"] != "humaneval":
        raise ValueError("This offline scorer is for HumanEval only")
    source = json.loads(source_manifest.read_text())
    source_id = source["identity"]
    if (source_id["concept"] != plan["concept"] or source_id["model"] != plan["model"]
            or source_id["model_revision"] != plan["model_revision"]):
        raise ValueError("Direction source does not match the plan")
    directions = Directions({}, {"directions.pt": source["directions_sha256"]},
                            source_id["concept"], source_id["model"], source_id["model_revision"])
    model = SimpleNamespace(config=SimpleNamespace(_commit_hash=plan["model_revision"]))
    tokenizer = SimpleNamespace(init_kwargs={"_commit_hash": source_id["tokenizer_revision"]})
    tasks = load_tasks(plan, data_root)
    for row in plan["conditions"]:
        condition = SteeringConfig(**row)
        identity = _identity(plan, condition, directions, model, tokenizer, data_root)
        folder = location(data_root / "results", identity)
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest["identity"] != identity:
            raise ValueError(f"Result identity differs from pinned plan: {folder}")
        answers = _saved_for_condition(folder, condition, tasks, plan["model"])
        if len(answers) != len(tasks):
            raise ValueError(f"Incomplete saved answers: {folder}")
        _score(plan, folder, condition, tasks, answers, data_root)
        print(f"SCORED {plan['concept']} {condition.method} L{condition.layer} "
              f"c={condition.strength:g}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    os.environ["GDN_DATA_ROOT"] = str(data_root)
    score_saved(args.plan, args.source_manifest, data_root)


if __name__ == "__main__":
    main()
