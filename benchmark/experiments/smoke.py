"""One real batch per steering family before a queued benchmark sweep."""

import json
from pathlib import Path

from hybrid_steering.benchmarks import load_tasks, read_plan
from hybrid_steering.steering import SteeredModel, SteeringConfig, load_directions


def check(plan_path: Path, directions_path: Path, data_root: Path, model, tokenizer,
          output_path: Path | None = None) -> Path:
    plan = read_plan(plan_path)
    prompt = load_tasks(plan, data_root)[0]["prompt"]
    directions = load_directions(directions_path)
    chosen = representative_conditions(plan["conditions"])
    rows = []
    for item in chosen:
        condition = SteeringConfig(**item)
        response = SteeredModel(model, tokenizer, directions, condition).generate([prompt], 64)[0]
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"Empty GPU smoke response for {item}")
        rows.append({"condition": item, "prompt": prompt, "response": response})
        print(f"SMOKE {condition.method} L{condition.layer} c={condition.strength:g}: "
              f"{len(response)} chars", flush=True)
    output = output_path or directions_path.parent / "smoke.json"
    if output.exists() and json.loads(output.read_text()) != rows:
        raise ValueError("Saved GPU smoke differs")
    if not output.exists():
        output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    return output


def representative_conditions(conditions: list[dict]) -> list[dict]:
    """Take one real GPU example per configured steering method."""
    chosen = []
    seen = set()
    for row in conditions:
        if row["method"] not in seen:
            chosen.append(row)
            seen.add(row["method"])
    return chosen
