"""Residual layer calibration using the existing extraction/generation/Judge pipeline."""

import csv
import json
from collections import defaultdict

import run as pipeline

from hybrid_steering.layer_selection import select_layer


def search(config, suite, model, concept, residual, output, run_judge):
    import pareto

    settings = suite["residual_layer_search"]
    layers, scales = settings["layers"], settings["strengths"]
    metric = settings["metric_column"]
    if metric not in {"concept_score", "hit"}:
        raise ValueError("metric_column must explicitly be concept_score or hit")
    direction, *_ = pipeline.load_direction(residual)
    if not set(layers) <= set(direction):
        raise ValueError(f"candidate layers absent from direction: {set(layers) - set(direction)}")
    folder = output / model["id"] / concept["id"] / "residual-layer-search"
    rows = pipeline.read_lines(pareto._path(config, suite["screen_dataset"]["path"]))[:20]
    if len(rows) != 20 or len({r["prompt"] for r in rows}) != 20:
        raise ValueError("need 20 distinct calibration prompts")
    folder.mkdir(parents=True, exist_ok=True)
    prompts = folder / "prompts-20.jsonl"
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    if prompts.exists() and prompts.read_text() != text:
        raise ValueError("calibration prompts changed")
    prompts.write_text(text)
    dataset = {"path": str(prompts.resolve()), "sha256": pareto._sha(prompts)}
    benchmark = pareto._dataset(config, suite["benchmarks"][0])
    conditions = [
        {
            "name": f"residual-L{layer}",
            "method": "residual_add",
            "direction": str(residual),
            "layer": layer,
            "scales": scales,
        }
        for layer in layers
    ]
    plan = pareto._plan(
        model,
        concept,
        dataset,
        benchmark,
        pareto._path(config, suite["judge_config"]),
        conditions,
        512,
    )
    plan["layer_selection"] = settings
    plan_file = folder / "screen.json"
    pareto._write_plan(plan_file, plan)
    scored_dir, ready = pareto._run_role(plan_file, folder / "screen", "judge", run_judge, model)
    if not ready:
        return None
    scores_file = scored_dir / "judge_scores.jsonl"
    groups = defaultdict(list)
    for row in pipeline.read_lines(scores_file):
        groups[(row["condition"], float(row["scale"]))].append(row)
    expected = {(f"residual-L{layer}", float(scale)) for layer in layers for scale in scales}
    if set(groups) != expected:
        raise ValueError("unexpected or missing calibration conditions")
    prompts_by_key = None
    cells = []
    for layer in layers:
        for scale in scales:
            group = groups[(f"residual-L{layer}", float(scale))]
            keys = {r["key"]: r["prompt"] for r in group}
            if len(group) != 20 or len(keys) != 20:
                raise ValueError("incomplete or duplicated calibration answers")
            if prompts_by_key is None:
                prompts_by_key = keys
            if keys != prompts_by_key or set(keys.values()) != {r["prompt"] for r in rows}:
                raise ValueError("unmatched calibration prompts")
            values = []
            for row in group:
                value = 0 if row.get("evaluable") is False else row.get(metric)
                if type(value) not in {int, float, bool}:
                    raise ValueError("missing metric; resolve Judge errors before selecting")
                values.append(float(value))
            cells.append(
                {"layer": layer, "strength": float(scale), "value": sum(values) / 20, "n": 20}
            )
    result = select_layer(cells, layers, scales, settings.get("relative_threshold", 0.95))
    identity = pipeline.identity(
        plan_file, plan, pipeline.load_plan(plan_file)[1]["judge"], "judge"
    )
    result.update(
        model=model["model"],
        concept=concept["id"],
        metric=metric,
        scores_sha256=pareto._sha(scores_file),
        plan_sha256=pareto._sha(plan_file),
        provenance=identity,
        cells=cells,
    )
    pareto._write_plan(folder / "selected-residual-layer.json", result)
    with (folder / "layer-sweep.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    with (folder / "selected-residual-layer.csv").open("w", newline="") as stream:
        fields = [
            "model",
            "concept",
            "metric",
            "status",
            "layer",
            "strength",
            "value",
            "maximum",
            "threshold",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(result)
    if result["layer"] is None:
        raise ValueError("no expression on any layer; inspect saved layer-sweep.csv")
    return result["layer"]
