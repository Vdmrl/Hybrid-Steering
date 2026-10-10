"""Priority language sweep: independent GPU workers, four-point selection, IFEval."""

import argparse
import copy
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import pareto
import run as pipeline
from report import build_report

from hybrid_steering import load_direction
from hybrid_steering.scoring import injected_norm
from hybrid_steering.selection import select_spread_levels

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def expression(row):
    score = row.get("concept_score")
    if score is None and row.get("evaluable") is False:
        return 0.0
    if type(score) is not int or not 0 <= score <= 4:
        raise ValueError("Incomplete Judge expression; no guessed score")
    return 0.0 if score == 0 else 0.5 if score <= 2 else 1.0


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No measurements to export")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def select_rows(rows, cases):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"], float(row["scale"])].append(row)
    measurements, selections, final = [], {}, []
    for case in cases:
        cells = []
        for scale in case["scales"]:
            items = grouped[case["name"], float(scale)]
            if len(items) != 50 or len({r["key"] for r in items}) != 50:
                raise ValueError("Every screen condition requires exactly 50 unique scored answers")
            cell = {
                "condition": case["name"],
                "method": case["method"],
                "strength": scale,
                "expression": sum(expression(r) for r in items) / 50,
                "n": 50,
            }
            measurements.append(cell)
            cells.append(cell)
        if case["method"] == "baseline":
            final.append(copy.deepcopy(case))
        else:
            selection = select_spread_levels(cells)
            selections[case["name"]] = selection
            final.append({**case, "scales": [p["strength"] for p in selection["selected"]]})
    return final, selections, measurements


def conditions(model, concept, recurrent, residual, layer):
    direction, *_ = load_direction(recurrent)
    full = injected_norm(direction, None)
    family = model["family"]
    grids = json.loads((HERE / "language-grids.json").read_text())[concept["id"]]
    result = [{"name": "baseline", "method": "baseline", "scales": [0]}]
    for label, rank, mode in [
        ("fullrank", None, "add"),
        ("rank1", 1, "add"),
        ("rank1-clamp", 1, "clamp"),
    ]:
        case = {
            "name": label,
            "method": f"{family}_{mode}",
            "direction": str(recurrent),
            "normalize": False,
            "gain": full / max(injected_norm(direction, rank), 1e-8),
            "scales": grids[label],
        }
        if rank is not None:
            case["rank"] = rank
        if mode == "add":
            case["prompt_position"] = -1
        result.append(case)
    result.append(
        {
            "name": f"residual-L{layer}",
            "method": "residual_add",
            "direction": str(residual),
            "layer": layer,
            "scales": grids["residual"],
        }
    )
    return result


def plots(rows, folder, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"rank1": "#148692", "rank1-clamp": "#7860ad", "fullrank": "#dca23b"}
    names = sorted({r["condition"] for r in rows if r["condition"] != "baseline"})
    comparisons = [(a, b) for a, b in itertools.combinations(names, 2)]
    sets = [("pareto", names)] + [(a + "-vs-" + b, [a, b]) for a, b in comparisons]
    for filename, methods in sets:
        fig, ax = plt.subplots(figsize=(6.5, 4.3))
        for i, name in enumerate(methods):
            points = sorted([r for r in rows if r["condition"] == name], key=lambda r: r["scale"])
            if not points:
                continue
            x = [100 * r["judge_mean"] for r in points]
            y = [100 * r["benchmark_score"] for r in points]
            ax.plot(
                x,
                y,
                "-o",
                label=name,
                color=colors.get(name, "#bb5148"),
                linewidth=1.5,
                markersize=4,
            )
            ax.errorbar(
                x,
                y,
                xerr=[
                    [100 * (r["judge_mean"] - r["judge_low"]) for r in points],
                    [100 * (r["judge_high"] - r["judge_mean"]) for r in points],
                ],
                fmt="none",
                ecolor=colors.get(name, "#bb5148"),
                alpha=0.35,
                capsize=2,
            )
        for baseline in [r for r in rows if r["condition"] == "baseline"]:
            ax.axhline(100 * baseline["benchmark_score"], color="#999", linestyle="--", linewidth=1)
        ax.set(
            xlim=(-2, 102),
            ylim=(-2, 102),
            xlabel="Expression (%)",
            ylabel="IFEval prompt-strict (%)",
            title=title,
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
        ax.legend(frameon=False)
        fig.text(
            0.5,
            0.015,
            "95% expression CI; lines connect measured strengths",
            ha="center",
            fontsize=8,
            color="#666",
        )
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        for suffix in ("png", "svg"):
            fig.savefig(folder / f"{filename}.{suffix}", dpi=200)
        plt.close(fig)


def worker(job):
    model, concept = job["model"], job["concept"]
    output = Path(job["output"])
    config = Path(job["config"])
    recurrent, residual = pareto._directions(config, model, concept, output.parent.parent)
    cases = conditions(model, concept, recurrent, residual, job["residual_layer"])
    plan = pareto._plan(
        model,
        concept,
        job["screen"],
        job["benchmark"],
        ROOT / "config/judge.yaml",
        cases,
        job["screen_tokens"],
    )
    plan.update(
        parallel_backend="single",
        model_revision=model["revision"],
        batch_size=job["initial_batch"],
        auto_batch=job["auto_batch"],
    )
    plan["judge"].update(metric="judge_expression", batch_size=job["judge_batch"])
    screen_path = output / "screen.json"
    pareto._write_plan(screen_path, plan)
    screen_dir, ready = pareto._run_role(
        screen_path, output / "screen", "judge", job["run_judge"], model
    )
    if not ready:
        raise RuntimeError(
            "Judge tasks prepared. Resume with --run-judge to obtain expression and IFEval."
        )
    rated = pipeline.read_lines(screen_dir / "judge_scores.jsonl")
    selected, choices, measurements = select_rows(rated, cases)
    provenance = {
        "screen_scores_sha256": pareto._sha(screen_dir / "judge_scores.jsonl"),
        "expression_metric": "judge_two_tier_v1",
        "selection_code_sha256": pareto._sha(ROOT / "src/hybrid_steering/selection.py"),
        "methods": choices,
    }
    pareto._write_plan(output / "selected-ifeval.json", provenance)
    write_csv(
        output / "selected-ifeval.csv",
        [
            {
                **p,
                "status": result["status"],
                "minimum_expression_distance": result["minimum_expression_distance"],
            }
            for result in choices.values()
            for p in result["selected"]
        ],
    )
    write_csv(output / "screen.csv", measurements)
    write_csv(
        output / "per-answer-expression.csv",
        [
            {
                "condition": r["condition"],
                "scale": r["scale"],
                "key": r["key"],
                "concept_score": r.get("concept_score"),
                "expression": expression(r),
                "evaluable": r.get("evaluable"),
                "response": r["response"],
            }
            for r in rated
        ],
    )
    final = {**plan, "conditions": selected, "max_new_tokens": job["ifeval_tokens"]}
    final_path = output / "ifeval.json"
    pareto._write_plan(final_path, final)
    benchmark_dir, _ = pareto._run_role(final_path, output / "ifeval", "benchmark", False, model)
    # Preserve the original null/integer Judge labels. Report-only copy supplies the
    # requested two-tier expression, including zero for explicitly unevaluable rows.
    report_judge = output / "report-expression"
    report_judge.mkdir(exist_ok=True)
    pipeline.write_jsonl(
        report_judge / "judge_scores.jsonl",
        [
            {**r, "concept_score": 0 if r.get("evaluable") is False else r["concept_score"]}
            for r in rated
        ],
    )
    (report_judge / "manifest.json").write_text((screen_dir / "manifest.json").read_text())
    report_dir = build_report(report_judge, benchmark_dir, output, two_tier=True)
    report = json.loads((report_dir / "report.json").read_text())
    rows = report["rows"]
    write_csv(output / "ifeval.csv", rows)
    indexed = {(r["condition"], r["scale"]): r for r in rows}
    write_csv(
        output / "measurements.csv",
        [{**p, **indexed.get((p["condition"], float(p["strength"])), {})} for p in measurements],
    )
    plots(rows, output, f"{model['model'].split('/')[-1]} · {concept['id']}")
    (output / "COMPLETE.json").write_text(
        json.dumps({"report": str(report_dir), "points": len(rows)}) + "\n"
    )


def available_gpus(requested=None):
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.split(",") for line in result.stdout.splitlines() if line.strip()]
    wanted = set(requested.split(",")) if requested else None
    devices = []
    for row in rows:
        index, name, total, used, utilization = [v.strip() for v in row]
        if wanted is not None and index not in wanted:
            continue
        if int(used) > 2048 or int(utilization) > 10:
            if wanted is not None:
                raise RuntimeError(f"GPU {index} is occupied; no unrelated work will be stopped")
            continue
        if int(total) < 40000:
            raise RuntimeError(
                f"GPU {index} ({name}) has less than 40GB; this single-GPU plan expects H100-class capacity"
            )
        devices.append(index)
    if not devices or wanted is not None and set(devices) != wanted:
        raise RuntimeError("No matching idle GPU available; verify nvidia-smi")
    return devices


def schedule(jobs, output, devices):
    pending, active = list(jobs), []
    while pending or active:
        free = [device for device in devices if device not in {r[0] for r in active}]
        for device in free:
            if not pending:
                break
            available_gpus(device)
            job = pending.pop(0)
            path = output / f"job-{job['model']['id']}-{job['concept']['id']}.json"
            pareto._write_plan(path, job)
            log = (output / f"{job['model']['id']}-{job['concept']['id']}.log").open("a")
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": device,
                "HYBRID_PARALLEL_BACKEND": "single",
                "HYBRID_MODEL_REVISION": job["model"]["revision"],
            }
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--worker", str(path)],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            active.append((device, process, log, path))
            print(f"GPU {device}: {job['model']['id']} / {job['concept']['id']}", flush=True)
        for item in list(active):
            device, process, log, path = item
            if process.poll() is not None:
                log.close()
                active.remove(item)
                if process.returncode:
                    # Let unrelated successful workers finish; expose failure without
                    # leaving untracked GPU children or falsely claiming completion.
                    for _, remaining, other_log, _ in active:
                        remaining.wait()
                        other_log.close()
                    raise RuntimeError(
                        f"Worker failed ({process.returncode}): {path}; inspect its log"
                    )
        if active:
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--concept-root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/language-priority")
    parser.add_argument(
        "--gpus", help="Idle physical GPU IDs, e.g. 0,1; otherwise detect idle GPUs"
    )
    parser.add_argument("--residual-layer", type=int, default=16)
    parser.add_argument(
        "--run-judge", action="store_true", help="Explicitly allow paid Judge scoring"
    )
    parser.add_argument(
        "--plan", action="store_true", help="Prepare and validate inputs without model execution"
    )
    args = parser.parse_args()
    if args.worker:
        worker(json.loads(args.worker.read_text()))
        return
    from huggingface_hub import HfApi, snapshot_download

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.residual_layer < 0:
        raise ValueError("Residual layer must be nonnegative")
    saved = output / "campaign.json"
    old = json.loads(saved.read_text()) if saved.exists() else {}
    concept_revision = old.get("concept_revision")
    if args.concept_root:
        concept_root = args.concept_root.resolve()
    else:
        repo = "hybrid-steering/hybrid-steering-concepts"
        concept_revision = concept_revision or HfApi().dataset_info(repo).sha
        concept_root = Path(
            snapshot_download(
                repo,
                repo_type="dataset",
                revision=concept_revision,
                allow_patterns=[
                    "concepts/en-ar/data/pairs.jsonl",
                    "concepts/en-fr/data/pairs.jsonl",
                ],
            )
        )
    screen = pareto._dataset(
        Path(__file__), {"path": str(HERE / "prompts/language-screen-50.jsonl")}
    )
    if len(pipeline.read_lines(Path(screen["path"]))) != 50:
        raise ValueError("Language screen must contain 50 prompts")
    benchmark = pareto._dataset(
        Path(__file__),
        {
            "name": "ifeval",
            "path": str(HERE / "vendor/ifeval/input_data.jsonl"),
            "evaluator": {"root": str(HERE / "vendor/ifeval")},
        },
    )
    if len(pipeline.read_lines(Path(benchmark["path"]))) != 541:
        raise ValueError("IFEval must contain 541 prompts")
    models = []
    for ident, model_name, family in [
        ("qwen", "Qwen/Qwen3.5-9B", "gdn"),
        ("falcon", "tiiuae/Falcon-H1-7B-Instruct", "mamba"),
    ]:
        revision = old.get("model_revisions", {}).get(ident) or HfApi().model_info(model_name).sha
        models.append(
            {
                "id": ident,
                "model": model_name,
                "family": family,
                "revision": revision,
                "parallel_backend": "single",
                "batch_size": 8,
            }
        )
    jobs = []
    for language, code in [("arabic", "ar"), ("french", "fr")]:
        concept = {
            "id": language,
            "feature": language + "_expression",
            "pairs": str(concept_root / f"concepts/en-{code}/data/pairs.jsonl"),
            "source": "en",
            "target": code,
        }
        pareto._check_pairs(Path(__file__), concept, screen, screen)
        for model in models:
            jobs.append(
                {
                    "model": model,
                    "concept": concept,
                    "screen": screen,
                    "benchmark": benchmark,
                    "output": str(output / model["id"] / language),
                    "config": str(Path(__file__)),
                    "residual_layer": args.residual_layer,
                    "initial_batch": 16,
                    "auto_batch": {"maximum": 128, "memory_fraction": 0.85},
                    "screen_tokens": 512,
                    "ifeval_tokens": 1024,
                    "judge_batch": 8,
                    "run_judge": True,
                }
            )
    campaign = {
        "jobs": jobs,
        "model_revisions": {m["id"]: m["revision"] for m in models},
        "concept_revision": concept_revision,
    }
    pareto._write_plan(saved, campaign)
    if args.plan:
        print(
            "Prepared 4 studies: 134 positive strengths × 50 screen prompts, then up to 64 IFEval points + 4 baselines. No TP."
        )
        return
    if not args.run_judge:
        raise ValueError(
            "Complete execution requires --run-judge; --plan remains offline with local inputs"
        )
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "Set the existing Judge API credential in the environment; never put it in config"
        )
    devices = available_gpus(args.gpus)
    schedule(jobs, output, devices)
    measurements = []
    links = []
    for job in jobs:
        folder = Path(job["output"])
        with (folder / "measurements.csv").open() as stream:
            measurements.extend(
                {**r, "model": job["model"]["model"], "concept": job["concept"]["id"]}
                for r in csv.DictReader(stream)
            )
        links.append(
            f"- [{job['model']['id']} · {job['concept']['id']}]({folder.relative_to(output)}/pareto.png)"
        )
    write_csv(output / "all-measurements.csv", measurements)
    (output / "README.md").write_text(
        "# Language results\n\n"
        + "\n".join(links)
        + "\n\n[All measurements](all-measurements.csv)\n"
    )
    print(f"Complete: {output / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
