"""Run the portable, single-GPU §4.4 generation jobs."""

import argparse
import json
import math
import os
import random
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path

import torch
from build_job_configs import PACKAGE, load_directions, read, read_rows, sha, write_rows
from interventions import (
    FEATURES,
    HYS_ALPHA,
    HYS_BETA,
    LAYER_SETS,
    LAYERS,
    WINDOWS,
    RecurrentWrite,
    caa_phases,
    hys_coefficients,
)
from native_runtime import generate, load_model, model_provenance

MODES = {"caa24": ("rss", "raw"), "hys": ("full", "no_rss", "no_gram", "no_clamp")}


def caa_options(config, condition, selected, alpha):
    opts = {**config, **selected, **condition}
    layers = opts.get("layers", list(LAYERS))
    norm = opts["norm_mode"]
    use_windows = opts["use_windows"]
    windows = opts.get("windows", WINDOWS if use_windows else {})
    named = (
        next((name for name, indices in LAYER_SETS.items() if tuple(layers) == indices), "custom")
        if isinstance(layers, list)
        else "invalid"
    )
    layer_set = opts.get("layer_set", named)
    family = opts.get("family", f"{layer_set}_{norm}" + ("_win" if use_windows else ""))
    if (
        not isinstance(layers, list)
        or not layers
        or any(type(i) is not int or not 0 <= i < 32 for i in layers)
        or len(set(layers)) != len(layers)
        or layer_set != named
        or norm not in ("natural", "unit")
        or type(use_windows) is not bool
        or not isinstance(family, str)
        or not family
        or not isinstance(windows, dict)
        or (
            use_windows
            and (
                not windows
                or any(m not in WINDOWS or type(n) is not int or n <= 0 for m, n in windows.items())
            )
        )
        or (not use_windows and windows)
    ):
        raise ValueError(f"Invalid CAA layers, norm or windows: {family}")
    dose = next(iter(alpha.values())) if len(alpha) == 1 else (alpha or None)
    if "alpha_or_gain" in condition and condition["alpha_or_gain"] != dose:
        raise ValueError("CAA dose metadata does not match alpha")
    return dict(
        family=family,
        layer_set=layer_set,
        layers=layers,
        norm_mode=norm,
        use_windows=use_windows,
        windows=windows,
        alpha_or_gain=dose,
    )


def heterogeneous_options(specs, active, family):
    if not isinstance(specs, dict) or set(specs) != set(active):
        raise ValueError("Per-concept settings must match the active features")
    for s in specs.values():
        if not isinstance(s, dict) or not {"layers", "norm_mode", "value", "window"} <= s.keys():
            raise ValueError("Per-concept settings need layers, norm_mode, value and window")
        layers, dose, window = s["layers"], s["value"], s["window"]
        if (
            not isinstance(layers, list)
            or not layers
            or any(type(i) is not int or not 0 <= i < 32 for i in layers)
            or len(set(layers)) != len(layers)
            or s["norm_mode"] not in ("natural", "unit")
            or type(dose) not in (int, float)
            or not math.isfinite(dose)
            or dose <= 0
            or (window is not None and (type(window) is not int or window <= 0))
        ):
            raise ValueError(f"Invalid per-concept settings: {s}")
    scoped = {m: specs[m] for m in active}
    layers = sorted({i for s in scoped.values() for i in s["layers"]})
    windows = {m: s["window"] for m, s in scoped.items() if s["window"] is not None}
    alpha = {m: s["value"] for m, s in scoped.items()}
    return alpha, dict(
        family=family,
        layers=layers,
        layer_set=next((name for name, ix in LAYER_SETS.items() if tuple(layers) == ix), "custom"),
        norm_mode="per_concept",
        use_windows=bool(windows),
        windows=windows,
        per_concept=scoped,
        alpha_or_gain=next(iter(alpha.values())) if len(alpha) == 1 else alpha,
    )


def selected_specs(selection):
    if not isinstance(selection, dict):
        raise ValueError("Selected CAA config must be a JSON object")
    specs = selection.get("per_concept", selection)
    heterogeneous_options(specs, FEATURES, "selection")
    return specs


def scheduled_tasks(config, prompts):
    all_tasks = {(c["id"], p["id"]): (c, p) for c in config["conditions"] for p in prompts}
    if "tasks" not in config:
        return all_tasks
    tasks = [tuple(pair) for pair in config["tasks"]]
    if not tasks or len(set(tasks)) != len(tasks) or any(key not in all_tasks for key in tasks):
        raise ValueError("Shard must contain unique known condition/question pairs")
    return {key: all_tasks[key] for key in tasks}


def resolve_job(config, prompts, gains=None):
    if (
        config["schema_version"] != 1
        or not config["conditions"]
        or len({p["id"] for p in prompts}) != len(prompts)
        or not all(p["id"] and p["text"].strip() for p in prompts)
        or len({c["id"] for c in config["conditions"]}) != len(config["conditions"])
        or len(scheduled_tasks(config, prompts)) != config["expected_answers"]
    ):
        raise ValueError("Conditions and prompts must be unique and match expected_answers")
    if (
        type(config["max_new_tokens"]) is not int
        or not 1 <= config["max_new_tokens"] <= 4096
        or config["norm_mode"] not in ("natural", "unit")
        or type(config["use_windows"]) is not bool
    ):
        raise ValueError("Invalid decoding or CAA options")
    if gains is not None and not isinstance(gains, dict):
        raise ValueError("Selected CAA config must be a JSON object")
    selected, per_selected, named = {}, None, {}
    selection_name = "per_concept"
    if gains is not None and (
        "per_concept" in gains
        or (set(gains) == set(FEATURES) and any(isinstance(v, dict) for v in gains.values()))
    ):
        per_selected = selected_specs(gains)
        selection_name = gains.get("family", selection_name)
        gains = None
    if gains is not None and "values" in gains:
        required = {"family", "layers", "norm_mode", "use_windows", "windows"}
        if not required <= gains.keys():
            raise ValueError("Missing selected CAA family, layers, norm or windows")
        selected = {k: gains[k] for k in required}
        gains = gains["values"]
        if not isinstance(gains, dict):
            raise ValueError("Selected CAA values must be a feature-to-dose map")
    if gains is not None and (
        set(gains) != set(FEATURES)
        or any(
            type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in gains.values()
        )
    ):
        raise ValueError("--gains requires five finite positive numeric values N/F/T/R/P")
    resolved = []
    for c in config["conditions"]:
        m, v, active, strength = c["method"], c["mode"], c["active"], c["lambda"]
        if (
            v not in MODES.get(m, ())
            or not c["id"]
            or not isinstance(active, str)
            or active != "".join(f for f in FEATURES if f in active)
            or type(strength) not in (int, float)
            or not math.isfinite(strength)
            or strength <= 0
        ):
            raise ValueError(f"Invalid condition: {c}")
        specs = c.get("per_concept")
        if m == "caa24" and active and specs is None:
            if (ref := c.get("caa_config")) is not None:
                if ref not in ("best_multilayer", "gdn24_strict"):
                    raise ValueError(f"Unknown packaged CAA selection {ref}")
                if ref not in named:
                    named[ref] = selected_specs(
                        read(PACKAGE / "jobs/configs" / f"selected_caa_{ref}.json")
                    )
                specs = {f: named[ref][f] for f in active}
            elif per_selected is not None and c.get("alpha") is None:
                specs = {f: per_selected[f] for f in active}
        if specs is not None:
            if m != "caa24" or not active:
                raise ValueError("Only active CAA conditions accept per-concept settings")
            alpha, options = heterogeneous_options(
                specs,
                active,
                c.get("family", c.get("caa_config", selection_name)),
            )
            if c.get("alpha") is not None and c["alpha"] != alpha:
                raise ValueError("Frozen alpha conflicts with per-concept values")
        elif m == "hys":
            if "alpha" in c:
                raise ValueError("HyS Table 4 gains are fixed by the runner")
            alpha = {f: HYS_ALPHA[f] for f in active}
            options = {}
        else:
            if active and c.get("alpha") is None and gains is None:
                raise ValueError("Template needs --gains selected on Job 1 before running")
            alpha = c.get("alpha")
            if alpha is None:
                alpha = {f: gains[f] for f in active}
            if set(alpha) != set(active) or any(
                type(x) not in (float, int) or not math.isfinite(x) or x <= 0
                for x in alpha.values()
            ):
                raise ValueError(f"Invalid CAA coefficients for {c['id']}")
            options = caa_options(config, c, selected if c.get("alpha") is None else {}, alpha)
        resolved.append(
            {
                **c,
                **options,
                "alpha": {f: alpha[f] for f in active},
                "beta": {f: HYS_BETA[f] for f in active} if m == "hys" else {},
            }
        )
    return {**config, "conditions": resolved}


def plan(job, banks, device):
    prepared = {}
    for c in job["conditions"]:
        method, mode, alpha = c["method"], c["mode"], c["alpha"]
        args = banks[method], alpha, c["lambda"]
        if not alpha:
            prepared[c["id"]] = (
                None,
                {
                    "min": 1.0,
                    "median": 1.0,
                    "max": 1.0,
                    "per_layer": dict.fromkeys(c.get("layers", LAYERS), 1.0),
                },
            )
        elif method == "caa24":
            prepared[c["id"]] = caa_phases(
                *args,
                c["norm_mode"],
                mode,
                c["use_windows"],
                device,
                c["layers"],
                c["windows"],
                per_concept=c.get("per_concept"),
            )
        else:
            prepared[c["id"]] = hys_coefficients(*args, mode)
    return prepared


def identity(job, prompts, directions, runtime, system):
    return {
        "job": job,
        "prompts": prompts,
        "directions": directions,
        "runtime": runtime,
        "system_prompt": system,
        "dtype": "bfloat16" if runtime["kind"] != "cpu-mock" else "float32",
        "tp_size": 1,
        "offload": False,
        "decoding": "split-prefill, newline bridge, greedy, thinking disabled",
        "seed": 20260911,
        "gdn_layers": list(LAYERS),
        "judge_api_used": False,
        "code_sha256": {p.name: sha(p) for p in sorted(PACKAGE.glob("*.py"))},
    }


def export_judge(path, generations, fingerprint):
    rows = sorted(generations, key=lambda r: r["answer_id"])
    random.Random(20261009).shuffle(rows)
    # Quality Judge requires one JSONL row per prompt. Trait Judge also accepts grouped rows.
    grouped = {}
    for r in rows:
        p = grouped.setdefault(
            r["prompt_id"],
            {"prompt_id": r["prompt_id"], "scenario": r["prompt"], "answers": []},
        )
        if p["scenario"] != r["prompt"]:
            raise ValueError(f"Conflicting text for {r['prompt_id']}")
        p["answers"].append({"answer_id": r["answer_id"], "text": r["response"]})
    for name in ("judge_input.jsonl", "blind.jsonl"):
        atomic_rows(path / name, grouped.values())
    atomic_json(
        path / "judge_input_manifest.json",
        {
            "run_sha256": fingerprint,
            "answers": len(rows),
            "sha256": sha(path / "judge_input.jsonl"),
            "shuffle_seed": 20261009,
        },
    )


def atomic_json(path, record):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def atomic_rows(path, rows):
    temp = path.with_suffix(path.suffix + ".tmp")
    write_rows(temp, rows)
    temp.replace(path)


def fingerprint_of(frozen):
    return sha256(json.dumps(frozen, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def opaque_id(fingerprint, key):
    return "answer-" + sha256(json.dumps([fingerprint, *key]).encode()).hexdigest()[:24]


@contextmanager
def output_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "run.lock"
    try:
        fd = os.open(lock, os.O_EXCL | os.O_CREAT | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Output is locked: {lock}") from None
    try:
        with os.fdopen(fd, "w", encoding="ascii") as handle:
            handle.write(f"{os.getpid()}\n")
        yield
    finally:
        lock.unlink()


@contextmanager
def model_load_lock():
    path = os.environ.get("HYS_CAA_MODEL_LOAD_LOCK")
    if not path:
        yield
        return
    import fcntl

    with Path(path).open("a") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        yield


def run_job(job, prompts, output, *, resume=False, model_path=None, mock=False, namespace=None):
    methods = sorted({c["method"] for c in job["conditions"]})
    if mock:
        from mock_runtime import mock_backend

        model, tokenizer, device, banks = mock_backend()
        runtime = {"kind": "cpu-mock", "model": "synthetic"}
        sources = {m: "synthetic mock tensors" for m in methods}
    else:
        manifest = read(PACKAGE / "directions/manifest.json")
        sources = {m: manifest["methods"][m] for m in methods}
        banks = {m: load_directions(m) for m in methods}
        runtime = {"kind": "single-native", **model_provenance(model_path)}
        device = torch.device("cuda", 0)
    system = read(PACKAGE / "directions/manifest.json")["system_prompt"]
    frozen = identity(job, prompts, sources, runtime, system)
    if namespace:
        frozen["parallel_namespace"] = namespace
    fingerprint = fingerprint_of(frozen)
    with output_lock(output):
        _run_locked(
            job,
            prompts,
            output,
            resume,
            fingerprint,
            frozen,
            banks,
            device,
            (model, tokenizer) if mock else None,
            model_path,
        )


def _run_locked(
    job,
    prompts,
    output,
    resume,
    fingerprint,
    frozen,
    banks,
    device,
    backend,
    model_path,
):
    manifest_path = output / "manifest.json"
    raw = output / "generations.jsonl"
    scheduled = scheduled_tasks(job, prompts)
    namespace = frozen.get("parallel_namespace", fingerprint)

    if raw.exists() and not manifest_path.exists():
        raise ValueError("Missing provenance for existing generations")
    if manifest_path.exists() and (not resume or read(manifest_path)["run_sha256"] != fingerprint):
        raise ValueError(
            "Existing output requires --resume with identical job, inputs, code and runtime"
        )
    saved = read_rows(raw) if raw.exists() else []
    seen = set()
    for row in saved:
        key = row["condition_id"], row["question_id"]
        if key not in scheduled or key in seen or row["answer_id"] != opaque_id(namespace, key):
            raise ValueError("Output contains an unexpected or duplicate generation row")
        seen.add(key)
    if raw.exists() and raw.stat().st_size:
        with raw.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read() != b"\n":
                raise ValueError(
                    "Incomplete final JSONL line. Preserve a backup and check it before resume."
                )
    if len(seen) == len(scheduled) and manifest_path.exists():
        export_judge(output, saved, fingerprint)
        atomic_json(
            manifest_path,
            {**read(manifest_path), "status": "complete", "completed": len(seen)},
        )
        print(f"Already complete: {len(seen)}/{len(scheduled)}")
        return

    if backend is None:
        with model_load_lock():
            model, tokenizer, actual = load_model(model_path)
        if actual != device:
            raise RuntimeError("Unexpected model device")
    else:
        model, tokenizer = backend
    prepared = plan(job, banks, device)
    manifest = {
        **frozen,
        "run_sha256": fingerprint,
        "status": "running",
        "gamma_by_condition": {c: item[1] for c, item in prepared.items()},
        "expected_answers": len(scheduled),
        "completed_at_start": len(seen),
    }
    atomic_json(manifest_path, manifest)
    with raw.open("a", encoding="utf-8", newline="\n") as output_stream:
        for key, (c, prompt) in scheduled.items():
            if key in seen:
                continue
            precomputed = prepared[c["id"]][0]
            recurrent, caa = None, None
            if precomputed and c["method"] == "hys":
                recurrent = RecurrentWrite(banks["hys"], c["alpha"], precomputed, c["mode"], device)
            elif precomputed:
                caa = precomputed
            result = generate(
                model,
                tokenizer,
                device,
                prompt["text"],
                frozen["system_prompt"],
                job["max_new_tokens"],
                caa=caa,
                recurrent=recurrent,
            )
            row = {
                "answer_id": opaque_id(namespace, key),
                "question_id": prompt["id"],
                "prompt_id": prompt["id"],
                "prompt": prompt["text"],
                "condition_id": c["id"],
                "job": c if c["active"] else None,
                "phase": "screen" if c["active"] else "baseline",
                "method": c["method"],
                "mode": c["mode"],
                "active_features": list(c["active"]),
                "lambda": c["lambda"],
                "runtime_mode": frozen["runtime"]["kind"],
                **{
                    k: c[k]
                    for k in (
                        "family",
                        "layer_set",
                        "layers",
                        "norm_mode",
                        "use_windows",
                        "windows",
                        "alpha_or_gain",
                        "per_concept",
                    )
                    if k in c
                },
                **result,
            }
            output_stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
            output_stream.flush()
            os.fsync(output_stream.fileno())
            saved.append(row)
            print(
                f"{len(saved)}/{len(scheduled)} {c['id']} {prompt['id']} {row['finish_reason']}",
                flush=True,
            )
    export_judge(output, saved, fingerprint)
    atomic_json(manifest_path, {**manifest, "status": "complete", "completed": len(saved)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("caa24", "hys", "job", "smoke-cpu"))
    parser.add_argument("--job", type=Path)
    parser.add_argument("--model", type=Path, help="Local BF16 Qwen3.5-9B snapshot")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gains", "--selected-caa-config", type=Path)
    parser.add_argument("--mock", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--parallel-namespace", help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--norm-mode", choices=("natural", "unit"))
    parser.add_argument("--composition-mode", choices=MODES["caa24"])
    parser.add_argument("--use-windows", action="store_true", default=None)
    parser.add_argument("--hys-mode", choices=MODES["hys"])
    parser.add_argument(
        "--single-gpu",
        action="store_true",
        default=True,
        help="Single native GPU is the default",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate all inputs on CPU without generation",
    )
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    if args.command == "smoke-cpu":
        from mock_runtime import smoke_config

        if any(
            getattr(args, key) is not None
            for key in (
                "job",
                "model",
                "gains",
                "max_new_tokens",
                "norm_mode",
                "composition_mode",
                "hys_mode",
                "use_windows",
            )
        ):
            parser.error("smoke-cpu uses its own fixed 2-prompt, 32-token job")
        cfg, prompts = smoke_config()
    else:
        if not args.job:
            parser.error("--job is required")
        cfg = read(args.job)
        prompts = read_rows(PACKAGE / "banks" / f"{cfg['bank']}.jsonl")
        methods = {c["method"] for c in cfg["conditions"]}
        if args.command != "job" and methods != {args.command}:
            parser.error("Subcommand does not match this job. Use job for mixed workloads.")
        for name, method in (("composition_mode", "caa24"), ("hys_mode", "hys")):
            override = getattr(args, name)
            if override:
                if methods != {method} or len({c["mode"] for c in cfg["conditions"]}) != 1:
                    parser.error(
                        f"--{name.replace('_', '-')} needs a single-method, single-mode job"
                    )
                for c in cfg["conditions"]:
                    c["mode"] = override
        for name in ("norm_mode", "use_windows", "max_new_tokens"):
            if getattr(args, name) is not None:
                cfg[name] = getattr(args, name)
    job = resolve_job(cfg, prompts, read(args.gains) if args.gains else None)
    if args.check_config:
        if args.command != "smoke-cpu":
            for method in {c["method"] for c in job["conditions"]}:
                load_directions(method)
        print(
            f"Valid: {len(job['conditions'])} conditions, {len(prompts)} prompts, {job['expected_answers']} answers"
        )
        return
    mock = args.command == "smoke-cpu" or args.mock
    if not args.output or (not mock and not args.model):
        parser.error("--output and, for a GPU run, --model are required")
    run_job(
        job,
        prompts,
        args.output,
        resume=args.resume,
        model_path=args.model,
        mock=mock,
        namespace=args.parallel_namespace,
    )


if __name__ == "__main__":
    main()
