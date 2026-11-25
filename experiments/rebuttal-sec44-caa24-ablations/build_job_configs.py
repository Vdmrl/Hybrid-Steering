"""Prepare portable prompt banks, pinned directions and deterministic job grids."""

import argparse
import ast
import json
import math
import random
import shutil
from hashlib import file_digest
from itertools import combinations
from pathlib import Path

import torch
from interventions import FEATURES, LAYER_SETS, LAYERS, WINDOWS
from safetensors.torch import load_file

PACKAGE = Path(__file__).resolve().parent
CAA = ".worktrees/caa-fr-p18w32-final-20261011/experiments/caa-fr-p18w32-final-20261011"
HYS = "local/gdn-handoff-20261009/source"
HYS_ARCHIVE = (
    "local/steering-night-check-20261009-0934/qwen9b-gdn-fr-repaired-final-20261008/local/"
    "final-gdn-fr-repaired32-shard0/manifest.json"
)
FORGETTING = ".worktrees/caa-fr-p18w32-final-20261011/experiments/forgetting/questions.py"
PROMPTS = "local/tuning-history-9b-20261010/neutral-prompts.jsonl"
NAME = dict(
    zip(
        FEATURES,
        (
            "numbered",
            "french",
            "technical",
            "theistic_framing",
            "probabilistic_framing",
        ),
        strict=True,
    )
)
REFERENCES = dict(
    zip(
        FEATURES,
        ((16, (8, 16)), (4, (8,)), (8, (16,)), (12, (8,)), (18, (8,))),
        strict=True,
    )
)
MODEL = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
HYS_SHAPE = (1, 32, 128, 128)
CAA_SHAPE = (4096,)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_rows(path):
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for r in rows:
            stream.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")


def simple_questions(source, count=50, seed=42):
    """Port of the reference function, with its literal question pool read by AST."""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    pool = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "SIMPLE_QUESTIONS" for t in node.targets)
    )
    return [
        {"id": f"simple-{i:02d}", "text": q}
        for i, q in enumerate(random.Random(seed).sample(pool, count))
    ]


def canonical(text):
    return " ".join(text.casefold().split()).split(" use plain, familiar words", 1)[0].strip()


def import_assets(root):
    historical = read(root / CAA / "frozen-config.json")
    hys_task = read(root / HYS_ARCHIVE)["task"]
    if historical["revision"] != REVISION or historical["model"] != MODEL:
        raise ValueError("Unexpected model provenance")
    specs = {
        "model": MODEL,
        "revision": REVISION,
        "system_prompt": hys_task["system"],
        "methods": {},
    }
    for method in ("caa24", "hys"):
        sources = {}
        for tag, feature in NAME.items():
            if method == "caa24":
                source = historical["sources"][feature]
                relative = f"{CAA}/{source['path']}"
            else:
                source = hys_task["sources"][feature]
                relative = f"{HYS}/{feature}/rank1.safetensors"
            dest = PACKAGE / "directions" / method / f"{tag}.safetensors"
            original = root / relative
            if sha(original) != source["sha256"]:
                raise ValueError(f"Source hash mismatch for {method}/{tag}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                scratch = dest.with_suffix(".partial")
                shutil.copyfile(original, scratch)
                scratch.replace(dest)
            if sha(dest) != source["sha256"]:
                raise ValueError(f"Packaged direction hash mismatch: {dest}")
            sources[tag] = {
                "path": dest.relative_to(PACKAGE).as_posix(),
                "sha256": source["sha256"],
                "source": relative,
            }
        specs["methods"][method] = sources
    dump(PACKAGE / "directions/manifest.json", specs)

    original = read_rows(root / PROMPTS)
    banks = {
        name: [{"id": r["prompt_id"], "text": r["prompt"]} for r in original if r["bank"] == name]
        for name in ("selection8", "independent16", "heldout32")
    }
    hys_shards = (root / HYS_ARCHIVE).parent.parent.glob(
        "final-gdn-fr-repaired32-shard*/manifest.json"
    )
    archived = {p["id"]: p["text"] for shard in hys_shards for p in read(shard)["task"]["prompts"]}
    caa_final = read(root / "local/caa-p18-core-judge-20261012/source-manifest.json")["task"]
    if (
        archived != {p["id"]: p["text"] for p in banks["heldout32"]}
        or archived != {p["id"]: p["text"] for p in caa_final["prompts"]}
        or caa_final["system"] != specs["system_prompt"]
    ):
        raise ValueError("Held-out prompt diverges from original shard")
    banks["simple50"] = simple_questions(root / FORGETTING)
    validate_banks(banks)
    for name, bank in banks.items():
        write_rows(PACKAGE / "banks" / f"{name}.jsonl", bank)


def validate_banks(banks):
    ids, stems = set(), set()
    for name, size in (
        ("selection8", 8),
        ("independent16", 16),
        ("heldout32", 32),
        ("simple50", 50),
    ):
        current = banks[name]
        bank_ids = {r["id"] for r in current}
        bank_stems = {canonical(r["text"]) for r in current}
        if (
            len(current) != size
            or len(bank_ids) != size
            or len(bank_stems) != size
            or ids & bank_ids
            or stems & bank_stems
            or "" in bank_stems
        ):
            raise ValueError(f"Unexpected size or overlapping prompt in {name}")
        ids |= bank_ids
        stems |= bank_stems


def load_directions(method):
    manifest = read(PACKAGE / "directions/manifest.json")
    result = {}
    shape = CAA_SHAPE if method == "caa24" else HYS_SHAPE
    stored = range(32) if method == "caa24" else LAYERS
    specs = manifest["methods"][method]
    if set(specs) != set(FEATURES):
        raise ValueError(f"Incomplete direction bank for {method}")
    for tag, record in specs.items():
        path = PACKAGE / record["path"]
        if sha(path) != record["sha256"]:
            raise ValueError(f"Invalid direction hash for {method}/{tag}")
        tensors = load_file(str(path), device="cpu")
        if set(tensors) != {f"layer_{i}" for i in stored}:
            raise ValueError(f"Unexpected stored layers for {method}/{tag}")
        result[tag] = {}
        for i in stored:
            tensor = tensors[f"layer_{i}"]
            norm = tensor.norm().item()
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype != torch.float32
                or not bool(torch.isfinite(tensor).all())
                or not math.isfinite(norm)
                or norm <= 0
            ):
                raise ValueError(f"Invalid direction {method}/{tag}/layer_{i}")
            result[tag][i] = tensor
    return result


CALIBRATION_GRID = {
    "N": (0.05, 0.09, 0.16, 0.28, 0.50, 0.85),
    "F": (0.03, 0.05, 0.085, 0.14, 0.24, 0.40),
    "T": (0.04, 0.07, 0.12, 0.20, 0.33, 0.55),
    "R": (0.04, 0.07, 0.13, 0.23, 0.40, 0.70),
    "P": (0.08, 0.15, 0.28, 0.52, 0.95, 1.75),
}
DOSE_LAYERS = (0, 8, 16, 30)


def calibration(directions):
    result = {}
    for tag, (gain, ref) in REFERENCES.items():
        norms = {i: float(directions[tag][i].double().norm()) for i in LAYERS}
        each = gain / math.sqrt(len(ref))
        ratios = [each / norms[i] for i in ref]
        anchor = math.exp(sum(math.log(v) for v in ratios) / len(ratios))
        grid = list(CALIBRATION_GRID[tag])
        result[tag] = {
            "legacy_gain": gain,
            "reference_layers": list(ref),
            "gain_per_reference_layer": each,
            "reference_norms": {i: norms[i] for i in ref},
            "native_alpha_anchor": anchor,
            "layer_norms": norms,
            "layer_l2_rss": math.sqrt(sum(v * v for v in norms.values())),
            "grid": grid,
            "dose_l2_at_grid_endpoints": {
                i: {
                    "alpha_min": grid[0],
                    "l2_at_min": grid[0] * norms[i],
                    "alpha_max": grid[-1],
                    "l2_at_max": grid[-1] * norms[i],
                }
                for i in DOSE_LAYERS
            },
        }
    return result


def condition(method, mode, active, strength=1.0, alpha=None, suffix=""):
    key = f"{method}_{mode}__{active or 'baseline'}__l{strength:g}{suffix}"
    c = {
        "id": key,
        "method": method,
        "mode": mode,
        "active": active,
        "lambda": strength,
    }
    if method == "caa24":
        c["alpha"] = alpha
    return c


def log_grid(low, high, n=8):
    return [float(f"{low * (high / low) ** (i / (n - 1)):.6g}") for i in range(n)]


def mega_conditions():
    natural = {m: [0.02, 0.035, 0.06, 0.10, 0.17, 0.28, 0.46, 0.75, 1.20, 1.90] for m in FEATURES}
    natural["F"] = [0.015, 0.025, 0.04, 0.065, 0.105, 0.17, 0.27, 0.43, 0.68, 1.10]
    natural["P"] = [0.03, 0.055, 0.10, 0.18, 0.32, 0.56, 1.00, 1.75, 3.00, 5.00]
    unit = {m: [1.0, 1.6, 2.5, 4.0, 6.3, 10, 15.8, 25, 40, 63] for m in FEATURES}
    unit["F"] = [0.6, *unit["F"][:-1]]
    mid_bounds = {"F": (0.02, 1.6), "P": (0.05, 6.0)}
    cells = [condition("caa24", "rss", "", alpha={})]

    def add(layer_set, norm, grids, windows=False, supplemental=False):
        family = f"{layer_set}_{norm}" + ("_win" if windows else "")
        for tag, doses in grids.items():
            for dose in doses:
                cells.append(
                    {
                        **condition(
                            "caa24", "rss", tag, alpha={tag: dose}, suffix=f"__{family}__{dose:g}"
                        ),
                        "family": family,
                        "layer_set": layer_set,
                        "layers": list(LAYER_SETS[layer_set]),
                        "norm_mode": norm,
                        "use_windows": windows,
                        "windows": WINDOWS if windows else {},
                        "alpha_or_gain": dose,
                        "supplemental": supplemental,
                    }
                )

    for name, layers in LAYER_SETS.items():
        if name == "gdn24":
            add(name, "natural", natural)
            add(name, "unit", unit)
        else:
            scale = 24 / len(layers)
            scaled = {}
            for m, grid in natural.items():
                lo, hi = grid[0] * scale, grid[-1] * scale
                if name != "all32":
                    floor, ceiling = mid_bounds.get(m, (0.03, 3.0))
                    lo, hi = max(floor, lo), min(ceiling, hi)
                scaled[m] = log_grid(lo, hi)
            add(name, "natural", scaled)
            add(
                name,
                "unit",
                {m: log_grid(0.6, 32) if m == "F" else log_grid(1, 50) for m in FEATURES},
            )
        if name in ("gdn24", "all32"):
            # The specified core grids total 468 cells. Add five per family
            # to reach the requested 478 without replacing any core point.
            add(
                name,
                "natural",
                {
                    m: [float(f"{math.sqrt(grid[0] * grid[1]):.6g}")]
                    for m, grid in (natural if name == "gdn24" else scaled).items()
                },
                supplemental=True,
            )
    add(
        "gdn24",
        "natural",
        {m: log_grid(natural[m][0], natural[m][-1]) for m in WINDOWS},
        windows=True,
    )
    for name in ("gdn24", "mid12_gdn"):
        add(name, "unit", {m: log_grid(1.5, 63) for m in WINDOWS}, windows=True)
    return cells


def build_jobs(grid):
    subsets = ["".join(c) for k in range(1, 6) for c in combinations(FEATURES, k)]
    reduced = [(s, 1.0) for s in subsets if len(s) in (1, 4, 5)] + [(FEATURES, 0.75)]
    full = [(s, 1.0) for s in ["", *subsets]] + [(FEATURES, 0.75)]
    independent = [("", 1.0), (FEATURES, 0.75), (FEATURES, 1.0)]

    def conditions(method, mode, pairs):
        return [condition(method, mode, subset, lam) for subset, lam in pairs]

    jobs = [
        (
            "job1_caa24_calib_selection8",
            "selection8",
            8,
            1024,
            [
                condition("caa24", "rss", m, alpha={m: a}, suffix=f"__a{i}")
                for m in FEATURES
                for i, a in enumerate(grid[m]["grid"], 1)
            ],
        ),
        (
            "job1_caa_mega_calib_selection8",
            "selection8",
            8,
            1024,
            mega_conditions(),
        ),
        (
            "job1b_caa24_val_independent16",
            "independent16",
            16,
            1024,
            conditions("caa24", "rss", [(m, 1.0) for m in FEATURES]),
        ),
        (
            "job4a_hys_external_simple50",
            "simple50",
            50,
            4096,
            conditions("hys", "full", [("", 1.0), (FEATURES, 0.5), *independent[1:]]),
        ),
        (
            "job2_hys_ablations_and_ctrl",
            "heldout32",
            32,
            4096,
            [
                c
                for mode in ("no_rss", "no_gram", "no_clamp")
                for c in conditions("hys", mode, reduced)
            ]
            + conditions("hys", "full", independent[:2]),
        ),
        (
            "job3_caa24_final_template",
            "heldout32",
            32,
            4096,
            conditions("caa24", "rss", full) + conditions("caa24", "raw", reduced),
        ),
        (
            "job4_external_lambda_simple50",
            "simple50",
            50,
            4096,
            conditions("hys", "full", independent) + conditions("caa24", "rss", independent),
        ),
    ]
    for job_id, bank, n, limit, cells in jobs:
        assert len({c["id"] for c in cells}) == len(cells)
        dump(
            PACKAGE / "jobs" / f"{job_id}.json",
            {
                "schema_version": 1,
                "job_id": job_id,
                "bank": bank,
                "max_new_tokens": limit,
                "norm_mode": "natural",
                "use_windows": False,
                "expected_answers": n * len(cells),
                "conditions": cells,
            },
        )
    template = PACKAGE / "jobs/selected_alpha.example.json"
    dump(template, {tag: None for tag in FEATURES})
    dump(
        PACKAGE / "jobs/selected_caa_config.json",
        {
            "family": "gdn24_natural",
            "layers": list(LAYERS),
            "norm_mode": "natural",
            "use_windows": False,
            "windows": {},
            "values": dict.fromkeys(FEATURES),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--import-from", type=Path, help="One-time import from the audited repository"
    )
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    if args.import_from:
        import_assets(args.import_from.resolve())
    data = calibration(load_directions("caa24"))
    dump(PACKAGE / "jobs/calibration_grid.json", data)
    build_jobs(data)
    for tag, report in data.items():
        print(tag, report["reference_norms"], report["grid"])
