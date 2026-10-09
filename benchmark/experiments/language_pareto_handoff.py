"""Run the frozen RU/FR/AR language-rate sweep and selected benchmarks.

The external concept repository supplies pair text only. This command creates
model-specific directions from the same selected English prompts for both models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from experiments import optimize_benchmarks
from experiments.queue import write_jsonl
from hybrid_steering.extract import sha256
from hybrid_steering.paths import ROOT
from hybrid_steering.plans import write_unchanged


LANGUAGES = {"russian": ("en-ru", "ru", "Russian"),
             "french": ("en-fr", "fr", "French"),
             "arabic": ("en-ar", "ar", "Arabic")}
MODELS = {"qwen": "Qwen/Qwen3.5-9B", "falcon": "tiiuae/Falcon-H1-7B-Instruct"}
DATA_FILES = ("cache/ifeval/input_data.jsonl", "cache/humaneval/HumanEval.jsonl.gz",
              "cache/judge_prompts/language-screen-100.jsonl",
              "cache/judge_prompts/language-confirm-500.jsonl",
              "cache/google_research/instruction_following_eval/__init__.py",
              "cache/google_research/instruction_following_eval/evaluation_lib.py",
              "cache/google_research/instruction_following_eval/instructions.py",
              "cache/google_research/instruction_following_eval/instructions_registry.py",
              "cache/google_research/instruction_following_eval/instructions_util.py")
FROZEN_SHA256 = {
    "cache/ifeval/input_data.jsonl": "67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49",
    "cache/humaneval/HumanEval.jsonl.gz": "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef",
    "cache/judge_prompts/language-screen-100.jsonl": "fc7a1afcc15c2e8a57533b1526aef6a8a2c0da47c043b250c471445011bc81c2",
    "cache/judge_prompts/language-confirm-500.jsonl": "14b1a518044071247451a5ba6e3b197917e1a6c895e792fdc1aac0216ad91c9c",
    "cache/google_research/instruction_following_eval/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "cache/google_research/instruction_following_eval/evaluation_lib.py": "35decc06000718487f44d7deafa6d3f48a8ec0886281edf40162c0265b7d248c",
    "cache/google_research/instruction_following_eval/instructions.py": "60e086f5342a03ce8e18b64bbcccf86308f523c08aa826707a562150a52f3edf",
    "cache/google_research/instruction_following_eval/instructions_registry.py": "ec92d72c264f6d906978613085db262356174300370a3fffe6fefd5969ce9cfc",
    "cache/google_research/instruction_following_eval/instructions_util.py": "a73797261eee5bf447e279d82a2b700b1bdd3cb1193412dbab1270a85832bc6b",
}


def stage_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(target) != sha256(source):
            raise ValueError(f"Frozen input differs: {target}")
    else:
        shutil.copy2(source, target)


def stage_datasets(root: Path) -> None:
    for name in DATA_FILES:
        if sha256(ROOT / "data" / name) != FROZEN_SHA256[name]:
            raise ValueError(f"Frozen bundle input changed: {name}")
        stage_file(ROOT / "data" / name, root / name)


def stage_nltk(root: Path) -> None:
    """Fetch the official evaluator's tokenizer data before any GPU work."""
    import nltk

    folder = root / "cache/nltk"
    folder.mkdir(parents=True, exist_ok=True)
    os.environ["NLTK_DATA"] = str(folder)
    nltk.data.path.insert(0, str(folder))
    for package in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{package}")
        except LookupError:
            if not nltk.download(package, download_dir=str(folder), quiet=True):
                raise RuntimeError(f"Could not fetch NLTK {package} for official IFEval scoring")
            nltk.data.find(f"tokenizers/{package}")


def stage_training(concepts: Path, root: Path, language: str, count: int) -> None:
    folder, _, _ = LANGUAGES[language]
    source = concepts / "concepts" / folder / "data" / "pairs.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"Pull the concept repository before running: {source}")
    pairs = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if len(pairs) != 2000 or len({row["pair_id"] for row in pairs}) != len(pairs):
        raise ValueError(f"Expected 2000 distinct {folder} pairs")
    eligible = [row for row in pairs if row.get("split") == "train"
                and isinstance(row.get("negative_text"), str)
                and row["negative_text"].strip()
                and isinstance(row.get("positive_text"), str)
                and row["positive_text"].strip()]
    if len(eligible) < count:
        raise ValueError(f"Only {len(eligible)} usable training pairs in {source}")
    chosen = sorted(eligible, key=lambda row: (hashlib.sha256(
        row["pair_id"].encode()).hexdigest(), row["pair_id"]))[:count]
    rows = [{"id": row["pair_id"], "prompt": row["negative_text"].strip()}
            for row in chosen]
    if len({row["prompt"].casefold() for row in rows}) != len(rows):
        raise ValueError(f"Duplicate selected English prompts in {source}")
    target = root / "concepts" / language
    write_jsonl(target / "train.jsonl", rows)
    write_unchanged(target / "source.json", {"source": str(source.resolve()),
                    "source_sha256": sha256(source), "selected_pair_ids": [row["id"] for row in rows],
                    "train_sha256": sha256(target / "train.jsonl")})


def dataset(root: Path, name: str, count: int, tokens: int, batch: int) -> dict:
    return {"dataset": name, "sha256": sha256(root / name), "expected_n": count,
            "max_new_tokens": tokens, "batch_size": batch}


def specification(config: dict, root: Path, language: str, model_key: str) -> dict:
    model = MODELS[model_key]
    _, code, english_name = LANGUAGES[language]
    recurrent_prefix = "mamba" if model_key == "falcon" else "gdn"
    recurrent = ["rank1", "rank2", "clamp_rank1", "full"]
    methods = [{"method": f"{recurrent_prefix}_{name}",
                **({"normalize_to_full": True} if name != "full" else {})}
               for name in recurrent]
    methods.extend({"method": name, "layers": [16, 20]}
                   for name in ("residual", "residual_clamp"))
    profile = config["models"][model_key]
    queue = {"model": model, "models": {model: {
        "revision": profile["revision"], "gpu_count": profile["gpu_count"],
        "min_total_mib": profile["min_total_mib"],
        "placement_config": profile["placement_config"]}},
        "enable_thinking": False, "gpu_ids": config["gpu_ids"],
        "gpu_pool": config["gpu_ids"], "max_used_mib": config["max_used_mib"],
        "poll_seconds": config["poll_seconds"], "hf_home": "cache/huggingface",
        "dataset": f"concepts/{language}/train.jsonl", "direction_count": config["train_count"],
        "seed": 42, "concepts": {language: f"Answer entirely in {english_name}. "
                    "Preserve the facts and requested format. Do not mention this language instruction."},
        "extraction": {"pair_max_new_tokens": 256, "prefix_tokens": 32,
                       "batch_size": config["batch_size"],
                       "extraction_batch_size": config["extraction_batch_size"],
                       "residual_layers": [16, 20]},
        "gpu_smoke": True, "runtime_benchmark_batch_size": config["batch_size"]}
    return {"id": config["id"], "concept": language,
            "selection_rule": "nearest", "plot_family_curves": True,
            "concept_metric": {"type": "language_rate", "language": code},
            "targets": [0.3, 0.5, 0.7, 0.9, 0.99],
            "strengths": config["strengths"], "methods": methods,
            "screen": dataset(root, "cache/judge_prompts/language-screen-100.jsonl",
                              100, 512, config["batch_size"]),
            "concept_evaluation": dataset(root, "cache/judge_prompts/language-confirm-500.jsonl",
                                          500, 512, config["batch_size"]),
            "benchmarks": {
                "ifeval": dataset(root, "cache/ifeval/input_data.jsonl", 541, 2048,
                                  config["batch_size"]),
                "humaneval": dataset(root, "cache/humaneval/HumanEval.jsonl.gz", 164,
                                     2048, config["batch_size"])}, "queue": queue}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run"))
    parser.add_argument("--concept-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--config", type=Path, default=ROOT / "config/language_pareto.json")
    args = parser.parse_args()
    root = args.data_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    stage_datasets(root)
    if args.mode == "run":
        stage_nltk(root)
        from hybrid_steering.humaneval_score import check_sandbox
        check_sandbox()
    for language in LANGUAGES:
        stage_training(args.concept_root.resolve(), root, language, config["train_count"])
    for model_key in MODELS:
        for language in LANGUAGES:
            spec = specification(config, root, language, model_key)
            optimize_benchmarks.validate(spec, root)
            print(f"{model_key} {language}: 97 screen conditions × 100 prompts; "
                  "40 selected conditions × (500+541+164) prompts", flush=True)
            if args.mode == "run":
                optimize_benchmarks.run(spec, root, False, None)


if __name__ == "__main__":
    main()
