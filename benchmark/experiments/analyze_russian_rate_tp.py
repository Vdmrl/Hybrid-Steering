"""Audit complete paired Russian-rate answers and report 95% intervals."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from hybrid_steering.study import paired_interval
from hybrid_steering.paths import ROOT


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("Wilson interval requires 0 <= successes <= n")
    if successes == 0:
        return [0.0, z * z / (n + z * z)]
    if successes == n:
        return [n / (n + z * z), 1.0]
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def analyze(config: dict, root: Path, runner_sha256: str | None = None) -> dict:
    dataset = root / config["dataset"]
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != config["dataset_sha256"]:
        raise ValueError("Evaluation dataset changed")
    prompts = {row["key"]: row["prompt"] for row in map(json.loads, dataset.read_text().splitlines())}
    if len(prompts) != config["expected_n"]:
        raise ValueError("Incomplete or duplicate dataset")
    base = root / "results/9b/russian"
    runner_hash = runner_sha256 or hashlib.sha256(
        (ROOT / "experiments/tp_russian_rate.py").read_bytes()).hexdigest()
    report = []
    paired = {}
    for condition in config["conditions"]:
        matches = []
        for path in base.rglob("manifest.json"):
            manifest = json.loads(path.read_text())
            identity = manifest.get("identity", {})
            if (identity.get("dataset_sha256") == config["dataset_sha256"]
                    and identity.get("direction_sha256") == {"directions.pt": config["direction_sha256"]}
                    and identity.get("parallel_backend") == "torch_tp_replicated_gdn_v1"
                    and identity.get("tp_runner_sha256") == runner_hash
                    and identity.get("model_revision") == config["model_revision"]
                    and identity.get("method") == condition["method"]
                    and identity.get("layer") == condition.get("layer", -1)
                    and float(condition["strength"]) in manifest.get("strengths", [])):
                matches.append(path.parent)
        if len(matches) != 1:
            raise ValueError(f"Expected one matching result for {condition}, found {len(matches)}")
        folder = matches[0]
        answers = [row for row in map(json.loads, (folder / "answers.jsonl").read_text().splitlines())
                   if float(row["strength"]) == condition["strength"]]
        labels = json.loads((folder / f'language-scores-c{condition["strength"]:g}.json').read_text())
        by_key = {row["key"]: row for row in answers}
        language = {row["key"]: row for row in labels}
        if (len(by_key) != len(answers) or len(language) != len(labels)
                or set(by_key) != set(prompts) or set(language) != set(prompts)):
            raise ValueError(f"Missing or duplicate paired keys: {condition}")
        for key, answer in by_key.items():
            if (answer["prompt"] != prompts[key] or
                    language[key]["response_sha256"] != hashlib.sha256(answer["response"].encode()).hexdigest()):
                raise ValueError(f"Prompt/label differs from saved answer: {key}")
        flags = {key: int(language[key]["language"] == "ru") for key in prompts}
        successes = sum(flags.values())
        report.append({**condition, "n": len(prompts), "russian": successes,
                       "russian_rate": successes / len(prompts),
                       "ci95_wilson": wilson(successes, len(prompts)), "folder": str(folder)})
        paired[(condition["method"], condition.get("layer", -1), condition["strength"])] = flags
    gdn = paired[("gdn_clamp_rank1", -1, 1.25)]
    residual = paired[("residual", 7, 0.3)]
    keys = sorted(prompts)
    difference = [gdn[key] - residual[key] for key in keys]
    return {"dataset_sha256": config["dataset_sha256"], "n": len(keys), "alpha": 0.05,
            "marginals": report, "primary_pair": {"gdn": "rank1-clamp 1.25",
            "residual": "L7 0.3", "difference": sum(difference) / len(keys),
            "ci95_paired_exact": paired_interval([gdn[key] for key in keys],
                                                   [residual[key] for key in keys], 1, 0.05),
            "discordant": sum(value != 0 for value in difference)}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(json.loads(args.config.read_text()), args.data_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
