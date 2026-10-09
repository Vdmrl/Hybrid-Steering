#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
from hybrid_steering.paths import ROOT as PROJECT_ROOT

import torch
from langdetect import DetectorFactory, LangDetectException, detect

from hybrid_steering import language_benchmark as bench
from hybrid_steering import runner as run


ROOT = bench.ROOT
OUT = ROOT / "language_search"
CONFIG = PROJECT_ROOT / "data/legacy-config/configs/4b/language_search.json"
DetectorFactory.seed = 0


def prompts(config):
    values = [template.format(topic=topic) for template in config["prompt_templates"]
              for topic in config["topics"]]
    return [{"id": i, "prompt": value} for i, value in enumerate(values)]


def truncate(tokenizer, text, limit):
    if not limit:
        return text
    ids = tokenizer(text, add_special_tokens=False).input_ids[:limit]
    return tokenizer.decode(ids, skip_special_tokens=True)


def extract_states(model, tokenizer, rows, answer_fields, config, system, label):
    totals = {source: {name: {} for name in answer_fields}
              for source in config["state_token_limits"]}
    for start in range(0, len(rows), 4):
        batch = rows[start:start + 4]
        for name, limit in config["state_token_limits"].items():
            texts = []
            for row in batch:
                texts += [run.chat(tokenizer, row["prompt"],
                                   truncate(tokenizer, row[field], limit), system)
                          for field in answer_fields.values()]
            states = run.recurrent_states_batch(model, tokenizer, texts, "cuda")
            for layer, state in states.items():
                for offset, language in enumerate(answer_fields):
                    values = state[offset::len(answer_fields)]
                    old = totals[name][language].get(layer, torch.zeros_like(values[0]))
                    totals[name][language][layer] = old + values.sum(0)
        print(f"{label}: {min(start + 4, len(rows))}/{len(rows)}", flush=True)
    return totals


def extract_shard(config, shard, num_shards):
    base = torch.load(ROOT / "artifacts/russian-vs-french.pt", map_location="cpu",
                      weights_only=False)
    pairs = [row for row in bench.read_jsonl(ROOT / "generated/pairs.jsonl")
             if row["id"] % num_shards == shard]
    model, tokenizer = bench.load_model(base["model"])
    system = bench.load_config()["system"]
    totals = extract_states(model, tokenizer, pairs,
                            {"russian": "russian_response", "french": "french_response"},
                            config, system, f"centroids shard={shard}")

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"count": len(pairs), "sources": totals, "fingerprint": base["fingerprint"]},
               OUT / f"centroids-{shard:02d}.pt")


def english_shard(config, shard, num_shards):
    base = torch.load(ROOT / "artifacts/russian-vs-french.pt", map_location="cpu",
                      weights_only=False)
    pairs = [row for row in bench.read_jsonl(ROOT / "generated/pairs.jsonl")
             if row["id"] % num_shards == shard]
    path = OUT / f"english-{shard:02d}.jsonl"
    old = {row["id"]: row for row in bench.read_jsonl(path)}
    missing = [row for row in pairs if row["id"] not in old]
    model, tokenizer = bench.load_model(base["model"])
    system = bench.load_config()["system"]

    for start in range(0, len(missing), config["batch_size"]):
        rows = missing[start:start + config["batch_size"]]
        answers = bench.adaptive_batches(
            rows, config["batch_size"],
            lambda batch: run.generate_batch(
                model, tokenizer,
                [row["prompt"] + "\n\n" + config["english_suffix"] for row in batch],
                None, None, 0, "once", "cuda", config["answer_max_new_tokens"], system,
                enable_thinking=config["enable_thinking"]))
        saved = [{"id": row["id"], "prompt": row["prompt"], "english_response": answer}
                 for row, answer in zip(rows, answers)]
        bench.append_jsonl(path, saved)
        old.update({row["id"]: row for row in saved})
        print(f"english shard={shard}: {len(old)}/{len(pairs)}", flush=True)

    rows = [old[row["id"]] for row in pairs]
    totals = extract_states(model, tokenizer, rows, {"english": "english_response"},
                            config, system, f"english states shard={shard}")
    torch.save({"count": len(rows), "sources": totals, "fingerprint": base["fingerprint"]},
               OUT / f"english-centroids-{shard:02d}.pt")


def merge_centroids(config, num_shards):
    merged = {name: {"russian": {}, "french": {}}
              for name in config["state_token_limits"]}
    count, fingerprint = 0, None
    for shard in range(num_shards):
        part = torch.load(OUT / f"centroids-{shard:02d}.pt", map_location="cpu",
                          weights_only=False)
        fingerprint = fingerprint or part["fingerprint"]
        if fingerprint != part["fingerprint"]:
            raise RuntimeError("Centroid shards use different training pairs")
        count += part["count"]
        for source, languages in part["sources"].items():
            for language, layers in languages.items():
                for layer, value in layers.items():
                    old = merged[source][language].get(layer, torch.zeros_like(value))
                    merged[source][language][layer] = old + value
    if count != 100:
        raise RuntimeError(f"Expected 100 pairs, got {count}")

    sources = {}
    for name, languages in merged.items():
        russian = {layer: value / count for layer, value in languages["russian"].items()}
        french = {layer: value / count for layer, value in languages["french"].items()}
        direction = {layer: russian[layer] - french[layer] for layer in russian}
        stats, rank_one = run.analyze(direction, "cpu")
        midpoint = {layer: (russian[layer] + french[layer]) / 2 for layer in russian}
        anchors = {layer: torch.einsum("hd,hdk->hk", rank_one[layer][0], midpoint[layer])
                   for layer in direction}
        sources[name] = {"direction": direction, "rank_one": rank_one,
                         "anchors": anchors, "stats": stats}
    torch.save({"count": count, "fingerprint": fingerprint, "sources": sources},
               OUT / "directions.pt")
    (OUT / "directions.json").write_text(json.dumps(
        {name: value["stats"] for name, value in sources.items()}, indent=2))


def merge_targets(config, num_shards):
    totals = {source: {language: {} for language in ("russian", "french", "english")}
              for source in config["state_token_limits"]}
    count, fingerprint = 0, None
    for shard in range(num_shards):
        parts = (torch.load(OUT / f"centroids-{shard:02d}.pt", map_location="cpu",
                            weights_only=False),
                 torch.load(OUT / f"english-centroids-{shard:02d}.pt", map_location="cpu",
                            weights_only=False))
        if parts[0]["count"] != parts[1]["count"]:
            raise RuntimeError(f"Mismatched shard {shard}")
        count += parts[0]["count"]
        for part in parts:
            fingerprint = fingerprint or part["fingerprint"]
            if fingerprint != part["fingerprint"]:
                raise RuntimeError("Centroid fingerprints differ")
            for source, languages in part["sources"].items():
                for language, layers in languages.items():
                    for layer, value in layers.items():
                        old = totals[source][language].get(layer, torch.zeros_like(value))
                        totals[source][language][layer] = old + value
    if count != 100:
        raise RuntimeError(f"Expected 100 triplets, got {count}")

    sources = {}
    for source, languages in totals.items():
        means = {language: {layer: value / count for layer, value in layers.items()}
                 for language, layers in languages.items()}
        sources[source] = {}
        for target in ("russian", "french"):
            direction = {layer: means[target][layer] - means["english"][layer]
                         for layer in means[target]}
            stats, rank_one = run.analyze(direction, "cpu")
            midpoint = {layer: (means[target][layer] + means["english"][layer]) / 2
                        for layer in direction}
            anchors = {layer: torch.einsum("hd,hdk->hk", rank_one[layer][0], midpoint[layer])
                       for layer in direction}
            sources[source][target] = {"direction": direction, "rank_one": rank_one,
                                       "anchors": anchors, "stats": stats}
    torch.save({"count": count, "fingerprint": fingerprint, "sources": sources},
               OUT / "target-directions.pt")


def settings(artifact, config):
    result = [{"id": "baseline", "source": "full", "schedule": "once",
               "subset": "all", "strength": 0}]
    for source, values in artifact["sources"].items():
        layers = sorted(values["direction"])
        subsets = {"all": layers, "late12": layers[-12:], "late6": layers[-6:]}
        for schedule in ("once", "clamp"):
            for strength in config["strengths"][schedule]:
                result.append({"id": f"{source}-{schedule}-all-{strength:+g}",
                               "source": source, "schedule": schedule,
                               "subset": "all", "strength": strength,
                               "layers": subsets["all"]})
        for subset in ("all", "late12", "late6"):
            for strength in config["strengths"]["centered"]:
                result.append({"id": f"{source}-centered-{subset}-{strength:+g}",
                               "source": source, "schedule": "centered",
                               "subset": subset, "strength": strength,
                               "layers": subsets[subset]})
    return result


def target_settings(artifact, config):
    return [{"id": f"{source}-{target}-once-{strength:+g}", "source": source,
             "target": target, "schedule": "once", "subset": "all",
             "strength": strength, "layers": sorted(values[target]["direction"])}
            for source, values in artifact["sources"].items()
            for target in ("russian", "french")
            for strength in config["target_strengths"]]


def select(values, layers, device):
    direction = {layer: values["direction"][layer].to(device) for layer in layers}
    rank_one = {layer: tuple(item.to(device) for item in values["rank_one"][layer])
                for layer in layers}
    anchors = {layer: values["anchors"][layer].to(device) for layer in layers}
    return direction, rank_one, anchors


def language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def run_shard(config, shard, num_shards, targets=False):
    artifact_name = "target-directions.pt" if targets else "directions.pt"
    output_name = "target-outputs" if targets else "outputs"
    artifact = torch.load(OUT / artifact_name, map_location="cpu", weights_only=False)
    all_jobs = target_settings(artifact, config) if targets else settings(artifact, config)
    jobs = [job for index, job in enumerate(all_jobs)
            if index % num_shards == shard]
    prompt_rows = prompts(config)
    output = OUT / f"{output_name}-{shard:02d}.jsonl"
    done = {(row["setting"], row["prompt_id"]) for row in bench.read_jsonl(output)}
    benchmark_config = bench.load_config()
    model_id, system = benchmark_config["model"], benchmark_config["system"]
    model, tokenizer = bench.load_model(model_id)

    for job in jobs:
        todo = [row for row in prompt_rows if (job["id"], row["id"]) not in done]
        values = artifact["sources"][job["source"]]
        if "target" in job:
            values = values[job["target"]]
        layers = job.get("layers", sorted(values["direction"]))
        direction, rank_one, anchors = select(values, layers, "cuda")
        for start in range(0, len(todo), config["batch_size"]):
            rows = todo[start:start + config["batch_size"]]
            answers = bench.adaptive_batches(
                rows, config["batch_size"],
                lambda batch: run.generate_batch(
                    model, tokenizer, [row["prompt"] for row in batch], direction, rank_one,
                    job["strength"], job["schedule"], "cuda", config["max_new_tokens"],
                    system, anchors, config["enable_thinking"]))
            saved = [{"setting": job["id"], "source": job["source"],
                      "target": job.get("target", "signed"),
                      "schedule": job["schedule"], "subset": job["subset"],
                      "strength": job["strength"], "prompt_id": row["id"],
                      "prompt": row["prompt"], "response": answer,
                      "language": language(answer)}
                     for row, answer in zip(rows, answers)]
            bench.append_jsonl(output, saved)
            done.update((job["id"], row["id"]) for row in rows)
        print(f"search shard={shard}: {job['id']}", flush=True)


def summarize(config, num_shards, targets=False):
    prefix = "target-outputs" if targets else "outputs"
    rows = [row for shard in range(num_shards)
            for row in bench.read_jsonl(OUT / f"{prefix}-{shard:02d}.jsonl")]
    summary = []
    for setting in sorted({row["setting"] for row in rows}):
        selected = [row for row in rows if row["setting"] == setting]
        if len(selected) != len(prompts(config)):
            raise RuntimeError(f"Incomplete setting {setting}: {len(selected)} rows")
        first = selected[0]
        russian = sum(row["language"] == "ru" for row in selected) / len(selected)
        french = sum(row["language"] == "fr" for row in selected) / len(selected)
        target_name = first.get("target", "signed")
        target = (russian if target_name == "russian" else
                  french if target_name == "french" else
                  russian if first["strength"] > 0 else french if first["strength"] < 0 else 0)
        summary.append({"setting": setting, "source": first["source"],
                        "target": target_name,
                        "schedule": first["schedule"], "subset": first["subset"],
                        "strength": first["strength"], "n": len(selected),
                        "russian_rate": russian, "french_rate": french,
                        "target_rate": target})
    summary.sort(key=lambda row: row["target_rate"], reverse=True)
    metrics_name = "target-metrics.csv" if targets else "metrics.csv"
    with (OUT / metrics_name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)
    bench.write_jsonl(OUT / f"{prefix}.jsonl", rows)
    if targets:
        plot_target_rates(summary)
        save_selected(summary, config)
    print(json.dumps(summary[:20], indent=2), flush=True)


def plot_target_rates(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, target in zip(axes, ("russian", "french")):
        for source in ("prefix32", "prefix64", "full"):
            rows = sorted((row for row in summary
                           if row["target"] == target and row["source"] == source),
                          key=lambda row: row["strength"])
            axis.plot([row["strength"] for row in rows],
                      [row["target_rate"] for row in rows], "o-", label=source)
        axis.set(xlabel="Steering strength", ylabel=f"{target.title()} rate",
                 title=f"One-sided {target.title()}−English direction", ylim=(-0.02, 1.02))
        axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "target-rates.png", dpi=180)
    plt.close(fig)


def save_selected(summary, config):
    choices = []
    sources = config["state_token_limits"]
    if config.get("selected_source"):
        sources = [config["selected_source"]]
    for source_index, source in enumerate(sources):
        for strength in config["target_strengths"]:
            rows = {row["target"]: row for row in summary
                    if row["source"] == source and row["strength"] == strength}
            if set(rows) == {"russian", "french"}:
                choices.append((min(row["target_rate"] for row in rows.values()),
                                -strength, -source_index, source, strength, rows))
    _, _, _, source, strength, rows = max(choices)
    artifact = torch.load(OUT / "target-directions.pt", map_location="cpu",
                          weights_only=False)
    selected = {"model": bench.load_config()["model"], "count": artifact["count"],
                "source": source, "schedule": "once", "strength": strength,
                "directions": {target: artifact["sources"][source][target]["direction"]
                               for target in ("russian", "french")},
                "rank_one": {target: artifact["sources"][source][target]["rank_one"]
                             for target in ("russian", "french")}}
    torch.save(selected, OUT / "selected-directions.pt")
    (OUT / "selected-directions.json").write_text(json.dumps(
        {key: selected[key] for key in ("model", "count", "source", "schedule", "strength")}
        | {"russian_rate": rows["russian"]["russian_rate"],
           "french_rate": rows["french"]["french_rate"]}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("extract", "merge", "run", "summarize",
                                             "english", "merge-targets", "run-targets",
                                             "summarize-targets"))
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=4)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "extract":
        extract_shard(config, args.shard_index, args.num_shards)
    elif args.command == "english":
        english_shard(config, args.shard_index, args.num_shards)
    elif args.command == "merge":
        merge_centroids(config, args.num_shards)
    elif args.command == "merge-targets":
        merge_targets(config, args.num_shards)
    elif args.command == "run":
        run_shard(config, args.shard_index, args.num_shards)
    elif args.command == "run-targets":
        run_shard(config, args.shard_index, args.num_shards, targets=True)
    elif args.command == "summarize-targets":
        summarize(config, args.num_shards, targets=True)
    else:
        summarize(config, args.num_shards)


if __name__ == "__main__":
    main()
