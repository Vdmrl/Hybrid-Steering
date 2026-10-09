#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path
from hybrid_steering.paths import ROOT as PROJECT_ROOT

import torch

from hybrid_steering import runner as run
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


ROOT = Path(os.environ.get("GDN_DATA_ROOT", PROJECT_ROOT / "data"))
CONFIG = PROJECT_ROOT / "data/legacy-config/configs/4b/language_benchmark.json"
IFEVAL_FILES = ("evaluation_lib.py", "instructions.py", "instructions_registry.py",
                "instructions_util.py")


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def config_fingerprint(config):
    fields = {key: config[key] for key in
              ("model", "concept", "system", "enable_thinking",
               "russian_suffix", "french_suffix", "pair_max_new_tokens",
               "pair_retry_max_new_tokens", "pair_do_sample", "seed")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16]


def load_config(path=CONFIG):
    return json.loads(Path(path).read_text())


def load_model(model_id, placement=None, revision=None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    options = {}
    if placement:
        if placement.get('transfer_via_cpu', False):
            from hybrid_steering.cpu_transfer import enable
            enable()
        options = {'device_map': placement.get('device_map', 'auto'),
                   'max_memory': {int(k): v for k, v in placement['max_memory'].items()}}
        if placement.get('attn_implementation'):
            options['attn_implementation'] = placement['attn_implementation']
        if placement.get('use_kernels', False):
            options['use_kernels'] = True
    architecture = AutoConfig.from_pretrained(model_id, revision=revision).model_type
    loader = AutoModelForCausalLM if architecture == "falcon_h1" else AutoModelForImageTextToText
    model = loader.from_pretrained(
        model_id, revision=revision, dtype=torch.bfloat16, low_cpu_mem_usage=True, **options).eval()
    if placement:
        assert all(str(device) not in {'cpu', 'disk'} for device in model.hf_device_map.values()), \
            'Model does not fit on the selected GPUs; CPU/disk offload is not allowed'
    else:
        model = model.cuda()
    from hybrid_steering.architecture import adapt_model
    adapt_model(model)
    return model, tokenizer


def adaptive_batches(items, batch_size, fn):
    """Runs fn(list) and halves a batch on CUDA OOM, preserving item order."""
    output = []
    for start in range(0, len(items), batch_size):
        pending = [items[start:start + batch_size]]
        while pending:
            batch = pending.pop(0)
            try:
                output.extend(fn(batch))
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(batch) == 1:
                    raise
                middle = len(batch) // 2
                print(f"OOM at batch={len(batch)}; retrying {middle}+{len(batch)-middle}",
                      flush=True)
                pending[:0] = [batch[:middle], batch[middle:]]
    return output


def build_prompts(config):
    prompts = [template.format(topic=topic) for template in config["prompt_templates"]
               for topic in config["topics"]]
    if len(prompts) != 100 or len(set(prompts)) != 100:
        raise ValueError(f"Expected 100 unique training prompts, got {len(set(prompts))}")
    rows = [{"id": index, "prompt": prompt} for index, prompt in enumerate(prompts)]
    write_jsonl(ROOT / "generated/train_prompts.jsonl", rows)
    return rows


def prepare(config, download_model=True):
    for name in ("artifacts", "cache", "generated", "logs", "outputs/shards",
                 "scores", "plots", "tmp"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    build_prompts(config)

    package = ROOT / "cache/google_research/instruction_following_eval"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").touch()
    commit = config["ifeval"]["commit"]
    base = f"https://raw.githubusercontent.com/google-research/google-research/{commit}/instruction_following_eval"
    for name in IFEVAL_FILES:
        target = package / name
        if not target.exists():
            urllib.request.urlretrieve(f"{base}/{name}", target)
    input_path = ROOT / "cache/ifeval/input_data.jsonl"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        urllib.request.urlretrieve(config["ifeval"]["input"], input_path)
    if len(read_jsonl(input_path)) != 541:
        raise RuntimeError("The pinned IFEval input must contain 541 prompts")

    if download_model:
        from huggingface_hub import snapshot_download
        snapshot_download(config["model"])
    print(f"Prepared data and caches under {ROOT}", flush=True)


def generate(model, tokenizer, prompts, config, batch_size, max_tokens):
    if config.get("pair_do_sample"):
        answers = adaptive_batches(
            prompts, config["pair_sample_batch_size"],
            lambda batch: sample(model, tokenizer, batch, config,
                                 config["pair_retry_max_new_tokens"]))
        if any(not answer for answer in answers):
            raise RuntimeError("Sampled thinking did not finish before pair_retry_max_new_tokens")
        return answers
    answers = adaptive_batches(
        prompts, batch_size,
        lambda batch: run.generate_batch(model, tokenizer, batch, None, None, 0,
                                         "once", "cuda", max_tokens, config["system"],
                                         enable_thinking=config["enable_thinking"]))
    for index, answer in enumerate(answers):
        if not answer:
            answers[index] = run.generate(
                model, tokenizer, prompts[index], None, None, 0, "once", "cuda",
                config["pair_retry_max_new_tokens"], config["system"],
                enable_thinking=config["enable_thinking"])
    if any(not answer for answer in answers):
        raise RuntimeError("Thinking did not finish before pair_retry_max_new_tokens")
    return answers


@torch.inference_mode()
def sample(model, tokenizer, prompts, config, max_tokens):
    texts = [run.chat(tokenizer, prompt, system=config["system"],
                      enable_thinking=config["enable_thinking"])
             for prompt in prompts]
    batch = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt")
    batch = {key: value.cuda() for key, value in batch.items()}
    width = batch["input_ids"].shape[1]
    torch.manual_seed(config["seed"])
    output = model.generate(**batch, do_sample=True, temperature=1.0,
                            top_p=0.95, top_k=20, max_new_tokens=max_tokens,
                            pad_token_id=tokenizer.pad_token_id)
    return run.decode_generated(tokenizer, output[:, width:].tolist(),
                                config["enable_thinking"])


def train_shard(config, shard, num_shards):
    prompts = build_prompts(config)
    prompts = [row for row in prompts if row["id"] % num_shards == shard]
    pair_path = ROOT / f"generated/pairs-{shard:02d}.jsonl"
    old = {row["id"]: row for row in read_jsonl(pair_path)
           if row["russian_response"] and row["french_response"]}
    missing = [row for row in prompts if row["id"] not in old]
    model, tokenizer = load_model(config["model"])

    batch_size = config["pair_batch_size"]
    for start in range(0, len(missing), batch_size):
        rows = missing[start:start + batch_size]
        conditioned = []
        for row in rows:
            conditioned += [row["prompt"] + "\n\n" + config["russian_suffix"],
                            row["prompt"] + "\n\n" + config["french_suffix"]]
        answers = generate(model, tokenizer, conditioned, config, 2 * batch_size,
                           config["pair_max_new_tokens"])
        saved = []
        for index, row in enumerate(rows):
            saved.append({"id": row["id"], "prompt": row["prompt"],
                          "russian_prompt": conditioned[2 * index],
                          "french_prompt": conditioned[2 * index + 1],
                          "russian_response": answers[2 * index],
                          "french_response": answers[2 * index + 1],
                          "model": config["model"],
                          "enable_thinking": config["enable_thinking"]})
        append_jsonl(pair_path, saved)
        old.update({row["id"]: row for row in saved})
        print(f"pairs shard={shard}: {len(old)}/{len(prompts)}", flush=True)

    pairs = [old[row["id"]] for row in prompts]
    sums, count = {}, 0
    state_batch_size = config["state_batch_size"]
    for start in range(0, len(pairs), state_batch_size):
        rows = pairs[start:start + state_batch_size]
        texts = []
        for row in rows:
            texts += [run.chat(tokenizer, row["prompt"], row["russian_response"], config["system"]),
                      run.chat(tokenizer, row["prompt"], row["french_response"], config["system"])]
        states = run.recurrent_states_batch(model, tokenizer, texts, "cuda")
        for layer, state in states.items():
            delta = (state[0::2] - state[1::2]).sum(0)
            sums[layer] = sums.get(layer, torch.zeros_like(delta)) + delta
        count += len(rows)
        print(f"states shard={shard}: {count}/{len(pairs)}", flush=True)
    artifact = {"count": count, "sum": sums, "ids": [row["id"] for row in pairs],
                "fingerprint": config_fingerprint(config)}
    path = ROOT / f"artifacts/partial-{shard:02d}.pt"
    torch.save(artifact, path)
    print(f"Saved {path}", flush=True)


def merge_direction(config, num_shards):
    pairs_by_id, total, sums = {}, 0, {}
    for shard in range(num_shards):
        pairs_by_id.update({row["id"]: row for row in
                            read_jsonl(ROOT / f"generated/pairs-{shard:02d}.jsonl")
                            if row["russian_response"] and row["french_response"]})
        part = torch.load(ROOT / f"artifacts/partial-{shard:02d}.pt", map_location="cpu",
                          weights_only=False)
        if part["fingerprint"] != config_fingerprint(config):
            raise RuntimeError(f"Stale direction shard {shard}")
        total += part["count"]
        for layer, value in part["sum"].items():
            sums[layer] = sums.get(layer, torch.zeros_like(value)) + value
    pairs = sorted(pairs_by_id.values(), key=lambda row: row["id"])
    if total != 100 or len(pairs) != 100 or len({row["id"] for row in pairs}) != 100:
        raise RuntimeError(f"Expected exactly 100 training pairs, got count={total}, rows={len(pairs)}")
    direction = {layer: value / total for layer, value in sums.items()}
    stats, rank_one = run.analyze(direction, "cpu")
    artifact = {"model": config["model"], "concept": config["concept"], "count": total,
                "fingerprint": config_fingerprint(config), "direction": direction,
                "rank_one": rank_one, "stats": stats,
                "orientation": "positive=Russian, negative=French"}
    torch.save(artifact, ROOT / "artifacts/russian-vs-french.pt")
    write_jsonl(ROOT / "generated/pairs.jsonl", pairs)
    (ROOT / "artifacts/russian-vs-french.json").write_text(
        json.dumps({key: artifact[key] for key in
                    ("model", "concept", "count", "fingerprint", "stats", "orientation")},
                   ensure_ascii=False, indent=2))
    print(json.dumps(artifact["stats"], indent=2), flush=True)


def evaluate_shard(config, shard, num_shards):
    inputs = read_jsonl(ROOT / "cache/ifeval/input_data.jsonl")
    inputs = [row for index, row in enumerate(inputs) if index % num_shards == shard]
    output_path = ROOT / f"outputs/shards/ifeval-{shard:02d}.jsonl"
    existing = {(row["schedule"], row["strength"], row["key"])
                for row in read_jsonl(output_path)}
    artifact = torch.load(ROOT / "artifacts/russian-vs-french.pt", map_location="cpu",
                          weights_only=False)
    if artifact["fingerprint"] != config_fingerprint(config):
        raise RuntimeError("Direction artifact does not match the current config")
    model, tokenizer = load_model(config["model"])
    direction = {layer: value.cuda() for layer, value in artifact["direction"].items()}
    rank_one = {layer: tuple(value.cuda() for value in factors)
                for layer, factors in artifact["rank_one"].items()}
    settings = [("once", strength) for strength in config["strengths"]]
    settings += [("clamp", strength) for strength in config["strengths"] if strength != 0]

    for schedule, strength in settings:
        todo = [row for row in inputs if (schedule, strength, row["key"]) not in existing]
        for start in range(0, len(todo), config["eval_batch_size"]):
            rows = todo[start:start + config["eval_batch_size"]]
            outputs = adaptive_batches(
                rows, config["eval_batch_size"],
                lambda batch: run.generate_batch(
                    model, tokenizer, [row["prompt"] for row in batch], direction, rank_one,
                    strength, schedule, "cuda", config["eval_max_new_tokens"], config["system"],
                    enable_thinking=config["enable_thinking"]))
            saved = [{"model": config["model"], "concept": config["concept"],
                      "schedule": schedule, "strength": strength, "key": row["key"],
                      "prompt": row["prompt"], "response": response}
                     for row, response in zip(rows, outputs)]
            append_jsonl(output_path, saved)
            existing.update((schedule, strength, row["key"]) for row in rows)
            print(f"eval shard={shard} {schedule} c={strength:+g}: "
                  f"{min(start + len(rows), len(todo))}/{len(todo)}", flush=True)


def ifeval_scores(outputs):
    import random
    from langdetect import DetectorFactory

    DetectorFactory.seed = 0
    sys.path.insert(0, str(ROOT / "cache/google_research"))
    from instruction_following_eval import evaluation_lib
    inputs = evaluation_lib.read_prompt_list(ROOT / "cache/ifeval/input_data.jsonl")
    responses = {row["prompt"]: row["response"] for row in outputs}
    random.seed(0)
    strict = [evaluation_lib.test_instruction_following_strict(row, responses) for row in inputs]
    random.seed(0)
    loose = [evaluation_lib.test_instruction_following_loose(row, responses) for row in inputs]

    def aggregate(rows):
        followed = [flag for row in rows for flag in row.follow_instruction_list]
        return (sum(row.follow_all_instructions for row in rows) / len(rows),
                sum(followed) / len(followed))

    strict_prompt, strict_instruction = aggregate(strict)
    loose_prompt, loose_instruction = aggregate(loose)
    flags = {row.prompt: {"ifeval_strict": row.follow_all_instructions,
                          "ifeval_strict_instructions": row.follow_instruction_list}
             for row in strict}
    return {"ifeval_prompt_strict": strict_prompt,
            "ifeval_instruction_strict": strict_instruction,
            "ifeval_prompt_loose": loose_prompt,
            "ifeval_instruction_loose": loose_instruction}, flags


def language_score(text):
    from langdetect import DetectorFactory, LangDetectException, detect_langs
    DetectorFactory.seed = 0
    try:
        probabilities = {item.lang: item.prob for item in detect_langs(text)}
    except LangDetectException:
        probabilities = {}
    return probabilities.get("ru", 0.0) - probabilities.get("fr", 0.0), probabilities


def score_outputs(config, num_shards):
    rows = [row for shard in range(num_shards)
            for row in read_jsonl(ROOT / f"outputs/shards/ifeval-{shard:02d}.jsonl")]
    baseline = [row for row in rows if row["schedule"] == "once" and row["strength"] == 0]
    rows += [{**row, "schedule": "clamp", "baseline_reused": True} for row in baseline]
    expected = 541 * 2 * len(config["strengths"])
    unique = {(row["schedule"], row["strength"], row["key"]) for row in rows}
    if len(rows) != expected or len(unique) != expected:
        raise RuntimeError(f"Expected {expected} unique outputs, got rows={len(rows)}, unique={len(unique)}")
    rows.sort(key=lambda row: (row["schedule"], row["strength"], row["key"]))

    metrics, scored = [], []
    for schedule in ("once", "clamp"):
        for strength in config["strengths"]:
            selected = [row for row in rows
                        if row["schedule"] == schedule and row["strength"] == strength]
            ifeval, flags = ifeval_scores(selected)
            language, labels, lengths = [], [], []
            for row in selected:
                value, probabilities = language_score(row["response"])
                language.append(value)
                labels.append(max(probabilities, key=probabilities.get) if probabilities else "unknown")
                lengths.append(len(row["response"]))
                scored.append({**row, **flags[row["prompt"]], "language_score": value,
                               "language": labels[-1], "language_probabilities": probabilities})
            rates = {name: labels.count(name) / len(labels) for name in ("ru", "fr", "en")}
            metrics.append({"schedule": schedule, "strength": strength, "n": len(selected),
                            "language_score": sum(language) / len(language),
                            "target_language_rate": (rates["ru"] if strength > 0 else
                                                     rates["fr"] if strength < 0 else ""),
                            "russian_rate": rates["ru"], "french_rate": rates["fr"],
                            "english_rate": rates["en"],
                            "other_language_rate": 1 - sum(rates.values()),
                            "no_language_rate": labels.count("unknown") / len(labels),
                            "mean_response_chars": sum(lengths) / len(lengths), **ifeval})

    write_jsonl(ROOT / "outputs/all_outputs.jsonl", rows)
    write_jsonl(ROOT / "outputs/scored_outputs.jsonl", scored)
    metrics_path = ROOT / "scores/metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0])
        writer.writeheader()
        writer.writerows(metrics)
    plot(metrics)
    print(f"Saved {metrics_path} and {ROOT / 'plots/steering-vs-ifeval.png'}", flush=True)


def plot(metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for axis, schedule, title in zip(
            axes, ("once", "clamp"), ("Full rank: add once", "Rank 1: clamp every token")):
        rows = [row for row in metrics if row["schedule"] == schedule]
        x = [row["strength"] for row in rows]
        language = [row["language_score"] for row in rows]
        quality = [row["ifeval_prompt_strict"] for row in rows]
        axis.plot(x, language, "o-", color="tab:blue", label="P(Russian) - P(French)")
        axis.axhline(0, color="0.75", linewidth=1)
        axis.set(xlabel="Steering strength", ylabel="Language steering score", title=title,
                 ylim=(-1.05, 1.05))
        other = axis.twinx()
        other.plot(x, quality, "s--", color="tab:orange", label="IFEval strict")
        other.set_ylabel("IFEval strict prompt accuracy")
        other.set_ylim(0, 1)
        lines = axis.lines[:1] + other.lines
        axis.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout()
    fig.savefig(ROOT / "plots/steering-vs-ifeval.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, schedule, title in zip(
            axes, ("once", "clamp"), ("Full rank: add once", "Rank 1: clamp every token")):
        rows = [row for row in metrics if row["schedule"] == schedule]
        x = [row["strength"] for row in rows]
        for field, label, color in (("russian_rate", "Russian", "tab:red"),
                                    ("french_rate", "French", "tab:blue"),
                                    ("english_rate", "English", "tab:green"),
                                    ("other_language_rate", "Other / unknown", "tab:gray")):
            axis.plot(x, [row[field] for row in rows], "o-", label=label, color=color)
        axis.set(xlabel="Steering strength", ylabel="Fraction of responses", title=title,
                 ylim=(-0.02, 1.02))
        axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(ROOT / "plots/language-rates.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, schedule, title in zip(
            axes, ("once", "clamp"), ("Full rank: add once", "Rank 1: clamp every token")):
        rows = [row for row in metrics if row["schedule"] == schedule]
        x = [row["strength"] for row in rows]
        for field, label in (("ifeval_prompt_strict", "Prompt strict"),
                             ("ifeval_prompt_loose", "Prompt loose"),
                             ("ifeval_instruction_strict", "Instruction strict")):
            axis.plot(x, [row[field] for row in rows], "o-", label=label)
        axis.set(xlabel="Steering strength", ylabel="IFEval accuracy", title=title,
                 ylim=(0, 1))
        axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(ROOT / "plots/ifeval-breakdown.png", dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "train", "merge", "evaluate", "score"))
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--skip-model-download", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.command == "prepare":
        prepare(config, not args.skip_model_download)
    elif args.command == "train":
        train_shard(config, args.shard_index, args.num_shards)
    elif args.command == "merge":
        merge_direction(config, args.num_shards)
    elif args.command == "evaluate":
        evaluate_shard(config, args.shard_index, args.num_shards)
    else:
        score_outputs(config, args.num_shards)


if __name__ == "__main__":
    main()
