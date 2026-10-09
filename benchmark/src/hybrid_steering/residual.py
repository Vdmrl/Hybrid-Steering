"""Small residual-output sweep; reuse existing chat, generation and JSONL helpers."""
import argparse
import csv
import hashlib
import json
import os
import socket
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from hybrid_steering.paths import ROOT as PROJECT_ROOT

import torch
from hybrid_steering import language_benchmark as bench
from hybrid_steering import language_search as search
from hybrid_steering import runner as run


def residual_weight(token: int, schedule: str) -> float:
    if token < 0:
        return 0.
    if schedule == "every":
        return 1.
    if schedule in ("first8", "first16"):
        return float(token < int(schedule[5:]))
    if schedule == "every8":
        return float(token % 8 == 0)
    if schedule in ("decay16", "decay32"):
        return 2. ** (-token / int(schedule[5:]))
    raise ValueError(f"Unknown residual schedule: {schedule}")


@contextmanager
def residual_hook(block, delta: torch.Tensor, schedule: str = "every",
                  target: torch.Tensor | None = None, strength: float = 1.,
                  clamp_target: torch.Tensor | None = None):
    """Skip split-prefill; add only at output-token prediction positions 0,1,..."""
    if target is not None and clamp_target is not None:
        raise ValueError("Choose either legacy target interpolation or hard projection clamp")
    if clamp_target is not None and not torch.allclose(
            delta.float().square().sum(), delta.new_tensor(1.).float(), atol=1e-4):
        raise ValueError("Residual clamp direction must be a unit vector")
    calls = 0
    placed_delta = None
    def hook(module, args, output):
        nonlocal calls, placed_delta
        calls += 1
        original = output
        output = output[0] if isinstance(output, tuple) else output
        if clamp_target is not None:
            # Split prefill is call 1; act at the same output-token positions as add.
            if calls >= 2:
                if placed_delta is None:
                    placed_delta = delta.to(device=output.device, dtype=torch.float32)
                current = output[:, -1, :].float() @ placed_delta
                correction = clamp_target.to(current) - current
                output[:, -1, :].add_((correction[:, None] * placed_delta).to(output))
            return original
        if target is not None:
            # Stolfo et al. Eq. 2: adjust every position, including prompt prefill.
            if strength:
                u = delta.to(device=output.device, dtype=torch.float32)
                current = output.float() @ u
                correction = strength * (target.to(current) - current)
                output.add_((correction[..., None] * u).to(output))
            return original
        weight = residual_weight(calls - 2, schedule)
        if weight:
            if placed_delta is None:
                placed_delta = delta.to(output)
            output[:, -1, :].add_(placed_delta, alpha=weight)
        return original
    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def extract(config: dict, root: Path) -> None:
    path = root / "directions.pt"
    sources = sorted((root / "inputs").glob("*.jsonl"))
    fingerprint = {p.name: digest(p) for p in sources}
    if path.exists():
        old = torch.load(path, weights_only=False, map_location="cpu")
        if old["config"] != config or old["inputs"] != fingerprint:
            raise ValueError("Existing direction belongs to different inputs/config")
        return
    pairs = bench.read_jsonl(root / "inputs/pairs.jsonl")
    english = {r["id"]: r for p in (root / "inputs").glob("english-*.jsonl")
               for r in bench.read_jsonl(p)}
    rows = [r for r in pairs if r["id"] in english and r["russian_response"].strip()
            and english[r["id"]]["english_response"].strip()][:config["pair_count"]]
    assert len(rows) == config["pair_count"]
    assert all(r["model"] == config["model"] for r in rows)
    assert all(r["prompt"] == english[r["id"]]["prompt"] for r in rows)
    model, tokenizer = bench.load_model(config["model"])
    blocks = model.model.language_model.layers
    collected = {lang: {layer: [] for layer in config["layers"]} for lang in ("ru", "en")}
    for lang, field in (("ru", "russian_response"), ("en", "english_response")):
        for start in range(0, len(rows), 4):
            batch_rows = rows[start:start + 4]
            sequences = []
            for row in batch_rows:
                instruction = config.get("representation") == "instruction"
                prompt_text = row["russian_prompt"] if instruction and lang == "ru" else row["prompt"]
                prompt = run.chat(tokenizer, prompt_text, system=None, enable_thinking=False)
                tokens = tokenizer.encode(prompt, add_special_tokens=False)
                if not instruction:
                    answer = row[field] if lang == "ru" else english[row["id"]][field]
                    tokens += tokenizer.encode(answer, add_special_tokens=False)[:config["prefix_tokens"]]
                sequences.append(tokens)
            batch = tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt").to("cuda")
            handles = []
            for layer in config["layers"]:
                def capture(module, args, output, layer=layer):
                    collected[lang][layer].append(output[:, -1].float().cpu())
                handles.append(blocks[layer].register_forward_hook(capture))
            try:
                mask = batch.attention_mask
                positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 0)
                model(**batch, position_ids=positions, use_cache=False, logits_to_keep=1)
            finally:
                for handle in handles:
                    handle.remove()
            print(f"extract {lang}: {start + len(batch_rows)}/{len(rows)}", flush=True)
    directions, norms, targets = {}, {}, {}
    for layer in config["layers"]:
        ru = torch.cat(collected["ru"][layer])
        en = torch.cat(collected["en"][layer])
        delta = ru.mean(0) - en.mean(0)
        if not torch.isfinite(delta).all() or delta.norm() == 0:
            raise ValueError("Invalid residual direction")
        norms[layer] = en.norm(dim=-1).mean().item()
        unit = delta / delta.norm()
        if config.get("representation") == "instruction":
            directions[layer] = unit
            targets[layer] = (ru @ unit).mean()
        else:
            directions[layer] = unit * norms[layer]
    torch.save({"directions": directions, "norms": norms, "config": config,
                "inputs": fingerprint, "targets": targets, "pair_ids": [r["id"] for r in rows],
                "layer_types": {i: blocks[i].block_type for i in config["layers"]}}, path)


def benchmark_cases(benchmark: dict) -> list[tuple]:
    cases = [tuple(case) if len(case) == 3 else (*case, "every")
             for case in benchmark["cases"]]
    assert len(cases) == len(set(cases)), "Duplicate benchmark conditions"
    for layer, strength, schedule in cases:
        residual_weight(0, schedule)
    return cases


def evaluate(config: dict, root: Path, shard: int, shards: int,
             strengths: list[float] | None = None, benchmark: dict | None = None,
             sweep: dict | None = None) -> None:
    artifact = torch.load(root / "directions.pt", map_location="cpu", weights_only=False)
    assert artifact["config"] == config
    if benchmark:
        prompts = [{**r, "id": r["key"]} for r in bench.read_jsonl(root / benchmark["input"])]
        assert len(prompts) == benchmark.get("expected_n", 541)
    elif config.get('eval_prompt_config'):
        texts = json.loads(Path(config['eval_prompt_config']).read_text())['prompts']
        prompts = [{'id': i, 'prompt': text} for i, text in enumerate(texts)]
    else:
        prompts = search.prompts(json.loads(Path(config["prompt_config"]).read_text()))
        assert len(prompts) == 24
    if sweep:
        prompts += sweep.get("context_prompts", [])
    training = {r["prompt"] for r in bench.read_jsonl(root / "inputs/pairs.jsonl")}
    assert not training.intersection(p["prompt"] for p in prompts)
    path = root / f"outputs-{shard}.jsonl"
    done = {(r["layer"], r["strength"], r.get("schedule", "every"), r["id"])
            for r in bench.read_jsonl(path)}
    strengths = config["strengths"] if strengths is None else strengths
    jobs = [(-1, 0)] + [(layer, c) for layer in config["layers"] for c in strengths]
    jobs = [(layer, c, "every") for layer, c in jobs]
    if benchmark:
        jobs = benchmark_cases(benchmark)
    if sweep:
        jobs = [(-1, 0, "every")] + [(layer, c, schedule) for layer in sweep["layers"]
                for c in sweep["strengths"] for schedule in sweep["schedules"]]
    generation = sweep or benchmark or config
    model, tokenizer = bench.load_model(config["model"])
    blocks = model.model.language_model.layers
    manifest = {"config": {**config, "strengths": strengths}, "benchmark": benchmark, "sweep": sweep,
                "dataset_sha256": digest(root / benchmark["input"]) if benchmark else None,
                "extraction_config": artifact["config"],
                "direction_sha256": digest(root / "directions.pt"),
                "code": {name: digest(PROJECT_ROOT / name) for name in
                         ("src/hybrid_steering/residual.py", "src/hybrid_steering/runner.py",
                          "src/hybrid_steering/language_benchmark.py", "src/hybrid_steering/language_search.py")},
                "shard": shard, "shards": shards, "layer_types": artifact["layer_types"],
                "model_revision": getattr(model.config, "_commit_hash", None),
                "tokenizer_revision": tokenizer.init_kwargs.get('_commit_hash'),
                "host": socket.gethostname(), "physical_gpu": os.environ.get('CUDA_VISIBLE_DEVICES'),
                "data_root": str(root), "gpu": torch.cuda.get_device_name(),
                "generation": generation, "eval_prompts": prompts,
                "normalization": "unit RU-EN difference times mean EN activation L2 norm",
                "historical_answer_generation_revision": "unknown; smoke-test only"}
    (root / f"manifest-{shard}.json").write_text(json.dumps(manifest, indent=2))
    for index, (layer, strength, schedule) in enumerate(jobs):
        if index % shards != shard:
            continue
        todo = [p for p in prompts if (layer, strength, schedule, p["id"]) not in done]
        for start in range(0, len(todo), generation["batch_size"]):
            rows = todo[start:start + generation["batch_size"]]
            started = time.monotonic()
            def generate(batch):
                def call():
                    return run.generate_batch(model, tokenizer, [r["prompt"] for r in batch],
                                              None, None, 0, "once", "cuda", generation["max_new_tokens"],
                                              system=None, enable_thinking=False)
                if layer == -1:
                    return call()
                if config.get("representation") == "instruction":
                    with residual_hook(blocks[layer], artifact["directions"][layer],
                                       target=artifact["targets"][layer], strength=strength):
                        return call()
                with residual_hook(blocks[layer], strength * artifact["directions"][layer], schedule):
                    return call()
            answers = bench.adaptive_batches(rows, generation["batch_size"], generate)
            saved = [{**r, "layer": layer, "strength": strength, "schedule": schedule, "response": answer,
                      "language": search.language(answer), "thinking": False,
                      "model": config["model"], "method": ("instruction_projection_all_positions"
                      if config.get("representation") == "instruction" else "residual_output_" + schedule)}
                     for r, answer in zip(rows, answers)]
            bench.append_jsonl(path, saved)
            print(f"layer={layer} c={strength} schedule={schedule} n={len(saved)} seconds={time.monotonic()-started:.1f} "
                  f"ru={sum(r['language']=='ru' for r in saved)}/{len(saved)}", flush=True)


def score_ifeval(root: Path, benchmark: dict) -> None:
    """Score saved residual answers with the shared official IFEval evaluator."""
    rows = [r for p in root.glob("outputs-*.jsonl") for r in bench.read_jsonl(p)]
    prompts = {r["key"]: r["prompt"] for r in bench.read_jsonl(root / benchmark["input"])}
    metrics = []
    for layer, strength, schedule in benchmark_cases(benchmark):
        selected = [r for r in rows if (r["layer"], r["strength"], r.get("schedule", "every"))
                    == (layer, strength, schedule)]
        assert len(selected) == len(prompts) == benchmark.get("expected_n", 541)
        assert {r["key"] for r in selected} == set(prompts)
        assert all(r["prompt"] == prompts[r["key"]] for r in selected)
        quality, flags = bench.ifeval_scores(selected)
        bench.write_jsonl(root / f"scored-layer-{layer}-c{strength}-{schedule}.jsonl",
                          [{**r, **flags[r["prompt"]]} for r in selected])
        metrics.append({"layer": layer, "strength": strength, "schedule": schedule, "n": len(selected),
                        "language_rate": sum(r["language"] == "ru" for r in selected)/len(selected),
                        **quality})
    with (root / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=metrics[0])
        writer.writeheader()
        writer.writerows(metrics)
    print(json.dumps(metrics), flush=True)


def summarize_search(root: Path, config: dict, sweep: dict) -> None:
    """Descriptive screening only: exact fact retention is not semantic judging."""
    prompts = search.prompts(json.loads(Path(config["prompt_config"]).read_text())) + sweep["context_prompts"]
    expected = {p["id"]: p["prompt"] for p in prompts}
    groups = defaultdict(list)
    for path in root.glob("outputs-*.jsonl"):
        for row in bench.read_jsonl(path):
            groups[row["layer"], row["strength"], row["schedule"]].append(row)
    expected_jobs = {(-1, 0, "every")} | {(l, c, s) for l in sweep["layers"]
                    for c in sweep["strengths"] for s in sweep["schedules"]}
    assert set(groups) == expected_jobs
    metrics = []
    for (layer, strength, schedule), rows in sorted(groups.items()):
        assert len(rows) == len(expected)
        assert {r["id"]: r["prompt"] for r in rows} == expected
        context = [r for r in rows if "required" in r]
        neutral = [r for r in rows if "required" not in r]
        def retains(row):
            return all(value in row["response"] for value in row["required"])
        def repeated(row):
            words = row["response"].lower().split()
            grams = Counter(tuple(words[i:i+4]) for i in range(len(words)-3))
            return max(grams.values(), default=0) >= 4
        metrics.append({"layer": layer, "strength": strength, "schedule": schedule,
                        "n": len(rows), "neutral_n": len(neutral), "context_n": len(context),
                        "language_rate": sum(r["language"] == "ru" for r in rows)/len(rows),
                        "neutral_language_rate": sum(r["language"] == "ru" for r in neutral)/len(neutral),
                        "context_fact_rate": sum(retains(r) for r in context)/len(context),
                        "context_ru_and_facts": sum(retains(r) and r["language"] == "ru" for r in context)/len(context),
                        "repetition_rate": sum(repeated(r) for r in rows)/len(rows),
                        "empty_rate": sum(not r["response"].strip() for r in rows)/len(rows)})
    with (root / "search-metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=metrics[0])
        writer.writeheader()
        writer.writerows(metrics)
    print(json.dumps(metrics), flush=True)


def load_search(path: Path) -> dict:
    config = json.loads(path.read_text())
    if "context_prompt_config" in config:
        config["context_prompts"] = json.loads(Path(config["context_prompt_config"]).read_text())["context_prompts"]
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("extract", "evaluate", "score", "summarize"))
    parser.add_argument("--ifeval-config", type=Path)
    parser.add_argument("--search-config", type=Path)
    parser.add_argument("--config", default="data/legacy-config/configs/4b/residual_sweep.json")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--strengths", type=float, nargs="+",
                        help="Override evaluation strengths without rebuilding directions")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = bench.ROOT
    root.mkdir(parents=True, exist_ok=True)
    benchmark = json.loads(args.ifeval_config.read_text()) if args.ifeval_config else None
    if args.stage == "extract":
        extract(config, root)
    elif args.stage == "score":
        score_ifeval(root, benchmark)
    elif args.stage == "summarize":
        summarize_search(root, config, load_search(args.search_config))
    else:
        sweep = load_search(args.search_config) if args.search_config else None
        if sweep and benchmark:
            raise ValueError("Search and full IFEval are separate protocols")
        evaluate(config, root, args.shard, args.shards, args.strengths, benchmark, sweep)
