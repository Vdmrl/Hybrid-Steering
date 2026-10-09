#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from hybrid_steering.paths import ROOT as PROJECT_ROOT

DATA_ROOT = Path(os.environ.get("GDN_DATA_ROOT", str(PROJECT_ROOT / "data")))
os.environ.setdefault("HF_HOME", str(DATA_ROOT / "cache/huggingface"))
os.environ.setdefault("TORCH_HOME", str(DATA_ROOT / "cache/torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(DATA_ROOT / "cache"))
os.environ.setdefault("TRITON_CACHE_DIR", str(DATA_ROOT / "cache/triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(DATA_ROOT / "cache/torchinductor"))
os.environ.setdefault("CUDA_CACHE_PATH", str(DATA_ROOT / "cache/nv"))
os.environ.setdefault("TMPDIR", str(DATA_ROOT / "tmp"))

import torch
# Shared tensor operations for the retained decoder.
from hybrid_steering.state import add, analyze, clamp, clamp_centered, clamp_interval


SYSTEM = "Отвечай по-русски, кратко и по существу."


def chat(tokenizer, user, assistant=None, system=SYSTEM, enable_thinking=True):
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": user})
    if assistant is not None:
        messages.append({"role": "assistant", "content": assistant})
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=assistant is None,
                                         enable_thinking=enable_thinking)


@torch.inference_mode()
def recurrent_states_batch(model, tokenizer, texts, device):
    batch = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt")
    ids, mask = batch.input_ids.to(device), batch.attention_mask.to(device)
    cache = model(input_ids=ids, attention_mask=mask, use_cache=True,
                  logits_to_keep=1).past_key_values
    states = {}
    for i, layer in enumerate(cache.layers):
        recurrent = getattr(layer, "recurrent_states", None)
        if recurrent and recurrent[0] is not None:
            states[i] = recurrent[0].float().cpu()
    if not states:
        raise RuntimeError("No GDN recurrent states found in the model cache")
    return states


def recurrent_states(model, tokenizer, text, device):
    return {i: state[0] for i, state in
            recurrent_states_batch(model, tokenizer, [text], device).items()}


def fingerprint(model_id, concept):
    raw = json.dumps({"model": model_id, "system": concept.get("system", SYSTEM),
                      "pairs": concept["pairs"]}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def pair_fingerprint(model_id, concept):
    raw = json.dumps({
        "model": model_id,
        "train_prompts": concept["train_prompts"],
        "positive_system": concept["positive_system"],
        "negative_system": concept["negative_system"],
        "pair_max_new_tokens": concept.get("pair_max_new_tokens", 192),
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def extract_direction(model, tokenizer, model_id, name, concept, device, rebuild=False):
    path = DATA_ROOT / "artifacts" / f"{name}.pt"
    fp = fingerprint(model_id, concept)
    if path.exists() and not rebuild:
        saved = torch.load(path, map_location="cpu")
        if saved["fingerprint"] == fp:
            return saved["direction"]

    direction = None
    system = concept.get("system", SYSTEM)
    for pair in concept["pairs"]:
        positive = recurrent_states(model, tokenizer, chat(tokenizer, pair["prompt"], pair["positive"], system), device)
        negative = recurrent_states(model, tokenizer, chat(tokenizer, pair["prompt"], pair["negative"], system), device)
        if direction is None:
            direction = {i: positive[i] - negative[i] for i in positive}
        else:
            for i in direction:
                direction[i].add_(positive[i] - negative[i])
    for state in direction.values():
        state.div_(len(concept["pairs"]))
    path.parent.mkdir(exist_ok=True)
    torch.save({"fingerprint": fp, "direction": direction}, path)
    return direction




def position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Positions for left-padded prompts and their growing decode masks."""
    return (attention_mask.long().cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)


def decode_generated(tokenizer, generated, enable_thinking):
    if enable_thinking:
        close = tokenizer.convert_tokens_to_ids("</think>")
        generated = [ids[ids.index(close) + 1:] if close in ids else [] for ids in generated]
    return [text.strip() for text in tokenizer.batch_decode(generated, skip_special_tokens=True)]


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts, direction, rank_one, strength, schedule,
                   device, max_new_tokens, system=SYSTEM, anchors=None,
                   enable_thinking=True):
    if not prompts:
        return []
    interval = clamp_interval(schedule)
    texts = [chat(tokenizer, prompt, system=system, enable_thinking=enable_thinking)
             for prompt in prompts]
    batch = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt")
    ids, mask = batch.input_ids.to(device), batch.attention_mask.to(device)
    # Split prefill is intentional: steer before the last prompt token predicts
    # output token 0. Keep this timing identical for all schedules.
    first = model(input_ids=ids[:, :-1], attention_mask=mask[:, :-1], use_cache=True,
                  position_ids=position_ids(mask[:, :-1]), logits_to_keep=1)
    cache = first.past_key_values
    if schedule == "once":
        add(cache, direction, strength)
    elif schedule == "centered":
        clamp_centered(cache, rank_one, anchors, strength)
    elif interval or schedule == "clamp_once":
        clamp(cache, rank_one, strength)
    out = model(input_ids=ids[:, -1:], attention_mask=mask, past_key_values=cache,
                position_ids=position_ids(mask)[:, -1:], use_cache=True, logits_to_keep=1)
    generated = [[] for _ in prompts]
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    finished = [False] * len(prompts)
    fill_token = next(iter(eos), tokenizer.eos_token_id or tokenizer.pad_token_id)

    for step in range(max_new_tokens):
        token = out.logits[:, -1].argmax(-1)
        for row, token_id in enumerate(token.tolist()):
            if finished[row]:
                token[row] = fill_token
            elif token_id in eos:
                finished[row] = True
            else:
                generated[row].append(token_id)
        if all(finished):
            break
        if step + 1 == max_new_tokens:
            break
        if interval and (step + 1) % interval == 0:
            clamp(cache, rank_one, strength)
        elif schedule == "centered":
            clamp_centered(cache, rank_one, anchors, strength)
        mask = torch.cat((mask, torch.ones((len(prompts), 1), dtype=mask.dtype,
                                           device=device)), dim=1)
        out = model(input_ids=token[:, None], attention_mask=mask, past_key_values=cache,
                    position_ids=position_ids(mask)[:, -1:], use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
    return decode_generated(tokenizer, generated, enable_thinking)


def generate(model, tokenizer, prompt, direction, rank_one, strength, schedule, device,
             max_new_tokens, system=SYSTEM, anchors=None, enable_thinking=True):
    return generate_batch(model, tokenizer, [prompt], direction, rank_one, strength,
                          schedule, device, max_new_tokens, system, anchors,
                          enable_thinking)[0]


def generate_pairs(model, tokenizer, model_id, name, concept, device, rebuild=False):
    if "train_prompts" not in concept:
        return concept

    path = DATA_ROOT / "generated" / f"{name}.json"
    fp = pair_fingerprint(model_id, concept)
    if path.exists() and not rebuild:
        saved = json.loads(path.read_text())
        if saved["fingerprint"] == fp:
            return {**concept, "pairs": saved["pairs"]}

    pairs = []
    limit = concept.get("pair_max_new_tokens", 192)
    for prompt in concept["train_prompts"]:
        positive = generate(model, tokenizer, prompt, None, None, 0, "once", device, limit,
                            concept["positive_system"])
        negative = generate(model, tokenizer, prompt, None, None, 0, "once", device, limit,
                            concept["negative_system"])
        pairs.append({"prompt": prompt, "positive": positive, "negative": negative})
        print(f"  pair {len(pairs)}/{len(concept['train_prompts'])}: "
              f"positive={positive[:80]!r} negative={negative[:80]!r}", flush=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fingerprint": fp, "model": model_id, "pairs": pairs},
                               ensure_ascii=False, indent=2))
    return {**concept, "pairs": pairs}


def score(metric, text):
    if metric["type"] == "regex":
        positive = len(re.findall(metric["positive"], text))
        negative = len(re.findall(metric.get("negative", r"(?!x)x"), text))
        value = max(-1.0, min(1.0, (positive - negative) / metric.get("target", 1)))
    else:
        lowered = text.lower()
        positive = sum(lowered.count(word) for word in metric["positive"])
        negative = sum(lowered.count(word) for word in metric["negative"])
        value = (positive - negative) / max(1, positive + negative)
    return value, {"positive_hits": positive, "negative_hits": negative}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/legacy-config/configs/shared/concepts.json")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--concepts", default="all", help="comma-separated names")
    parser.add_argument("--strengths", default="-2,0,2")
    parser.add_argument("--schedule", choices=["once", "clamp", "clamp_once"], default="clamp")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", default="results/quick.jsonl")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--regenerate-pairs", action="store_true")
    return parser.parse_args()


def main():
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    args = parse_args()
    for path in (DATA_ROOT / "artifacts", DATA_ROOT / "cache", DATA_ROOT / "generated",
                 DATA_ROOT / "results", DATA_ROOT / "tmp"):
        path.mkdir(parents=True, exist_ok=True)
    concepts = json.loads(Path(args.config).read_text())
    names = list(concepts) if args.concepts == "all" else args.concepts.split(",")
    strengths = [float(x) for x in args.strengths.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype).eval().to(device)
    output = Path(args.output)
    if not output.is_absolute():
        output = DATA_ROOT / output
    output.parent.mkdir(exist_ok=True)
    rows, ranks = [], {}

    for name in names:
        concept = concepts[name]
        print(f"\n[{name}] generating/loading model-authored pairs", flush=True)
        concept = generate_pairs(model, tokenizer, args.model, name, concept, device,
                                 args.regenerate_pairs)
        print(f"\n[{name}] extracting direction", flush=True)
        direction = extract_direction(model, tokenizer, args.model, name, concept, device, args.rebuild)
        ranks[name], rank_one = analyze(direction, device)
        print(f"[{name}] {ranks[name]}", flush=True)
        for strength in strengths:
            for prompt in concept["eval_prompts"]:
                text = generate(model, tokenizer, prompt, direction, rank_one, strength, args.schedule,
                                device, args.max_new_tokens, concept.get("system", SYSTEM))
                value, details = score(concept["metric"], text)
                row = {"concept": name, "strength": strength, "schedule": args.schedule,
                       "score": value, **details, "prompt": prompt, "output": text}
                rows.append(row)
                print(f"  c={strength:g} score={value:+.2f} | {text[:180]!r}", flush=True)

    with output.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    rank_path = output.with_suffix(".ranks.json")
    rank_path.write_text(json.dumps(ranks, ensure_ascii=False, indent=2))

    summary = []
    for name in names:
        for strength in strengths:
            values = [r["score"] for r in rows if r["concept"] == name and r["strength"] == strength]
            summary.append({"concept": name, "strength": strength, "mean_score": sum(values) / len(values)})
    summary_path = output.with_suffix(".csv")
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)

    print("\nconcept\tlow\thigh\teffect")
    low, high = min(strengths), max(strengths)
    for name in names:
        means = {r["strength"]: r["mean_score"] for r in summary if r["concept"] == name}
        print(f"{name}\t{means[low]:+.2f}\t{means[high]:+.2f}\t{means[high] - means[low]:+.2f}")
    print(f"\nraw={output} summary={summary_path} ranks={rank_path}")


if __name__ == "__main__":
    main()
