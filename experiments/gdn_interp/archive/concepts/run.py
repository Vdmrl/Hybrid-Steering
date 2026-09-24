"""Generate contrast texts/directions, steer the benchmark, and locally judge it."""

import argparse
import json
import re
from contextlib import ExitStack
from pathlib import Path

import torch

from gdn_interp import gdn_layers, load_qwen, steer_states, steering_deltas


ROOT = Path(__file__).parent
DATA = json.loads((ROOT / "concepts.json").read_text())


def selected(args):
    return {args.concept: DATA[args.concept]} if args.concept else DATA


def chat(tokenizer, messages, generation=True):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation, enable_thinking=False)


def model_and_tokenizer(args):
    model, tokenizer = load_qwen(args.model, dtype=getattr(torch, args.dtype))
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens, cache=None):
    device = model.get_input_embeddings().weight.device
    inputs = tokenizer(chat(tokenizer, [{"role": "user", "content": prompt}]), return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(**inputs, past_key_values=cache, do_sample=False, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)


def pairs(args):
    model, tokenizer = model_and_tokenizer(args)
    out = args.run_dir / "contrast_texts.jsonl"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for concept, spec in selected(args).items():
            for pair_id, (positive, negative) in enumerate(spec["pairs"]):
                for side, prompt in (("positive", positive), ("negative", negative)):
                    row = {"concept": concept, "pair_id": pair_id, "side": side, "prompt": prompt,
                           "response": generate(model, tokenizer, prompt, args.max_new_tokens)}
                    f.write(json.dumps(row) + "\n")
                    print(f"{concept} {pair_id} {side}", flush=True)


def direction(args):
    rows = [json.loads(x) for x in (args.run_dir / "contrast_texts.jsonl").read_text().splitlines()]
    model, tokenizer = model_and_tokenizer(args)
    found, captured = gdn_layers(model), {}

    def capture(module, _args, kwargs, _output):
        captured[module.layer_idx] = kwargs["cache_params"].layers[module.layer_idx].recurrent_states[0].detach().float().cpu().squeeze(0)

    result = {}
    with ExitStack() as stack:
        for _, module in found:
            handle = module.register_forward_hook(capture, with_kwargs=True)
            stack.callback(handle.remove)
        for concept in selected(args):
            means = []
            for side in ("positive", "negative"):
                states = []
                for row in (x for x in rows if x["concept"] == concept and x["side"] == side):
                    captured.clear()
                    prompt = chat(tokenizer, [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["response"]}], generation=False)
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.get_input_embeddings().weight.device)
                    with torch.inference_mode(): model(**inputs, use_cache=True)
                    states.append(torch.stack([captured[m.layer_idx] for _, m in found]))
                means.append(torch.stack(states).mean(0))
            result[concept] = means[0] - means[1]
    torch.save({"model": args.model, "gdn_layers": [name for name, _ in found], "directions": result,
                "orientation": {k: f'{v["positive_label"]} - {v["negative_label"]}' for k, v in DATA.items()}}, args.run_dir / "directions.pt")


def benchmark(args):
    model, tokenizer = model_and_tokenizer(args)
    artifact = torch.load(args.run_dir / "directions.pt", weights_only=True)
    out = args.run_dir / "responses.jsonl"
    with out.open("w") as f:
        for concept, spec in selected(args).items():
            for example_id, prompt in enumerate(spec["benchmark"][: args.limit]):
                for scale in args.scales:
                    if scale == 0:
                        response = generate(model, tokenizer, prompt, args.max_new_tokens)
                    else:
                        deltas = steering_deltas(artifact["directions"][concept], range(len(artifact["gdn_layers"])), scale)
                        with steer_states(model, deltas, mode=args.mode, batch_size=1) as cache:
                            response = generate(model, tokenizer, prompt, args.max_new_tokens, cache)
                    f.write(json.dumps({"concept": concept, "example_id": example_id, "prompt": prompt, "scale": scale,
                                        "mode": args.mode if scale else None, "response": response}) + "\n")
                    print(f"{concept} {example_id + 1}/20 scale={scale}", flush=True)


def judge(args):
    rows = [json.loads(x) for x in (args.run_dir / "responses.jsonl").read_text().splitlines()]
    model, tokenizer = model_and_tokenizer(args)
    template = (ROOT / "judge_prompt.txt").read_text()
    out = args.run_dir / "judgments.jsonl"
    with out.open("w") as f:
        for row in rows:
            spec = DATA[row["concept"]]
            prompt = template.format(**spec, prompt=row["prompt"], response=row["response"])
            raw = generate(model, tokenizer, prompt, args.max_new_tokens)
            match = re.search(r'\{\s*"score"\s*:\s*(-?1|0)\s*,\s*"reason"\s*:\s*"([^"]*)"\s*\}', raw, re.S)
            score, reason = (int(match.group(1)), match.group(2)) if match else (0, f"unparseable judge output: {raw[:160]}")
            f.write(json.dumps({**row, "score": score, "reason": reason, "judge_raw": raw}) + "\n")
            print(f"judged {row['concept']} {row['example_id']} scale={row['scale']}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("pairs", "direction", "benchmark", "judge"))
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--concept", choices=tuple(DATA), help="run one independent concept")
    p.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--scales", nargs="+", type=float, default=(-10, 0, 10))
    p.add_argument("--mode", choices=("initial", "repeated"), default="repeated")
    p.add_argument("--limit", type=int, help="benchmark prompt limit, for calibration")
    args = p.parse_args()
    globals()[args.command](args)


if __name__ == "__main__": main()
