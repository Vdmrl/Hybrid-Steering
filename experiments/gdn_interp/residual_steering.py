"""Extract and screen one-shot residual-stream steering directions."""

import argparse
import json
import math
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import torch

from gdn_interp import load_qwen


def extract(args: argparse.Namespace) -> None:
    rows = list(map(json.loads, Path(args.dataset).read_text().splitlines()))[: args.limit]
    model, tokenizer = load_qwen(args.model, dtype=getattr(torch, args.dtype))
    model.eval()
    device = model.get_input_embeddings().weight.device
    layers = list(model.model.layers)
    captured = {}

    def capture(index):
        def hook(_module, _args, output):
            captured[index] = (output[0] if isinstance(output, tuple) else output)[:, -1].detach().float().cpu()
        return hook

    means = []
    with ExitStack() as stack:
        for index, layer in enumerate(layers):
            stack.callback(layer.register_forward_hook(capture(index)).remove)
        for field in ("concept_text", "antagonist_text"):
            total = None
            for row in rows:
                prompt = tokenizer.apply_chat_template([{"role": "user", "content": row[field]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                captured.clear()
                with torch.inference_mode():
                    model(**tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(device))
                states = torch.cat([captured[index] for index in range(len(layers))])
                total = states if total is None else total + states
            means.append(total / len(rows))
    direction = means[0] - means[1]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": args.model, "dataset": str(args.dataset), "labels": ("optimism", "pessimism"), "count": len(rows), "max_length": args.max_length, "direction": direction, "layers": list(range(len(layers))), "norms": direction.norm(dim=1)}, args.run_dir / "residual_direction.pt")
    print(f"wrote {args.run_dir / 'residual_direction.pt'}")


def screen(args: argparse.Namespace) -> None:
    artifact = torch.load(args.run_dir / "residual_direction.pt", weights_only=True)
    model, tokenizer = load_qwen(artifact["model"], dtype=getattr(torch, args.dtype))
    model.eval()
    device = model.get_input_embeddings().weight.device
    layers = list(model.model.layers)
    rows = [json.loads(line) for line in (args.contexts or args.run_dir / "contexts.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row["target_tokens"] == args.calibration_length][: args.limit]
    output = args.run_dir / args.output_name
    done = {(row["example_id"], row.get("layer", -1), row["scale"]) for row in map(json.loads, output.read_text().splitlines())} if output.exists() else set()
    layers_to_test = args.layers or artifact["layers"]
    candidates = [(layer, scale) for layer in layers_to_test for scale in args.scales]
    direction = artifact["direction"].to(device=device, dtype=model.get_input_embeddings().weight.dtype)
    direction = direction / direction.float().norm(dim=1, keepdim=True).to(direction.dtype)
    with output.open("a") as stream, torch.inference_mode(), ExitStack() as stack:
        active = {}

        def add(layer):
            def hook(_module, _args, result):
                hidden = result[0] if isinstance(result, tuple) else result
                if layer in active and hidden.shape[1] > 1:
                    hidden.add_(active[layer][:, None, :])
                return result
            return hook

        for index, layer in enumerate(layers):
            stack.callback(layer.register_forward_hook(add(index)).remove)
        for row in rows:
            ids = torch.tensor([row["prompt_input_ids"]], device=device)
            active = {}
            if (row["example_id"], -1, 0.0) not in done:
                response = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.max_new_tokens, do_sample=False)
                stream.write(json.dumps({**row, "layer": -1, "scale": 0.0, "control": "baseline", "response": tokenizer.decode(response[0, ids.shape[1]:], skip_special_tokens=True)}) + "\n")
            pending = [(layer, scale) for layer, scale in candidates if (row["example_id"], layer, scale) not in done]
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start : start + args.batch_size]
                batch_ids = ids.expand(len(batch), -1)
                active = {index: torch.zeros(len(batch), direction.shape[1], device=device, dtype=direction.dtype) for index in range(len(layers))}
                for index, (layer, scale) in enumerate(batch):
                    active[layer][index] = direction[layer] * scale
                response = model.generate(input_ids=batch_ids, attention_mask=torch.ones_like(batch_ids), max_new_tokens=args.max_new_tokens, do_sample=False)
                for index, (layer, scale) in enumerate(batch):
                    stream.write(json.dumps({**row, "layer": layer, "scale": scale, "control": "residual_prefill_all_calibration", "response": tokenizer.decode(response[index, ids.shape[1]:], skip_special_tokens=True)}) + "\n")
                stream.flush()
            print(f"example {row['example_id']} done", flush=True)


def summarize(args: argparse.Namespace) -> None:
    rows = list(map(json.loads, (args.run_dir / args.judgments_name).read_text().splitlines()))
    baseline = {row["example_id"]: row["score_distribution"]["expected_score"] for row in rows if row["layer"] == -1}
    groups = {}
    for row in rows:
        if row["layer"] >= 0 and row["example_id"] in baseline:
            groups.setdefault((row["layer"], row["scale"]), []).append(row["score_distribution"]["expected_score"] - baseline[row["example_id"]])
    summary = []
    for (layer, scale), values in groups.items():
        mean = sum(values) / len(values)
        se = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) * (len(values) - 1))) if len(values) > 1 else 0
        summary.append({"layer": layer, "strength_l2": scale, "n": len(values), "mean_score_delta": mean, "ci95_low": mean - 1.96 * se, "ci95_high": mean + 1.96 * se})
    summary.sort(key=lambda row: (row["n"], row["mean_score_delta"]), reverse=True)
    output = {"baseline_mean_score": sum(baseline.values()) / len(baseline), "best": summary[0], "conditions": summary}
    (args.run_dir / args.summary_name).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


def generate_full(args: argparse.Namespace) -> None:
    """Inject the direction at every prompt position during prefill only, never during generation."""
    artifact = torch.load(args.run_dir / "residual_direction.pt", weights_only=True)
    model, tokenizer = load_qwen(artifact["model"], dtype=getattr(torch, args.dtype))
    model.eval()
    device = model.get_input_embeddings().weight.device
    direction = artifact["direction"][args.layer].to(device=device, dtype=model.get_input_embeddings().weight.dtype)
    direction = direction / direction.float().norm().to(direction.dtype)
    rows = list(map(json.loads, args.contexts.read_text().splitlines()))
    output = args.run_dir / args.output_name
    done = (
        {(row["example_id"], row["target_tokens"], row["scale"]) for row in map(json.loads, output.read_text().splitlines())}
        if output.exists()
        else set()
    )
    groups = defaultdict(list)
    for row in rows:
        groups[row["input_tokens"]].append(row)
    with output.open("a") as stream, torch.inference_mode(), ExitStack() as stack:
        active = None

        def add(_module, _args, result):
            hidden = result[0] if isinstance(result, tuple) else result
            if active is not None and hidden.shape[1] > 1:
                hidden.add_(active[:, None, :])
            return result

        stack.callback(model.model.layers[args.layer].register_forward_hook(add).remove)
        for length, group in sorted(groups.items()):
            batch_size = min(args.batch_size, max(1, args.batch_tokens // length))
            for scale in (0.0, args.strength):
                pending = [row for row in group if (row["example_id"], row["target_tokens"], scale) not in done]
                for start in range(0, len(pending), batch_size):
                    batch = pending[start : start + batch_size]
                    ids = torch.tensor([row["prompt_input_ids"] for row in batch], device=device)
                    active = direction.expand(len(batch), -1) * scale if scale else None
                    generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.max_new_tokens, do_sample=False)
                    active = None
                    for row, result in zip(batch, generated, strict=True):
                        stream.write(
                            json.dumps(
                                {
                                    **{key: value for key, value in row.items() if key not in {"filler", "filler_input_ids", "prompt_input_ids"}},
                                    "layer": args.layer,
                                    "scale": scale,
                                    "control": "residual_prefill_all",
                                    "normalization": "unit_l2",
                                    "response": tokenizer.decode(result[ids.shape[1]:], skip_special_tokens=True),
                                }
                            )
                            + "\n"
                        )
                    stream.flush()
                    print(f"length={length} scale={scale} {start + len(batch)}/{len(pending)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run-dir", type=Path, required=True)
    common.add_argument(
        "--dtype",
        choices=("float16", "bfloat16"),
        default="float16",
        help="pinned to match the fp16 numerics the GDN arm's artifacts were generated under",
    )
    extract_parser = commands.add_parser("extract", parents=[common])
    extract_parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    extract_parser.add_argument("--dataset", type=Path, default=Path("../shared/optimism_pessimism_stories.jsonl"))
    extract_parser.add_argument("--limit", type=int, default=64)
    extract_parser.add_argument("--max-length", type=int, default=512)
    extract_parser.set_defaults(func=extract)
    screen_parser = commands.add_parser("screen", parents=[common])
    screen_parser.add_argument("--contexts", type=Path)
    screen_parser.add_argument("--output-name", default="residual_calibration_responses.jsonl")
    screen_parser.add_argument("--calibration-length", type=int, default=128)
    screen_parser.add_argument("--limit", type=int, default=64)
    screen_parser.add_argument("--layers", type=int, nargs="+")
    screen_parser.add_argument("--scales", type=float, nargs="+", default=(0.5, 1.0, 2.0, 4.0, 8.0))
    screen_parser.add_argument("--batch-size", type=int, default=128)
    screen_parser.add_argument("--max-new-tokens", type=int, default=96)
    screen_parser.set_defaults(func=screen)
    summary_parser = commands.add_parser("summarize", parents=[common])
    summary_parser.add_argument("--judgments-name", default="residual_calibration_judgments.jsonl")
    summary_parser.add_argument("--summary-name", default="residual_calibration_summary.json")
    summary_parser.set_defaults(func=summarize)
    full_parser = commands.add_parser("generate-full", parents=[common])
    full_parser.add_argument("--contexts", type=Path, required=True)
    full_parser.add_argument("--output-name", default="residual_prefill_all_responses.jsonl")
    full_parser.add_argument("--layer", type=int, default=0)
    full_parser.add_argument("--strength", type=float, default=2.0)
    full_parser.add_argument("--batch-size", type=int, default=128)
    full_parser.add_argument("--batch-tokens", type=int, default=262144)
    full_parser.add_argument("--max-new-tokens", type=int, default=256)
    full_parser.set_defaults(func=generate_full)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
