"""Run small, reproducible controls for an S_0 steering run."""

import argparse
import json
from pathlib import Path

import torch
from transformers import DynamicCache

from gdn_interp import gdn_layers, initial_state_cache, load_qwen, steer_states, steering_deltas


def generate(model, input_ids, cache, tokens):
    kwargs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "max_new_tokens": tokens, "do_sample": False}
    if cache is not None:
        kwargs["past_key_values"] = cache
    return model.generate(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--scale", type=float, default=14.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16"),
        default="float16",
        help="pinned to match the fp16 numerics the run's artifacts were generated under",
    )
    parser.add_argument(
        "--output-name",
        default="steering_validation.jsonl",
        help="responses file name; vary this across runs so a rerun does not overwrite a prior one",
    )
    args = parser.parse_args()
    artifact = torch.load(args.run_dir / "direction.pt", weights_only=True)
    rows = [json.loads(line) for line in (args.run_dir / "contexts.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row["target_tokens"] == args.length][: args.limit]
    if not rows:
        raise ValueError("no matching contexts")
    model, tokenizer = load_qwen(artifact["model"], dtype=getattr(torch, args.dtype))
    model.eval()
    device = model.get_input_embeddings().weight.device
    selected = range(len(artifact["gdn_layers"]))
    direction = artifact["direction"]
    controls = {
        "baseline": None,
        "zero_cache": steering_deltas(direction, selected, 0.0),
        "initial": steering_deltas(direction, selected, args.scale),
        "sign_flip": steering_deltas(-direction, selected, args.scale),
        "shuffled": steering_deltas(direction.flatten()[torch.randperm(direction.numel(), generator=torch.Generator().manual_seed(0))].reshape_as(direction), selected, args.scale),
    }
    output = args.run_dir / args.output_name
    diagnostics = []
    with output.open("w") as stream, torch.inference_mode():
        for row in rows:
            ids = torch.tensor([row["prompt_input_ids"]], device=device)
            generated = {}
            for label, deltas in controls.items():
                cache = initial_state_cache(model, deltas) if deltas is not None else None
                result = generate(model, ids, cache, args.max_new_tokens)
                text = tokenizer.decode(result[0, ids.shape[1] :], skip_special_tokens=True)
                generated[label] = result[0, ids.shape[1] :].tolist()
                stream.write(json.dumps({**row, "control": label, "scale": args.scale if label not in {"baseline", "zero_cache"} else 0.0, "response": text}) + "\n")
            base_cache = DynamicCache(config=model.config)
            model.model(input_ids=ids, attention_mask=torch.ones_like(ids), past_key_values=base_cache, use_cache=True)
            with steer_states(model, controls["initial"]) as cache:
                model.model(input_ids=ids, attention_mask=torch.ones_like(ids), past_key_values=cache, use_cache=True)
                state_norm = sum((cache.layers[module.layer_idx].recurrent_states[0] - base_cache.layers[module.layer_idx].recurrent_states[0]).float().square().sum().item() for _, module in gdn_layers(model)) ** 0.5
            diagnostics.append({"example_id": row["example_id"], "zero_cache_matches_baseline": generated["zero_cache"] == generated["baseline"], "initial_differs_from_baseline": generated["initial"] != generated["baseline"], "state_delta_l2_after_prefill": state_norm})
    zero_cache_match_rate = sum(row["zero_cache_matches_baseline"] for row in diagnostics) / len(diagnostics)
    report = {
        "length": args.length,
        "examples": len(rows),
        "max_new_tokens": args.max_new_tokens,
        "zero_cache_match_rate": zero_cache_match_rate,
        "zero_cache_all_match": zero_cache_match_rate == 1.0,
        "initial_any_differs": any(row["initial_differs_from_baseline"] for row in diagnostics),
        "mean_state_delta_l2_after_prefill": sum(row["state_delta_l2_after_prefill"] for row in diagnostics) / len(diagnostics),
        "rows": diagnostics,
    }
    (args.run_dir / (Path(args.output_name).stem + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    if zero_cache_match_rate < 0.7:
        raise AssertionError(
            f"zero-cache/no-cache token agreement too low ({zero_cache_match_rate:.0%}); "
            "expected occasional single-token float divergence over long generations, "
            "not a systematic mismatch"
        )
    if not report["initial_any_differs"]:
        raise AssertionError("steered generation never differs from baseline")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
