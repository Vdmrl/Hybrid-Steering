"""Compare native generation and actual steering on a small local SQuAD batch."""

import argparse
import json
import time
from html import escape
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gdn_interp import GDNRunner, GDNTrace, LanguageDetector, truncate_svd
from .squad_sweep import MODEL, load_deltas, squad_prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deltas", type=Path, required=True)
    parser.add_argument("--questions", type=int, default=4)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.5, 0.8, 1.0, 1.2])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--prompt-steer-mode", choices=("exact", "chunk"), default="exact")
    args = parser.parse_args()
    if min(args.questions, args.max_new_tokens) < 1 or min(args.rank, args.offset) < 0:
        parser.error("counts must be positive; rank and offset must be non-negative")
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)

    deltas, _, target = load_deltas(args.deltas)
    deltas = {layer: truncate_svd(delta, args.rank) for layer, delta in deltas.items()}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    examples, prompts = squad_prompts(tokenizer, args.questions + args.offset)
    examples, prompts = examples[args.offset:], prompts[args.offset:]
    contents = [example["question"] for example in examples]

    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16, device_map="cuda").eval()
    runner = GDNRunner(model, tokenizer, sorted(deltas), deltas, normalize=True)
    encoded = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt").to("cuda")
    results: dict[str, list[str]] = {}
    timings: dict[str, float] = {}
    generated: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        native = model.generate(**encoded, do_sample=False, max_new_tokens=args.max_new_tokens)
        print("generation config", model.generation_config, flush=True)
        results["native"] = tokenizer.batch_decode(native[:, encoded.input_ids.shape[1]:], skip_special_tokens=True)
        print("native complete", flush=True)
        conditions = [("baseline", None, 0.0), ("zero_injection", -1, 0.0)] + [(f"steered_{scale:g}", -1, scale) for scale in args.scales]
        for name, position, scale in conditions:
            start = time.perf_counter()
            tokens = runner.generate(prompts, scale=scale, prompt_steer_position=position, prompt_steer_mode=args.prompt_steer_mode, max_new_tokens=args.max_new_tokens)
            torch.cuda.synchronize()
            timings[name] = time.perf_counter() - start
            generated[name] = tokens
            results[name] = tokenizer.batch_decode(tokens, skip_special_tokens=True)
            print(name, results[name], flush=True)
            (args.output / "responses.json").write_text(json.dumps({"examples": examples, "prompts": prompts, "results": results}, ensure_ascii=False, indent=2))

        mixed_scales = torch.tensor([args.scales[-1] if index % 2 else 0.0 for index in range(len(prompts))], device="cuda")
        mixed = runner.generate(prompts, scale=mixed_scales, prompt_steer_mode=args.prompt_steer_mode, max_new_tokens=args.max_new_tokens)
        results["mixed"] = tokenizer.batch_decode(mixed, skip_special_tokens=True)
        expected = torch.stack([generated[f"steered_{args.scales[-1]:g}"][index] if index % 2 else generated["zero_injection"][index] for index in range(len(prompts))])
        torch.testing.assert_close(mixed, expected, atol=0, rtol=0)
        torch.testing.assert_close(generated["baseline"], generated["zero_injection"], atol=0, rtol=0)
        assert results["baseline"] == results["native"]

        trace = GDNTrace()
        traced = runner.generate(prompts[:2], scale=args.scales[0], prompt_steer_mode=args.prompt_steer_mode, max_new_tokens=2, trace=trace)
        plain = runner.generate(prompts[:2], scale=args.scales[0], prompt_steer_mode=args.prompt_steer_mode, max_new_tokens=2)
        torch.testing.assert_close(traced, plain, atol=0, rtol=0)
        assert trace.states and trace.outputs
        del trace

    detector = LanguageDetector(target)
    matches = {name: sum(a == b for a, b in zip(results["native"], values, strict=True)) for name, values in results.items()}
    lengths = encoded.attention_mask.sum(-1)
    columns = torch.full_like(lengths, encoded.input_ids.shape[1] - 1)
    if args.prompt_steer_mode == "chunk":
        columns = torch.maximum(columns // 64 * 64, encoded.input_ids.shape[1] - lengths)
    metadata: dict[str, Any] = {
        "model": MODEL, "device": torch.cuda.get_device_name(), "dtype": "bfloat16",
        "deltas": str(args.deltas), "normalize": True, "rank": args.rank,
        "n": len(contents), "offset": args.offset, "prompt_steer_mode": args.prompt_steer_mode,
        "prompt_positions": (columns - encoded.input_ids.shape[1] + lengths).tolist(),
        "native_text_matches": matches, "mixed_batch_matches": True, "trace_matches": True,
        "seconds": timings,
        "target_language_counts": {name: sum(detector.label(text) == target for text in values) for name, values in results.items()},
    }
    (args.output / "responses.json").write_text(json.dumps({"examples": examples, "prompts": prompts, "results": results}, ensure_ascii=False, indent=2))
    (args.output / "summary.json").write_text(json.dumps(metadata, indent=2))
    table = "".join("<tr><td>" + escape(content) + "</td>" + "".join("<td>" + escape(values[index]) + "</td>" for values in results.values()) + "</tr>" for index, content in enumerate(contents))
    (args.output / "report.html").write_text('<!doctype html><meta charset="utf-8"><title>Local steering smoke</title><style>body{font:15px system-ui;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:12px;vertical-align:top;white-space:pre-wrap}</style><h1>Local GPU steering smoke</h1><pre>' + escape(json.dumps(metadata, indent=2)) + '</pre><table><tr><th>Prompt</th>' + ''.join('<th>' + escape(name) + '</th>' for name in results) + '</tr>' + table + '</table>')
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
