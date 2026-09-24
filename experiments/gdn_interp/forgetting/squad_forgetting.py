"""Measure how filler-token distance weakens one S_0 concept intervention."""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, cast

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedTokenizer

from experiments.forgetting.fillers import FILLERS
from gdn_interp import AnswerEquivalenceDetector, GDNRunner, GDNTrace, LLMJudge, LanguageDetector, truncate_svd

MODEL = "Qwen/Qwen3.5-9B"
SEED = 42
PREFIX_LENGTHS = (0, 32, 64, 128, 256, 512, 1024, 2048, 4096)
DEFAULT_SCALES = tuple(index / 2 for index in range(2, 11))
RANK = 2
NORMALIZE = True
MAX_NEW_TOKENS = 64
TARGET_CONCEPT = "ru"
DETECTOR = LanguageDetector(TARGET_CONCEPT)


def squad_questions(questions: int, seed: int) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for row in load_dataset("rajpurkar/squad", split="validation").shuffle(seed=seed):
        question = str(row["question"])
        if question:
            result.append({"source_id": str(row["id"]), "question": question})
        if len(result) == questions:
            return result
    raise RuntimeError(f"found only {len(result)} SQuAD questions")


def load_deltas(path: Path) -> dict[int, torch.Tensor]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    deltas = artifact.get("deltas")
    if not isinstance(deltas, dict) or not deltas:
        raise ValueError("deltas artifact must contain non-empty deltas")
    return {int(layer): delta for layer, delta in deltas.items()}


def filler_prefixes(tokenizer: PreTrainedTokenizer, filler_index: int) -> dict[int, str]:
    if not 0 <= filler_index < len(FILLERS):
        raise ValueError(f"filler_index must be in [0, {len(FILLERS) - 1}]")
    tokens = tokenizer(FILLERS[filler_index], add_special_tokens=False).input_ids
    if len(tokens) < max(PREFIX_LENGTHS):
        raise ValueError(f"filler {filler_index} has only {len(tokens)} tokens, need {max(PREFIX_LENGTHS)}")
    return {length: cast(str, tokenizer.decode(tokens[:length], clean_up_tokenization_spaces=False)) for length in PREFIX_LENGTHS}


def prompts(tokenizer: PreTrainedTokenizer, examples: list[dict[str, str]], filler: str) -> list[str]:
    return [
        cast(
            str,
            tokenizer.apply_chat_template(
                ([{"role": "user", "content": filler}] if filler else []) + [{"role": "user", "content": example["question"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ),
        )
        for example in examples
    ]


def trace_vectors(trace: GDNTrace, batch_size: int) -> dict[int, dict[int, torch.Tensor]]:
    vectors: dict[int, dict[int, torch.Tensor]] = {index: {} for index in range(batch_size)}
    for length, indices in trace.prefill_indices.items():
        for layer, values in trace.prefill_outputs[length].items():
            for row, value in zip(indices.tolist(), values):
                vectors[row][layer] = value
    if any(len(layers) != len(trace.prefill_outputs[next(iter(trace.prefill_outputs))]) for layers in vectors.values()):
        raise RuntimeError("prefill trace does not contain every configured GDN layer for every prompt")
    return vectors


def output_metrics(
    baseline: GDNTrace, steered: GDNTrace, batch_size: int
) -> list[dict[str, dict[str, float]]]:
    baseline_vectors = trace_vectors(baseline, batch_size)
    steered_vectors = trace_vectors(steered, batch_size)
    result: list[dict[str, dict[str, float]]] = []
    for baseline_layers, steered_layers in zip(baseline_vectors.values(), steered_vectors.values()):
        norms: dict[str, float] = {}
        cosine: dict[str, float] = {}
        for layer in baseline_layers:
            first = baseline_layers[layer].float().flatten()
            second = steered_layers[layer].float().flatten()
            norms[str(layer)] = torch.linalg.vector_norm(second - first).item()
            denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
            cosine[str(layer)] = torch.where(
                denominator > 0, torch.dot(first, second) / denominator, torch.zeros_like(denominator)
            ).item()
        result.append({"output_delta_norm": norms, "output_cosine": cosine})
    return result


def response_rows(
    examples: list[dict[str, str]],
    tokens: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    prefix_length: int,
    condition: str,
    metrics: list[dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for example, response_tokens in zip(examples, tokens):
        response = cast(str, tokenizer.decode(response_tokens, skip_special_tokens=True))
        result.append(
            {
                **example,
                "prefix_length": prefix_length,
                "condition": condition,
                "target_concept": TARGET_CONCEPT,
                "response": response,
                "detected_language": DETECTOR.label(response),
                "language_confidence": DETECTOR.confidence(response),
                "truncated": not response_tokens.eq(tokenizer.eos_token_id).any().item(),
                **metrics[len(result)],
            }
        )
    return result


def judge_rows(
    rows: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    detector: AnswerEquivalenceDetector,
    judge: LLMJudge,
) -> None:
    if [row["source_id"] for row in rows] != [row["source_id"] for row in baseline]:
        raise ValueError("steered and baseline rows must have matching source IDs")

    prompts = [
        detector.prompt("", row["question"], row["response"], baseline_row["response"])
        for row, baseline_row in zip(rows, baseline, strict=True)
    ]
    judge_outputs = judge(prompts)
    if len(judge_outputs) != len(rows):
        raise ValueError("judge must return one output per row")

    for row, raw in zip(rows, judge_outputs, strict=True):
        row["factual_match"] = detector.parse(raw)
        row["judge_raw"] = raw


def scale_name(scale: float) -> str:
    return f"scale_{scale:g}".replace(".", "_")


def prefill_batch_size(prefix_length: int, batch_size: int, token_budget: int) -> int:
    if prefix_length >= 4096:
        return 1
    return min(batch_size, max(1, token_budget // max(1, prefix_length)))


def cpu_prefill_trace(trace: GDNTrace) -> GDNTrace:
    return GDNTrace(
        prefill_outputs={
            length: {layer: values.cpu() for layer, values in layers.items()}
            for length, layers in trace.prefill_outputs.items()
        },
        prefill_indices={length: indices.cpu() for length, indices in trace.prefill_indices.items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure S_0 concept-steering retention after user-message filler.")
    parser.add_argument("--deltas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--filler-index", type=int)
    parser.add_argument("--questions", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--prefill-token-budget", type=int, default=8192)
    parser.add_argument("--scales", type=float, nargs="+", default=DEFAULT_SCALES)
    parser.add_argument("--judge-max-tokens", type=int, default=16)
    parser.add_argument("--judge-max-workers", type=int, default=16)
    args = parser.parse_args()
    if args.questions < 1:
        parser.error("--questions must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.prefill_token_budget < 1:
        parser.error("--prefill-token-budget must be positive")
    if args.judge_max_tokens < 1 or args.judge_max_workers < 1:
        parser.error("--judge-max-tokens and --judge-max-workers must be positive")
    if any(scale <= 0 for scale in args.scales):
        parser.error("--scales must contain positive values")
    judge_model = os.environ.get("JUDGE_MODEL")
    if not judge_model:
        parser.error("JUDGE_MODEL must be set")

    deltas = load_deltas(args.deltas)
    tokenizer = cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(MODEL))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    examples = squad_questions(args.questions, args.seed)
    filler_index = args.filler_index if args.filler_index is not None else random.Random(args.seed).randrange(len(FILLERS))
    prefixes = filler_prefixes(tokenizer, filler_index)
    runner = GDNRunner(
        model,
        tokenizer,
        sorted(deltas),
        {layer: truncate_svd(delta, RANK) for layer, delta in deltas.items()},
        normalize=NORMALIZE,
    )
    equivalence_detector = AnswerEquivalenceDetector()
    judge = LLMJudge(
        model=judge_model,
        api_base=os.environ.get("JUDGE_API_BASE"),
        api_key=os.environ.get("JUDGE_API_KEY"),
        max_tokens=args.judge_max_tokens,
        max_workers=args.judge_max_workers,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n")
    baseline_cache: dict[int, tuple[torch.Tensor, GDNTrace]] = {}
    sweep_summary: list[dict[str, Any]] = []
    with torch.inference_mode():
        for prefix_length, filler in prefixes.items():
            experiment_prompts = prompts(tokenizer, examples, filler)
            current_batch_size = prefill_batch_size(prefix_length, args.batch_size, args.prefill_token_budget)
            print(f"prefix_length={prefix_length} batch_size={current_batch_size}", flush=True)
            baseline_tokens: list[torch.Tensor] = []
            baseline_traces: list[GDNTrace] = []
            for start in range(0, len(experiment_prompts), current_batch_size):
                batch = experiment_prompts[start : start + current_batch_size]
                baseline_trace = GDNTrace()
                baseline_tokens.append(
                    runner.generate(batch, prompt_steer_position=None, max_new_tokens=MAX_NEW_TOKENS, trace=baseline_trace).cpu()
                )
                baseline_traces.append(cpu_prefill_trace(baseline_trace))
            baseline_cache[prefix_length] = (torch.cat(baseline_tokens), baseline_traces)

        for scale in args.scales:
            scale_output = args.output / scale_name(scale)
            scale_output.mkdir(parents=True, exist_ok=True)
            summary: list[dict[str, Any]] = []
            for prefix_length, filler in prefixes.items():
                experiment_prompts = prompts(tokenizer, examples, filler)
                baseline_tokens, baseline_traces = baseline_cache[prefix_length]
                current_batch_size = prefill_batch_size(prefix_length, args.batch_size, args.prefill_token_budget)
                steered_tokens: list[torch.Tensor] = []
                metrics: list[dict[str, dict[str, float]]] = []
                for start in range(0, len(experiment_prompts), current_batch_size):
                    batch = experiment_prompts[start : start + current_batch_size]
                    steered_trace = GDNTrace()
                    steered_tokens.append(
                        runner.generate(batch, scale=scale, prompt_steer_position=0, max_new_tokens=MAX_NEW_TOKENS, trace=steered_trace).cpu()
                    )
                    steered_trace = cpu_prefill_trace(steered_trace)
                    metrics.extend(output_metrics(baseline_traces[start // current_batch_size], steered_trace, len(batch)))
                baseline_rows = response_rows(examples, baseline_tokens, tokenizer, prefix_length, "baseline", metrics)
                steered_rows = response_rows(examples, torch.cat(steered_tokens), tokenizer, prefix_length, "steered", metrics)
                judge_rows(steered_rows, baseline_rows, equivalence_detector, judge)
                rows = [*baseline_rows, *steered_rows]
                (scale_output / f"prefix_{prefix_length}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
                baseline_share = sum(row["detected_language"] == TARGET_CONCEPT for row in rows if row["condition"] == "baseline") / len(examples)
                steered_share = sum(row["detected_language"] == TARGET_CONCEPT for row in rows if row["condition"] == "steered") / len(examples)
                factual_match_share = sum(row["factual_match"] for row in steered_rows) / len(examples)
                result = {
                    "scale": scale,
                    "prefix_length": prefix_length,
                    "target_concept": TARGET_CONCEPT,
                    "baseline_concept_share": baseline_share,
                    "steered_concept_share": steered_share,
                    "steering_effect": steered_share - baseline_share,
                    "factual_match_share": factual_match_share,
                }
                summary.append(result)
                print(
                    f"scale={scale:g} prefix_length={prefix_length} baseline={baseline_share:.3f} steered={steered_share:.3f} effect={steered_share - baseline_share:+.3f} factual_match={factual_match_share:.3f}",
                    flush=True,
                )
            (scale_output / "summary.json").write_text(json.dumps({"scale": scale, "results": summary}, ensure_ascii=False, indent=2) + "\n")
            sweep_summary.extend(summary)
    (args.output / "sweep_summary.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "target_concept": TARGET_CONCEPT,
                "filler_index": filler_index,
                "seed": args.seed,
                "scales": args.scales,
                "rank": RANK,
                "normalize": NORMALIZE,
                "questions": len(examples),
                "results": sweep_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
