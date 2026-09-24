"""Sweep collected GDN deltas on SQuAD questions."""

import argparse
import json
from pathlib import Path
from typing import Any, cast

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedTokenizer

from gdn_interp import GDNRunner, LanguageDetector, truncate_svd

MODEL = "Qwen/Qwen3.5-9B"
SEED = 42
CONTEXT_WORDS = 300
MIN_CONTEXT_WORDS = 80
MAX_QUESTION_WORDS = 25

def squad_prompts(tokenizer: PreTrainedTokenizer, questions: int) -> tuple[list[dict[str, str]], list[str]]:
    """Cap context at 300 words for tractable prompts, retaining answerable passages and concise questions."""
    examples = []
    for row in load_dataset("rajpurkar/squad", split="validation").shuffle(seed=SEED):
        context = " ".join(row["context"].split()[:CONTEXT_WORDS])
        question = row["question"]
        if len(context.split()) >= MIN_CONTEXT_WORDS and len(question.split()) <= MAX_QUESTION_WORDS:
            examples.append({"source_id": str(row["id"]), "context": context, "question": question})
        if len(examples) == questions:
            break
    if len(examples) != questions:
        raise RuntimeError(f"found only {len(examples)} SQuAD questions")
    messages = [
        [
            {
                "role": "user",
                "content": f"Answer this question.\n\nContext:\n{example['context']}\n\nQuestion:\n{example['question']}",
            }
        ]
        for example in examples
    ]
    prompts = [
        tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False) for message in messages
    ]
    return examples, prompts


def load_deltas(path: Path) -> tuple[dict[int, torch.Tensor], str, str]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    concept_a = artifact.get("concept_a", artifact.get("language_a"))
    concept_b = artifact.get("concept_b", artifact.get("language_b"))
    if not isinstance(concept_a, str) or not isinstance(concept_b, str) or artifact.get("delta") != f"{concept_b} - {concept_a}":
        raise ValueError("deltas must be a concept-B-minus-concept-A artifact")
    deltas = {int(layer): delta for layer, delta in artifact["deltas"].items()}
    return deltas, concept_a, concept_b


def rows(
    examples: list[dict[str, str]],
    tokens: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    scale: float,
    rank: int,
    concept_a: str,
    concept_b: str,
    position: str = "pre-final",
    mode: str = "add",
    eos_token_id: int | list[int] | None = None,
) -> list[dict[str, Any]]:
    detector = LanguageDetector(concept_b)
    eos = tokenizer.eos_token_id if eos_token_id is None else eos_token_id
    eos_ids = torch.as_tensor(eos if isinstance(eos, list) else [eos], device=tokens.device)
    result: list[dict[str, Any]] = []
    for example, response_tokens in zip(examples, tokens):
        response = cast(str, tokenizer.decode(response_tokens, skip_special_tokens=True))
        language, confidence = detector.label(response), detector.confidence(response)
        result.append(
            {
                **example,
                "scale": scale,
                "concept_a": concept_a,
                "concept_b": concept_b,
                "target": concept_b,
                "response": response,
                "detected_language": language,
                "language_confidence": confidence,
                "truncated": not torch.isin(response_tokens, eos_ids).any().item(),
                "rank": rank,
                "group": "all",
                "period": 0,
                "position": position,
                "mode": mode,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep language-B-minus-language-A deltas on SQuAD.")
    parser.add_argument("--deltas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--questions", type=int, default=100)
    parser.add_argument("--scales", type=float, nargs="+", default=(0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0))
    parser.add_argument("--ranks", type=int, nargs="+", default=(2,))
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prompt-steer-position", type=int, default=-1)
    parser.add_argument("--prompt-steer-mode", choices=("exact", "chunk"), default="exact")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if min(args.questions, args.batch_size, args.max_new_tokens) < 1 or any(rank < 0 for rank in args.ranks):
        parser.error("counts must be positive; ranks must be non-negative")

    deltas, concept_a, concept_b = load_deltas(args.deltas)
    tokenizer = cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(MODEL))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    examples, prompts = squad_prompts(tokenizer, args.questions)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    (args.output / "examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2))
    with torch.inference_mode():
        baseline_runner = GDNRunner(model, tokenizer, sorted(deltas), deltas, normalize=args.normalize)
        baseline_generated: list[torch.Tensor] = []
        for start in range(0, len(prompts), args.batch_size):
            baseline_generated.append(
                baseline_runner.generate(
                    prompts[start : start + args.batch_size], prompt_steer_position=None, max_new_tokens=args.max_new_tokens
                )
            )
        baseline = rows(examples, torch.cat(baseline_generated), tokenizer, 0.0, -1, concept_a, concept_b, "none", "baseline", eos_token_id=model.generation_config.eos_token_id)
        (args.output / "baseline.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in baseline))
        baseline_share = sum(row["detected_language"] == concept_b for row in baseline) / len(baseline)
        print(f"baseline {concept_b}_share={baseline_share:.3f}", flush=True)
        for rank in dict.fromkeys(args.ranks):
            runner = GDNRunner(
                model,
                tokenizer,
                sorted(deltas),
                {layer: truncate_svd(delta, rank) for layer, delta in deltas.items()},
                normalize=args.normalize,
            )
            output = args.output if len(args.ranks) == 1 else args.output / f"rank{rank}"
            output.mkdir(exist_ok=True)
            for scale in args.scales:
                generated: list[torch.Tensor] = []
                for start in range(0, len(prompts), args.batch_size):
                    generated.append(
                        runner.generate(
                            prompts[start : start + args.batch_size],
                            scale=scale,
                            prompt_steer_position=args.prompt_steer_position,
                            prompt_steer_mode=args.prompt_steer_mode,
                            max_new_tokens=args.max_new_tokens,
                        )
                    )
                result = rows(
                    examples,
                    torch.cat(generated),
                    tokenizer,
                    scale,
                    rank,
                    concept_a,
                    concept_b,
                    str(args.prompt_steer_position),
                    eos_token_id=model.generation_config.eos_token_id,
                )
                path = output / f"scale_{scale:g}.jsonl"
                path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result))
                target_share = sum(row["detected_language"] == concept_b for row in result) / len(result)
                print(f"rank={rank} scale={scale:g} {concept_b}_share={target_share:.3f}", flush=True)


if __name__ == "__main__":
    main()
