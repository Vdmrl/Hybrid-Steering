"""Collect directed GDN state deltas from matched contrastive texts."""

import argparse
import heapq
from pathlib import Path
from typing import Any, cast

import torch
from datasets import load_dataset
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from gdn_interp import GDNRunner, LayerAccumulator, gdn_layers
from gdn_interp.concepts import CONCEPT_PAIRS

MODEL = "Qwen/Qwen3.5-9B"
SEED = 42
FEATURE_STORIES = "AntonKorznikov/feature_stories"
ALPHABETS = {"en": frozenset("abcdefghijklmnopqrstuvwxyz"), "ru": frozenset("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")}


def has_only_language_letters(text: str, language: str) -> bool:
    """Return whether every letter in text belongs to language's alphabet."""
    alphabet = ALPHABETS.get(language)
    if alphabet is None:
        raise ValueError(f"strict script filtering does not support {language!r}")
    return all(not character.isalpha() or character.lower() in alphabet for character in text)


def has_letters_and_is_not_uppercase(text: str) -> bool:
    """Reject digit-only and all-uppercase texts."""
    return any(character.isalpha() for character in text) and not text.isupper()


def relative_token_difference(tokenizer: PreTrainedTokenizerBase, first: str, second: str) -> float:
    """Return token-length difference relative to the longer text."""
    first_tokens, second_tokens = (len(ids) for ids in tokenizer([first, second]).input_ids)
    return abs(first_tokens - second_tokens) / max(first_tokens, second_tokens)


def translation_pairs(
    language_a: str,
    language_b: str,
    pairs: int,
    min_words: int,
    tokenizer: PreTrainedTokenizerBase | None = None,
    strict_script: bool = False,
    reject_digit_only_and_uppercase: bool = False,
    max_relative_token_difference: float | None = None,
) -> list[tuple[str, str]]:
    dataset = load_dataset(
        "Helsinki-NLP/opus-100", "-".join(sorted((language_a, language_b))), split="train"
    ).shuffle(seed=SEED)
    result: list[tuple[str, str]] = []
    candidates: list[tuple[float, int, str, str]] = []
    eligible = 0
    if max_relative_token_difference is not None and tokenizer is None:
        raise ValueError("tokenizer is required for token-length filtering")
    for index, row in enumerate(dataset):
        translation = row["translation"]
        a, b = translation[language_a].strip(), translation[language_b].strip()
        if min(len(a.split()), len(b.split())) <= min_words:
            continue
        if strict_script and not (has_only_language_letters(a, language_a) and has_only_language_letters(b, language_b)):
            continue
        if reject_digit_only_and_uppercase and not (has_letters_and_is_not_uppercase(a) and has_letters_and_is_not_uppercase(b)):
            continue
        if max_relative_token_difference is None:
            result.append((a, b))
            if len(result) == pairs:
                return result
            continue
        difference = relative_token_difference(tokenizer, a, b)
        if difference > max_relative_token_difference:
            continue
        eligible += 1
        candidate = (-difference, -index, a, b)
        if len(candidates) < pairs:
            heapq.heappush(candidates, candidate)
        elif candidate > candidates[0]:
            heapq.heapreplace(candidates, candidate)
    if max_relative_token_difference is not None and len(candidates) == pairs:
        return [(a, b) for _, _, a, b in sorted(candidates, reverse=True)]
    raise RuntimeError(f"found only {eligible if max_relative_token_difference is not None else len(result)} usable translations")


def feature_story_pairs(concept_key: str, pairs: int) -> tuple[str, str, list[tuple[str, str]]]:
    concept, expected_antagonist = CONCEPT_PAIRS[concept_key]
    result: list[tuple[str, str]] = []
    antagonist: str | None = None
    dataset = load_dataset(FEATURE_STORIES, split="train")
    for row in dataset:
        if row["concept"] != concept:
            continue
        current_antagonist = str(row["antagonist"])
        if antagonist is None:
            antagonist = current_antagonist
        elif antagonist != current_antagonist:
            raise ValueError(f"concept {concept!r} has multiple antagonists")
        concept_text = str(row["concept_text"]).strip()
        antagonist_text = str(row["antagonist_text"]).strip()
        if concept_text and antagonist_text:
            result.append((concept_text, antagonist_text))
        if len(result) == pairs:
            if antagonist != expected_antagonist:
                raise ValueError(f"expected antagonist {expected_antagonist!r}, found {antagonist!r}")
            return concept, antagonist, result
    raise RuntimeError(f"found only {len(result)} usable pairs for concept {concept!r}")


def zero_deltas(model: Any, layers: list[int]) -> dict[int, Tensor]:
    return {
        layer: torch.zeros(
            model.model.layers[layer].linear_attn.num_v_heads,
            model.model.layers[layer].linear_attn.head_k_dim,
            model.model.layers[layer].linear_attn.head_v_dim,
            device=next(model.parameters()).device,
        )
        for layer in layers
    }


def final_states(runner: GDNRunner, texts: list[str]) -> dict[int, Tensor]:
    encoded = runner.tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt").to(runner.device)
    lengths = encoded.attention_mask.sum(-1)
    trace = runner.forward(texts, prompt_steer_position=None)
    states: dict[int, Tensor] = {}
    filled = torch.zeros(len(texts), dtype=torch.bool, device=runner.device)
    for position, by_layer in trace.states.items():
        indices = trace.indices[position]
        final = lengths[indices].eq(position)
        filled[indices[final]] = True
        for layer, values in by_layer.items():
            states.setdefault(layer, values.new_empty((len(texts), *values.shape[1:])))
            states[layer][indices[final]] = values[final]
    if not filled.all() or len(states) != len(runner.layers):
        raise RuntimeError("failed to collect a final recurrent state for every input")
    return states


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect concept-B-minus-concept-A GDN state deltas from matched texts.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concept", choices=CONCEPT_PAIRS)
    parser.add_argument("--language-a", default="en")
    parser.add_argument("--language-b", default="ru")
    parser.add_argument("--pairs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--min-words", type=int, default=20)
    parser.add_argument("--strict-script", action="store_true")
    parser.add_argument("--reject-digit-only-and-uppercase", action="store_true")
    parser.add_argument("--max-relative-token-difference", type=float)
    args = parser.parse_args()
    if min(args.pairs, args.batch_size, args.min_words) < 1:
        parser.error("--pairs, --batch-size, and --min-words must be positive")

    if args.concept is None and args.language_a == args.language_b:
        parser.error("--language-a and --language-b must differ")
    if args.concept is not None and (
        args.strict_script or args.reject_digit_only_and_uppercase or args.max_relative_token_difference is not None
    ):
        parser.error("translation filters cannot be used with --concept")
    if args.max_relative_token_difference is not None and args.max_relative_token_difference < 0:
        parser.error("--max-relative-token-difference must be non-negative")
    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(MODEL))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.concept is None:
        concept_a, concept_b = args.language_a, args.language_b
        pairs = translation_pairs(
            args.language_a,
            args.language_b,
            args.pairs,
            args.min_words,
            tokenizer,
            args.strict_script,
            args.reject_digit_only_and_uppercase,
            args.max_relative_token_difference,
        )
        source = "Helsinki-NLP/opus-100"
    else:
        concept_a, concept_b, pairs = feature_story_pairs(args.concept, args.pairs)
        source = FEATURE_STORIES
    print(f"using {len(pairs)} matched pairs ({2 * len(pairs)} texts)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    layers = gdn_layers(model)
    runner = GDNRunner(model, tokenizer, layers, zero_deltas(model, layers), normalize=False)
    accumulators: dict[int, LayerAccumulator] | None = None
    with torch.inference_mode():
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start : start + args.batch_size]
            states_a = final_states(runner, [a for a, _ in batch])
            states_b = final_states(runner, [b for _, b in batch])
            if accumulators is None:
                accumulators = {layer: LayerAccumulator(tuple(state.shape[1:]), state.device) for layer, state in states_a.items()}
            for layer, accumulator in accumulators.items():
                accumulator.add(states_b[layer], states_a[layer])
            print(f"{start + len(batch)}/{len(pairs)} matched pairs", flush=True)
    assert accumulators is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": MODEL,
            "source": source,
            "concept_key": args.concept,
            "pairs": len(pairs),
            "concept_a": concept_a,
            "concept_b": concept_b,
            "delta": f"{concept_b} - {concept_a}",
            "deltas": {layer: accumulator.mean_delta.cpu() for layer, accumulator in accumulators.items()},
        },
        args.output,
    )


if __name__ == "__main__":
    main()
