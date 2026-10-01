"""Generate target and plain answer pairs for a non-language concept.

The steered model writes both sides to the same neutral question. The
positive side shows the concept; the negative side is a plain answer that
neither names nor denies it. Pairs are matched within a question by token
length.

``questions`` asks the model for everyday questions by domain and drops any
question that is close to the evaluation pool in ``experiments/forgetting``.
``eval`` writes one evaluation pool from domains the pairs never saw.
``experiments/forgetting/questions.py`` splits it into tune and held-out by seed.
``pairs`` samples several answers per side, filters them by the concept's
lexicon, and writes ``pairs.jsonl`` in the dataset schema.

    uv run python experiments/concept-pairs/run.py questions --output runs/pairs-v2
    uv run python experiments/concept-pairs/run.py pairs --concept theistic_framing \\
        --output runs/pairs-v2
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch

from hybrid_steering import load_runtime
from hybrid_steering.runtime import batched, write_jsonl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "forgetting"))
from questions import SIMPLE_QUESTIONS  # noqa: E402

MODEL = "Qwen/Qwen3.5-9B"
LENGTH_RATIO = (0.8, 1.25)
FORMAT = (
    "Write one paragraph of 60 to 90 words. Plain text only: no title, no list, "
    "no markdown, no preamble. Address the question directly."
)
DOMAINS = (
    "cooking at home",
    "travel and holidays",
    "work and careers",
    "school and studying",
    "daily health habits",
    "personal finances",
    "friendship",
    "family life",
    "hobbies",
    "sports and exercise",
    "weather and seasons",
    "city life",
    "nature and the outdoors",
    "using everyday technology",
    "shopping",
    "pets",
    "gardening",
    "commuting and transport",
    "music",
    "books and reading",
    "art and crafts",
    "volunteering and community",
    "time management",
    "moving to a new home",
    "neighbors",
    "learning a new skill",
    "job interviews",
    "household chores",
    "planning events",
    "small business",
)
EVAL_DOMAINS = (
    "photography",
    "board and card games",
    "car ownership",
    "home repairs",
    "theatre and museums",
    "clothing and fashion",
    "sleep and rest",
    "paperwork and public services",
)
PLAIN = "Answer the question in a plain, direct, practical way."

# positive instruction, words the negative must avoid, words the positive must contain
CONCEPTS = {
    "theistic_framing": (
        "Answer as a sincere religious believer: build the answer on the premise that God "
        "acts, guides, or gives meaning, for example through faith, prayer, or divine purpose. "
        "Do not mention atheism, doubt, or science versus religion.",
        PLAIN + " Do not mention religion, God, faith, spirituality, or their absence.",
        r"\b(god|gods|divine|faith|deity|supernatural|creator|pray\w*|spiritu\w*|bless\w*|"
        r"heaven\w*|lord|sacred|providence|religio\w*|church|atheis\w*|scripture|soul)\b",
        r"\b(god|divine|faith|pray\w*|lord|providence)\b",
    ),
    "probabilistic_framing": (
        "Frame the answer in graded likelihoods: give rough probabilities, percentages, or "
        "odds for the possible outcomes, and say which are more or less likely. Do not "
        "discuss probability or uncertainty as a topic.",
        PLAIN + " State things with confidence. Do not use probabilities, percentages, odds, "
        "or words such as likely, chance, possibly, or uncertain.",
        r"(probab\w*|likel\w*|unlikel\w*|chance\w*|odds|percent|%|uncertain\w*|possib\w*)",
        r"(probab\w*|likel\w*|chance\w*|odds|percent|%)",
    ),
    "fictional_narrative": (
        "Answer by telling a short invented scene with a named character and concrete events "
        "that shows the answer. Do not call it a story and do not mention fiction.",
        PLAIN + " Do not tell a story, invent people or events, or mention stories, fiction, "
        "or narratives.",
        r"\b(story|stories|fiction\w*|narrative\w*|once upon|character\w*|plot|tale\w*)\b",
        None,
    ),
    "comparative_framing": (
        "Answer by explicitly comparing the subject with a named alternative, a baseline, or "
        "an earlier state, using comparisons such as 'compared with', 'unlike', or "
        "'more ... than'. Do not discuss comparison as a topic.",
        PLAIN + " Describe the subject on its own terms. Do not compare it with anything and "
        "do not mention alternatives or differences.",
        r"(compar\w*|contrast\w*|unlike|versus|\bvs\b|\bthan\b|whereas|alternative\w*|"
        r"similar\w*|differen\w*)",
        r"(compar\w*|unlike|\bthan\b|whereas|versus)",
    ),
    "technical_language": (
        "Answer like a domain specialist: use the precise technical terms of the relevant "
        "field and explain the underlying mechanism, how and why it works. Do not define "
        "terms for a lay reader and do not mention jargon or expertise as a topic.",
        PLAIN + " Use everyday words a child would know. Do not use technical terms or "
        "explain mechanisms.",
        None,
        None,
    ),
}


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def close_to(question: str, pool: list[set[str]], threshold: float = 0.5) -> bool:
    mine = words(question)
    return any(len(mine & other) / max(1, len(mine | other)) >= threshold for other in pool)


def sample(
    model, tokenizer, prompts: list[str], *, batch_size: int, max_new_tokens: int, seed: int
) -> list[tuple[str, bool]]:
    """Sampled completions and whether each one ended with EOS."""
    torch.manual_seed(seed)
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]
    tokenizer.padding_side = "left"
    results: list[tuple[str, bool]] = []
    for index, batch in enumerate(batched(texts, batch_size)):
        encoded = tokenizer(batch, add_special_tokens=False, padding=True, return_tensors="pt").to(
            model.device
        )
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        new = output[:, encoded.input_ids.shape[1] :]
        eos = torch.as_tensor(
            model.generation_config.eos_token_id or [tokenizer.eos_token_id], device=new.device
        ).flatten()
        for row in new:
            results.append(
                (
                    tokenizer.decode(row, skip_special_tokens=True).strip(),
                    bool(torch.isin(row, eos).any()),
                )
            )
        print(f"generated {min((index + 1) * batch_size, len(texts))}/{len(texts)}", flush=True)
    return results


def generate_questions(
    args: argparse.Namespace, domains: tuple[str, ...], excluded: list[str]
) -> list[str]:
    model, tokenizer = load_runtime(args.model)
    prompts = [
        f"Write {args.per_domain} different everyday questions about {domain} that a person "
        "might ask a helpful assistant. Each question must be answerable in one paragraph, "
        "open-ended, and neutral: no religion, statistics, fiction, or comparisons in the "
        "question itself. One question per line, no numbering, nothing else."
        for domain in domains
        for _ in range(args.rounds)
    ]
    outputs = sample(
        model,
        tokenizer,
        prompts,
        batch_size=args.batch_size,
        max_new_tokens=1024,
        seed=args.seed,
    )
    held = [words(question) for question in excluded]
    kept: list[str] = []
    seen: list[set[str]] = []
    for text, _ in outputs:
        for line in text.splitlines():
            question = re.sub(r"^[\s\-*\d.)]+", "", line).strip()
            if not question.endswith("?") or not 6 <= len(question.split()) <= 25:
                continue
            if close_to(question, held) or close_to(question, seen, 0.7):
                continue
            kept.append(question)
            seen.append(words(question))
    random.Random(args.seed).shuffle(kept)
    return kept


def make_questions(args: argparse.Namespace) -> None:
    kept = generate_questions(args, DOMAINS, list(SIMPLE_QUESTIONS))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "questions.json").write_text(json.dumps(kept[: args.keep], indent=1) + "\n")
    print(f"kept {min(len(kept), args.keep)} of {len(kept)} questions", flush=True)


def make_eval(args: argparse.Namespace) -> None:
    """Evaluation questions from domains that the pairs never saw."""
    pair_questions = json.loads((args.output / "questions.json").read_text())
    kept = generate_questions(args, EVAL_DOMAINS, [*SIMPLE_QUESTIONS, *pair_questions])
    rows = [
        {"source_id": f"eval-{index:04d}", "question": question}
        for index, question in enumerate(kept[: args.keep])
    ]
    write_jsonl(args.output / "eval_questions.jsonl", rows)
    print(f"kept {len(rows)} of {len(kept)} eval questions", flush=True)


def match(positives: list[tuple[str, int]], negatives: list[tuple[str, int]]) -> list[tuple]:
    """Pair texts of one question so that token lengths stay within LENGTH_RATIO."""
    free = list(negatives)
    pairs = []
    for positive, positive_length in positives:
        candidates = [
            (abs(positive_length / length - 1), index)
            for index, (_, length) in enumerate(free)
            if LENGTH_RATIO[0] <= positive_length / length <= LENGTH_RATIO[1]
        ]
        if not candidates:
            continue
        _, index = min(candidates)
        negative, _ = free.pop(index)
        pairs.append((positive, negative))
    return pairs


def make_pairs(args: argparse.Namespace) -> None:
    instruction, plain, avoid, require = CONCEPTS[args.concept]
    questions = json.loads((args.output / "questions.json").read_text())
    model, tokenizer = load_runtime(args.model)
    jobs = [
        (side, index, f"{prompt}\n{FORMAT}\n\nQuestion: {question}")
        for index, question in enumerate(questions)
        for side, prompt in (("positive", instruction), ("negative", plain))
        for _ in range(args.samples)
    ]
    outputs = sample(
        model,
        tokenizer,
        [prompt for *_, prompt in jobs],
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    kept: dict[tuple[str, int], list[tuple[str, int]]] = {}
    dropped = {"truncated": 0, "lexicon": 0, "format": 0}
    for (side, index, _), (text, finished) in zip(jobs, outputs, strict=True):
        if not finished:
            dropped["truncated"] += 1
            continue
        if "\n" in text.strip() or re.match(r"^(sure|here|certainly|okay)\b", text, re.I):
            dropped["format"] += 1
            continue
        if side == "negative" and avoid and re.search(avoid, text, re.I):
            dropped["lexicon"] += 1
            continue
        if side == "positive" and require and not re.search(require, text, re.I):
            dropped["lexicon"] += 1
            continue
        length = len(tokenizer(text, add_special_tokens=False).input_ids)
        kept.setdefault((side, index), []).append((text, length))
    rows = []
    for index, question in enumerate(questions):
        for positive, negative in match(
            kept.get(("positive", index), []), kept.get(("negative", index), [])
        ):
            rows.append(
                {
                    "pair_id": f"plain-{args.concept}-{len(rows):05d}",
                    "split": "train",
                    "negative_text": negative,
                    "positive_text": positive,
                    "source_id": f"question-{index:04d}",
                    "source_license": "generated",
                    "source_question": question,
                    "generation_model": args.model,
                }
            )
    directory = args.output / f"plain-{args.concept}"
    write_jsonl(directory / "pairs.jsonl", rows)
    (directory / "generation.json").write_text(
        json.dumps(
            {
                "concept": args.concept,
                "positive_instruction": instruction,
                "negative_instruction": plain,
                "format": FORMAT,
                "samples_per_side": args.samples,
                "questions": len(questions),
                "generations": len(jobs),
                "dropped": dropped,
                "pairs": len(rows),
                "seed": args.seed,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"{args.concept}: {len(rows)} pairs, dropped {dropped}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("questions", "eval", "pairs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--concept", choices=sorted(CONCEPTS))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--per-domain", type=int, default=25)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--keep", type=int, default=400)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()
    if args.stage == "questions":
        make_questions(args)
        return
    if args.stage == "eval":
        make_eval(args)
        return
    if args.concept is None:
        parser.error("pairs needs --concept")
    make_pairs(args)


if __name__ == "__main__":
    main()
