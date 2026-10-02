"""Does the answer_quality judge penalize the answer language?

The model answers eval questions in English. The judge model translates every
answer into each target language, keeping content, formatting, and the cut at
the token limit. The answer_quality guide then scores the English answer and
every translation against the same English question. Content is held fixed,
so a quality gap is the judge's language penalty.

Reported per language: mean quality, the paired difference to English with a
bootstrap 95% interval, the share of exact ties, and the largest drops.

    uv run python experiments/language-quality/run.py \\
        --questions runs/pairs/eval_questions.jsonl --output runs/language-quality
"""

from __future__ import annotations

import argparse
import random
from html import escape
from pathlib import Path

from hybrid_steering import Runner, chat_prompts, gdn_layers, load_runtime
from hybrid_steering.judge import score_steering
from hybrid_steering.judge.client import complete_batch
from hybrid_steering.report import write_page
from hybrid_steering.runtime import batched, read_jsonl, write_jsonl

LANGUAGES = {"ru": "Russian", "fr": "French", "zh": "Chinese (Simplified)", "ar": "Arabic"}
TRANSLATE = (
    "Translate the text below into {language}. Keep the meaning, the Markdown "
    "formatting, and the line breaks. If the text stops mid-sentence, stop the "
    "translation at the same place. Output only the translation.\n\n{text}"
)


def answer(model_name: str, questions: list[str], max_new_tokens: int, batch_size: int):
    model, tokenizer = load_runtime(model_name)
    runner = Runner(model, tokenizer, gdn_layers(model), normalize=False)
    texts = chat_prompts(tokenizer, questions)
    answers = []
    for batch in batched(texts, batch_size):
        for row in runner.generate(batch, prompt_position=None, max_new_tokens=max_new_tokens):
            ids = [int(t) for t in row if int(t) != tokenizer.pad_token_id]
            answers.append(tokenizer.decode(ids, skip_special_tokens=True))
    return answers


def bootstrap_mean(values: list[float], seed: int, rounds: int = 2000) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(rounds))
    return means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--judge-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    pool = read_jsonl(args.questions)
    questions = [
        row["question"] for row in random.Random(args.seed).sample(pool, min(args.count, len(pool)))
    ]
    english = answer(args.model, questions, args.max_new_tokens, args.batch_size)
    texts = {"en": english}
    for code, language in LANGUAGES.items():
        texts[code] = complete_batch(
            [
                [{"role": "user", "content": TRANSLATE.format(language=language, text=text)}]
                for text in english
            ],
            max_tokens=2048,
            batch_size=args.judge_batch_size,
            thinking=False,
        )

    rows = []
    for code, answers in texts.items():
        kept = [(i, a) for i, a in enumerate(answers) if a]
        judgments = score_steering(
            [(questions[i], a) for i, a in kept],
            "answer_quality",
            batch_size=args.judge_batch_size,
        )
        for (i, text), judgment in zip(kept, judgments, strict=True):
            rows.append(
                {
                    "index": i,
                    "language": code,
                    "question": questions[i],
                    "response": text,
                    "content_quality": judgment.content_quality if judgment else None,
                    "flags": list(judgment.flags) if judgment else [],
                    "reason": judgment.reason if judgment else None,
                }
            )
    write_jsonl(args.output / "rows.jsonl", rows)

    quality = {
        (row["language"], row["index"]): row["content_quality"]
        for row in rows
        if row["content_quality"] is not None
    }
    summary, drops = [], []
    for code in texts:
        paired = [
            (quality[(code, i)], quality[("en", i)])
            for i in range(len(questions))
            if (code, i) in quality and ("en", i) in quality
        ]
        differences = [mine - base for mine, base in paired]
        low, high = bootstrap_mean(differences, args.seed)
        summary.append(
            f"<tr><td>{code}</td><td>{len(paired)}</td>"
            f"<td>{sum(m for m, _ in paired) / len(paired):.2f}</td>"
            f"<td>{sum(differences) / len(differences):+.2f}</td><td>{low:+.2f}…{high:+.2f}</td>"
            f"<td>{sum(d == 0 for d in differences) / len(differences):.2f}</td>"
            f"<td>{sum(m == 0 for m, _ in paired) / len(paired):.2f}</td></tr>"
        )
        if code != "en":
            drops += [
                (quality[(code, i)] - quality[("en", i)], code, i)
                for i in range(len(questions))
                if (code, i) in quality and ("en", i) in quality
            ]
    by_key = {(row["language"], row["index"]): row for row in rows}
    examples = "".join(
        f"<tr><td>{code}</td><td>{difference:+d}</td><td style='text-align:left'>"
        f"{escape(questions[i])}</td><td style='text-align:left'>"
        f"{escape(by_key[(code, i)]['response'])}</td><td style='text-align:left'>"
        f"{escape(by_key[(code, i)]['reason'] or '')}<br><i>en: "
        f"{escape(by_key[('en', i)]['reason'] or '')}</i></td></tr>"
        for difference, code, i in sorted(drops)[:25]
    )
    write_page(
        args.output / "report.html",
        "answer_quality judge: same answer, different language",
        [
            "<h2>Quality per language, paired with the English original</h2><table><tr>"
            "<th>language</th><th>n</th><th>mean quality</th><th>mean diff vs en</th>"
            "<th>95% CI</th><th>tie share</th><th>quality 0 share</th></tr>"
            f"{''.join(summary)}</table>",
            "<h2>Largest drops</h2><table><tr><th>lang</th><th>diff</th><th>question</th>"
            f"<th>translation</th><th>judge reason (en reason)</th></tr>{examples}</table>",
        ],
    )
    print(f"wrote {args.output / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
