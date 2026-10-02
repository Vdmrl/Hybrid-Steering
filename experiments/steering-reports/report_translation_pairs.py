"""Render the translation pairs used to collect a steering direction."""

import argparse
from html import escape
from pathlib import Path

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from experiments.steering.collect_acts import MODEL, translation_pairs


def token_counts(
    tokenizer: PreTrainedTokenizerBase, pairs: list[tuple[str, str]], batch_size: int
) -> list[tuple[int, int, int, int]]:
    """Return unpadded and batch-padded token counts for every translation pair."""
    counts = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        english = tokenizer([source for source, _ in batch], padding=True, return_tensors="pt")
        russian = tokenizer([target for _, target in batch], padding=True, return_tensors="pt")
        counts.extend(
            zip(
                english.attention_mask.sum(-1).tolist(),
                [english.input_ids.shape[1]] * len(batch),
                russian.attention_mask.sum(-1).tolist(),
                [russian.input_ids.shape[1]] * len(batch),
                strict=True,
            )
        )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Render OPUS-100 translation pairs used for GDN direction collection.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language-a", default="en")
    parser.add_argument("--language-b", default="ru")
    parser.add_argument("--pairs", type=int, default=500)
    parser.add_argument("--min-words", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--strict-script", action="store_true")
    parser.add_argument("--reject-digit-only-and-uppercase", action="store_true")
    parser.add_argument("--max-relative-token-difference", type=float)
    args = parser.parse_args()
    if min(args.pairs, args.min_words, args.batch_size) < 1:
        parser.error("--pairs, --min-words, and --batch-size must be positive")
    if args.max_relative_token_difference is not None and args.max_relative_token_difference < 0:
        parser.error("--max-relative-token-difference must be non-negative")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
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
    counts = token_counts(tokenizer, pairs, args.batch_size)
    rows = "".join(
        "<tr>"
        f"<td>{index}</td><td>{english_tokens}</td><td>{english_padded}</td><td>{russian_tokens}</td><td>{russian_padded}</td><td>{abs(english_tokens - russian_tokens) / max(english_tokens, russian_tokens):.1%}</td><td>{escape(english)}</td>"
        f"<td>{escape(russian)}</td>"
        "</tr>"
        for index, ((english, russian), (english_tokens, english_padded, russian_tokens, russian_padded)) in enumerate(
            zip(pairs, counts, strict=True), start=1
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "<!doctype html><meta charset=utf-8><title>EN→RU GDN direction inputs</title>"
        "<style>body{font:15px system-ui;max-width:1800px;margin:auto;padding:24px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #ddd;text-align:left;vertical-align:top}td:nth-child(4),td:nth-child(7){white-space:pre-wrap}</style>"
        "<h1>Тексты для направления EN→RU</h1>"
        f"<p>Модель: {escape(MODEL)}. Выбрано {len(pairs)} пар из OPUS-100 ({escape(args.language_a)}→{escape(args.language_b)}), split <code>train</code>, конфигурация <code>{escape('-'.join(sorted((args.language_a, args.language_b))))}</code>, затем <code>shuffle(seed=42)</code>.</p>"
        f"<p>Для каждой записи берутся поля перевода, применяется только <code>.strip()</code>, а пары с числом слов ≤ {args.min_words} хотя бы с одной стороны отбрасываются. Strict script: {args.strict_script}. Reject digit-only and uppercase: {args.reject_digit_only_and_uppercase}. Максимальная относительная разница токенов: {args.max_relative_token_difference}. Ни chat template, ни инструкций, ни другой текстовой нормализации при сборе состояния нет. В таблице показаны именно строки после <code>.strip()</code>. Токенизация выполняется Qwen tokenizer; <em>tokens</em> — длина без padding, <em>padded</em> — общая длина соответствующей стороны в batch из {args.batch_size} пар.</p>"
        "<h2>Как вычисляется направление</h2>"
        "<p>Для каждой пары обе строки независимо прогоняются через Qwen без steering. Для каждого GDN-слоя берётся final recurrent state после последнего непаддингового токена. Затем для каждого слоя вычисляется среднее по всем русским состояниям и среднее по всем английским состояниям:</p>"
        "<p><code>Δₗ = mean(Sₗ(RU)) − mean(Sₗ(EN))</code>.</p>"
        "<p>Полученный <code>Δₗ</code> сохраняется для каждого слоя. При steering он при необходимости аппроксимируется SVD заданного rank, масштабируется коэффициентом и добавляется к recurrent state. При включённой normalisation его Frobenius-норма для каждого head приводится к норме текущего state перед умножением на scale.</p>"
        "<h2>Все использованные пары</h2>"
        "<table><tr><th>#</th><th>EN tokens</th><th>EN padded</th><th>RU tokens</th><th>RU padded</th><th>Difference</th><th>English text after strip</th><th>Russian text after strip</th></tr>"
        + rows
        + "</table>",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
