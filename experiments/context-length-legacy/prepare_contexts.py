"""Make exact-token neutral long-context prompts for the S_0 experiment."""

import argparse
import json
import random
import shutil
from itertools import pairwise
from pathlib import Path

import torch
from transformers import AutoTokenizer

DEFAULT_CONTEXT_LENGTHS = (128, 256, 512, 1024, 2048, 4096, 8192)
DEFAULT_SOURCE_ATTEMPTS = 32


def token_ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def chat_shell(tokenizer) -> tuple[list[int], list[int]]:
    """Split Qwen's chat template immediately around user content."""
    marker = "CODExNestedContextMarker"
    marker_ids = token_ids(tokenizer, marker)
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )["input_ids"]
    starts = [
        index
        for index in range(len(full) - len(marker_ids) + 1)
        if full[index : index + len(marker_ids)] == marker_ids
    ]
    if len(starts) != 1:
        raise RuntimeError("could not locate user content in the chat template")
    start = starts[0]
    return full[:start], full[start + len(marker_ids) :]


def tweet_only_prompt(tokenizer, shell, tweet: str) -> list[int]:
    head, tail = shell
    return head + token_ids(tokenizer, f"Tweet: {tweet}\n\nReply:") + tail


def nested_prompts(
    tokenizer,
    shell,
    source_ids,
    task,
    lengths,
    offset,
    rng,
    source_attempts,
):
    """Build exact prompts from nested token prefixes of one source span."""
    ids = source_ids.tolist()
    chat_head, chat_tail = shell
    content_head = token_ids(tokenizer, "Reference material (use only if helpful):\n")
    content_tail = token_ids(tokenizer, f"\n\nTweet: {task}\n\nReply:")
    fixed = len(chat_head) + len(content_head) + len(content_tail) + len(chat_tail)
    filler_lengths = [target - fixed for target in lengths]
    if min(filler_lengths) < 1:
        raise ValueError(f"shortest target is below the {fixed + 1}-token minimum")
    span_tokens = max(filler_lengths)
    last_start = len(ids) - span_tokens
    starts = list(
        dict.fromkeys(
            [
                offset % (last_start + 1),
                *rng.sample(
                    range(last_start + 1),
                    k=min(source_attempts - 1, last_start + 1),
                ),
            ]
        )
    )
    for start in starts:
        filler_ids = [ids[start : start + size] for size in filler_lengths]
        fillers = [
            tokenizer.decode(part, skip_special_tokens=True) for part in filler_ids
        ]
        if not all(
            longer.startswith(shorter) for shorter, longer in pairwise(fillers)
        ):
            continue
        prompts = [
            chat_head + content_head + part + content_tail + chat_tail
            for part in filler_ids
        ]
        if all(
            len(prompt) == target
            for prompt, target in zip(prompts, lengths, strict=True)
        ):
            return (
                list(zip(fillers, filler_ids, prompts, strict=True)),
                start,
                span_tokens,
            )
    raise RuntimeError("could not build exact nested neutral prompts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("../shared/twitter_pessimistic_tweets.jsonl"),
        help="JSONL tweets used as the fixed prompt suffixes",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, help="tweets to use; default is all rows")
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=DEFAULT_CONTEXT_LENGTHS
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source-attempts",
        type=int,
        default=DEFAULT_SOURCE_ATTEMPTS,
        help="candidate offsets tried per source when fitting exact nested prompts",
    )
    args = parser.parse_args()
    if args.source_attempts < 1:
        parser.error("--source-attempts must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    shell = chat_shell(tokenizer)
    tweets = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    if not tweets or not all(row.get("text") for row in tweets):
        raise ValueError(f"{args.dataset} must contain non-empty text fields")
    count = args.count or len(tweets)
    if count > len(tweets):
        raise ValueError(f"requested {count} tweets but {args.dataset} has {len(tweets)}")
    tweets = tweets[:count]
    records = torch.load(args.sources, weights_only=False)["records"]
    sources = [x for x in records if x["data_type"] == "coherent"]
    rng = random.Random(args.seed)
    output = args.run_dir / "contexts.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "contexts.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "count": count,
                "lengths": args.lengths,
                "seed": args.seed,
                "source_attempts": args.source_attempts,
                "dataset": str(args.dataset),
            },
            indent=2,
        )
        + "\n"
    )
    source_copy = args.run_dir / "inputs" / "source_sequences.pt"
    source_copy.parent.mkdir(exist_ok=True)
    if args.sources.resolve() != source_copy.resolve():
        shutil.copy2(args.sources, source_copy)
    source_metadata = args.run_dir / "source_metadata.jsonl"
    lengths = sorted(args.lengths)
    saved = []
    used_sources = set()
    with output.open("w") as out, source_metadata.open("w") as metadata:
        for example_id, tweet in enumerate(tweets):
            task = tweet["text"]
            for source_step in range(len(sources)):
                source_index = (
                    example_id * len(sources) // count + source_step
                ) % len(sources)
                source = sources[source_index]
                if source["text_id"] in used_sources:
                    continue
                initial_offset = rng.randrange(len(source["input_ids"]))
                try:
                    fitted, offset, span_tokens = nested_prompts(
                        tokenizer,
                        shell,
                        source["input_ids"],
                        task,
                        lengths,
                        initial_offset,
                        rng,
                        args.source_attempts,
                    )
                    used_sources.add(source["text_id"])
                    break
                except RuntimeError:
                    continue
            else:
                raise RuntimeError(f"no neutral source span for example {example_id}")
            source_info = {
                "example_id": example_id,
                "source_id": source["text_id"],
                "source_title": source["title"],
                "source_offset_tokens": offset,
                "source_span_tokens": span_tokens,
                "task": task,
                "tweet_id": tweet.get("id", example_id),
            }
            metadata.write(json.dumps(source_info) + "\n")
            tweet_prompt = tweet_only_prompt(tokenizer, shell, task)
            saved.append(
                {
                    **source_info,
                    "target_tokens": 0,
                    "input_tokens": len(tweet_prompt),
                    "tweet_only": True,
                    "filler": "",
                    "filler_input_ids": [],
                    "prompt_input_ids": tweet_prompt,
                }
            )
            out.write(json.dumps(saved[-1]) + "\n")
            for target, (filler, filler_ids, prompt) in zip(
                lengths, fitted, strict=True
            ):
                record = {
                    **source_info,
                    "target_tokens": target,
                    "input_tokens": len(prompt),
                    "tweet_only": False,
                    "filler": filler,
                    "filler_input_ids": filler_ids,
                    "prompt_input_ids": prompt,
                }
                saved.append(record)
                out.write(json.dumps(record) + "\n")
    groups = [
        [row for row in saved if row["example_id"] == example_id]
        for example_id in range(count)
    ]
    checks = {
        "row_count": len(saved),
        "exact_token_lengths": all(
            row["target_tokens"] == 0
            or len(row["prompt_input_ids"]) == row["target_tokens"] == row["input_tokens"]
            for row in saved
        ),
        "nested_filler_prefixes": all(
            longer["filler"].startswith(shorter["filler"])
            and longer["filler_input_ids"][: len(shorter["filler_input_ids"])]
            == shorter["filler_input_ids"]
            for group in groups
            for shorter, longer in pairwise(group)
        ),
        "shared_source_offset_task": all(
            len(
                {
                    (
                        row["source_id"],
                        row["source_offset_tokens"],
                        row["task"],
                    )
                    for row in group
                }
            )
            == 1
            for group in groups
        ),
        "unique_source_per_example": len(used_sources) == count,
    }
    if (
        not all(
            checks[key]
            for key in (
                "exact_token_lengths",
                "nested_filler_prefixes",
                "shared_source_offset_task",
                "unique_source_per_example",
            )
        )
    ):
        raise AssertionError(f"context validation failed: {checks}")
    (args.run_dir / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(
        f"wrote {count * (len(lengths) + 1)} prompts to {output}; "
        f"source metadata is in {source_metadata}"
    )


if __name__ == "__main__":
    main()
