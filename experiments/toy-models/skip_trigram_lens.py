"""Find skip-trigram circuits in a trained one-layer attention-only model."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import torch

from model import load_checkpoint
from train import PackedTokenDataset


def token_label(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([token_id]).replace("\n", "\\n")
    return repr(text)


def corpus_support(sequences, rows: list[dict], window: int) -> Counter:
    """Count sampled occurrences of ``source ... destination output``."""
    wanted: dict[tuple[int, int], set[int]] = defaultdict(set)
    for row in rows:
        wanted[(row["destination_id"], row["output_id"])].add(row["source_id"])
    counts = Counter()
    for sequence in sequences:
        values = sequence.tolist()
        for index in range(1, len(values) - 1):
            destination, output = values[index], values[index + 1]
            sources = wanted.get((destination, output))
            if sources:
                for source in sources.intersection(values[max(0, index - window) : index]):
                    counts[(source, destination, output)] += 1
    return counts


@torch.inference_mode()
def find_skip_trigrams(
    model,
    tokenizer,
    sequences,
    *,
    candidate_vocab: int,
    pairs_per_head: int,
    outputs_per_pair: int,
    skip_window: int,
) -> tuple[list[dict], list[int]]:
    """Return a compact, source-pivoted QK/OV head dump.

    This follows the selection heuristic from the framework paper, restricted
    to frequent tokens from the sampled validation corpus.  The old global-QK
    ranking mostly surfaced isolated, high-norm token pairs.
    """
    frequency = Counter(torch.cat(sequences).tolist())
    special = set(tokenizer.all_special_ids)
    candidates = [
        token
        for token, _ in frequency.most_common()
        if token not in special and "�" not in tokenizer.decode([token])
    ][:candidate_vocab]
    if len(candidates) < 2:
        raise ValueError("not enough candidate tokens in the sampled corpus")

    device = model.W_E.device
    candidate_ids = torch.tensor(candidates, device=device)
    probabilities = torch.tensor(
        [frequency[token] / sum(frequency.values()) for token in candidates],
        device=device,
    )
    # The first layer's LayerNorm can be folded into each token embedding.
    normalized_embeddings = model.blocks[0].ln1(model.W_E)
    candidate_embeddings = normalized_embeddings[candidate_ids]
    attention = model.blocks[0].attn
    default_embedding = normalized_embeddings[tokenizer.eos_token_id]
    rows = []
    for head in range(model.cfg.n_heads):
        queries = candidate_embeddings @ attention.W_Q[head]
        keys = candidate_embeddings @ attention.W_K[head]
        scale = math.sqrt(model.cfg.d_head)
        # QK scores are only meaningful up to a query-specific constant.  As
        # in the paper, use end-of-text as that query's reference key.
        qk = (queries @ keys.T - queries @ (default_embedding @ attention.W_K[head])) / scale
        ov_results = candidate_embeddings @ attention.W_V[head] @ attention.W_O[head]
        ov_logits = ov_results @ model.W_U
        ov_logits -= ov_logits.mean(dim=1, keepdim=True)
        key_scores = qk.max(dim=0).values * ov_logits.max(dim=1).values * probabilities.pow(0.1)
        key_count = min(pairs_per_head, len(candidates))
        _, source_indices = key_scores.topk(key_count)
        for source_index in source_indices.tolist():
            source = candidates[source_index]
            query_scores = qk[:, source_index] * probabilities.pow(0.1)
            query_count = min(pairs_per_head, len(candidates))
            query_scores, destination_indices = query_scores.topk(query_count)
            output_logits, output_ids = ov_logits[source_index].topk(outputs_per_pair)
            for destination_index, pair_score in zip(destination_indices.tolist(), query_scores, strict=True):
                destination = candidates[destination_index]
                direct_logits = normalized_embeddings[destination] @ model.W_U
                for output_logit, output in zip(output_logits, output_ids, strict=True):
                    output = int(output)
                    rows.append(
                        {
                            "head": head,
                            "source_id": source,
                            "source": token_label(tokenizer, source),
                            "destination_id": destination,
                            "destination": token_label(tokenizer, destination),
                            "output_id": output,
                            "output": token_label(tokenizer, output),
                            "qk_score": float(pair_score),
                            "ov_logit_delta": float(output_logit),
                            "key_score": float(key_scores[source_index]),
                            "direct_logit": float(direct_logits[output]),
                            "direct_plus_ov": float(
                                direct_logits[output] + output_logit
                            ),
                        }
                    )

    support = corpus_support(sequences, rows, skip_window)
    for row in rows:
        row["sample_occurrences"] = support[
            (row["source_id"], row["destination_id"], row["output_id"])
        ]
    return rows, candidates


def write_markdown(path: Path, report: dict, rows_per_head: int) -> None:
    lines = [
        "# One-layer skip-trigram circuit report",
        "",
        f"Checkpoint: `{report['checkpoint']}`",
        "",
        (
            "Each row reads `[source] … [query] → [output]`. Keys are selected "
            "by `max(QK) × max(OV) × token_probability^0.1`; QK is normalized "
            "against the end-of-text key and OV logits are mean-centered."
        ),
        "",
    ]
    for head in range(report["model"]["n_heads"]):
        head_rows = [row for row in report["rows"] if row["head"] == head]
        keys = {}
        for row in head_rows:
            key = keys.setdefault(
                row["source_id"],
                {"source": row["source"], "score": row["key_score"], "queries": {}, "outputs": {}},
            )
            key["queries"][row["destination"]] = row["qk_score"]
            key["outputs"][row["output"]] = row["ov_logit_delta"]
        ranked_keys = sorted(keys.values(), key=lambda key: key["score"], reverse=True)
        lines += [
            f"## Head {head}",
            "",
            "| Key | Queries that prefer key (normalized QK) | Effect on logits (centered OV) | Score |",
            "| --- | --- | --- | ---: |",
        ]
        for key in ranked_keys[:rows_per_head]:
            queries = "<br>".join(
                f"`{token}` ({score:.2f})"
                for token, score in sorted(key["queries"].items(), key=lambda item: item[1], reverse=True)
            )
            outputs = "<br>".join(
                f"`{token}` ({score:.2f})"
                for token, score in sorted(key["outputs"].items(), key=lambda item: item[1], reverse=True)
            )
            lines.append(
                f"| `{key['source']}` | {queries} | {outputs} | {key['score']:.3f} |"
            )
        lines.append("")
    lines += [
        "## Interpretation limits",
        "",
        "- This is the paper's content-only QK/OV path expansion; positional terms are omitted.",
        "- Logit contributions are shown before the model's nonlinear final LayerNorm.",
        "- Sample counts validate corpus support but are not a causal-ablation result.",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a QK/OV skip-trigram report from a trained checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-sequences", type=int, default=128)
    parser.add_argument("--candidate-vocab", type=int, default=512)
    parser.add_argument("--pairs-per-head", type=int, default=8, help="source keys and queries shown per head")
    parser.add_argument("--outputs-per-pair", type=int, default=8)
    parser.add_argument("--rows-per-head", type=int, default=20)
    parser.add_argument("--skip-window", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer, config, checkpoint = load_checkpoint(
        args.run_dir, args.checkpoint, args.device
    )
    data = config["data"]
    sample_data = PackedTokenDataset(
        files=[Path(path) for path in data["validation_files"]],
        tokenizer=tokenizer,
        sequence_length=config["model"]["n_ctx"],
        sequences=args.sample_sequences,
        seed=config["training"]["seed"],
    )
    sequences = list(sample_data)
    rows, candidates = find_skip_trigrams(
        model,
        tokenizer,
        sequences,
        candidate_vocab=args.candidate_vocab,
        pairs_per_head=args.pairs_per_head,
        outputs_per_pair=args.outputs_per_pair,
        skip_window=args.skip_window,
    )
    output_dir = (args.output_dir or args.run_dir / "reports").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "model": config["model"],
        "sample_sequences": args.sample_sequences,
        "candidate_token_ids": candidates,
        "skip_window": args.skip_window,
        "supported_rows": sum(row["sample_occurrences"] > 0 for row in rows),
        "rows": rows,
    }
    (output_dir / "skip_trigrams.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    write_markdown(output_dir / "skip_trigrams.md", report, args.rows_per_head)
    print(f"wrote {output_dir / 'skip_trigrams.md'}")


if __name__ == "__main__":
    main()
