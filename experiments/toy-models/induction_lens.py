"""Behavioral and causal induction-head diagnostics for a trained checkpoint."""

import argparse
import json
from pathlib import Path

import torch

from model import load_checkpoint
from train import PackedTokenDataset


@torch.inference_mode()
def repeated_token_batch(
    tokenizer, sequences, *, seq_len: int, batch: int, seed: int
) -> torch.Tensor:
    """Build ``[eos] + rand(n) + rand(n)`` probes from frequent corpus tokens."""
    frequency: dict[int, int] = {}
    for sequence in sequences:
        for token in sequence.tolist():
            frequency[token] = frequency.get(token, 0) + 1
    special = set(tokenizer.all_special_ids)
    vocab = torch.tensor(
        sorted(
            (token for token in frequency if token not in special),
            key=frequency.get,
            reverse=True,
        )[:10_000]
    )
    generator = torch.Generator().manual_seed(seed)
    half = vocab[torch.randint(len(vocab), (batch, seq_len), generator=generator)]
    eos = torch.full((batch, 1), tokenizer.eos_token_id, dtype=torch.long)
    return torch.cat([eos, half, half], dim=1)


def induction_indices(seq_len: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Query/key position pairs for induction, duplicate, and previous-token scores.

    Position layout: 0 = EOS, 1..n = first copy, n+1..2n = second copy.
    """
    n = seq_len
    second_half = torch.arange(n + 1, 2 * n + 1)
    all_but_first = torch.arange(1, 2 * n + 1)
    return {
        "induction": (second_half, second_half - (n - 1)),
        "duplicate": (second_half, second_half - n),
        "prev_token": (all_but_first, all_but_first - 1),
    }


@torch.inference_mode()
def head_scores(pattern: torch.Tensor, seq_len: int) -> dict[str, float]:
    """Mean attention paid from each query row to its designated key column.

    ``pattern`` is ``[batch, query, key]`` for one head, averaged over batch.
    """
    per_head = {}
    mean_pattern = pattern.mean(dim=0)
    for name, (queries, keys) in induction_indices(seq_len).items():
        per_head[name] = mean_pattern[queries, keys].mean().item()
    return per_head


@torch.inference_mode()
def icl_score(model, sequences: list[torch.Tensor], *, early: int = 50, late: int = 500) -> float:
    """Mean(loss at ``late``) - mean(loss at ``early``) over held-out sequences."""
    deltas = []
    for sequence in sequences:
        losses = model(sequence.unsqueeze(0), return_type="loss", loss_per_token=True)[0]
        deltas.append((losses[late - 1] - losses[early - 1]).item())
    return sum(deltas) / len(deltas)


@torch.inference_mode()
def ablate_head_loss(model, tokens: torch.Tensor, layer: int, head: int) -> torch.Tensor:
    """Per-token loss with one head's ``z`` zeroed at every position."""

    def zero_head(z, hook):
        z[:, :, head, :] = 0.0
        return z

    return model.run_with_hooks(
        tokens,
        fwd_hooks=[(f"blocks.{layer}.attn.hook_z", zero_head)],
        return_type="loss",
        loss_per_token=True,
    )


@torch.inference_mode()
def ablated_icl_score(
    model, sequences: list[torch.Tensor], layer: int, head: int, *, early: int = 50, late: int = 500
) -> float:
    deltas = []
    for sequence in sequences:
        losses = ablate_head_loss(model, sequence.unsqueeze(0), layer, head)[0]
        deltas.append((losses[late - 1] - losses[early - 1]).item())
    return sum(deltas) / len(deltas)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--icl-sequences", type=int, default=128)
    parser.add_argument("--ablate-sequences", type=int, default=8)
    parser.add_argument("--dump-patterns", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer, config, checkpoint = load_checkpoint(args.run_dir, args.checkpoint, args.device)
    if any(layer != "attn" for layer in config["model"]["layer_types"]):
        raise ValueError("induction_lens.py reads attention patterns; it does not support GDN layers")
    model.requires_grad_(False)
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    data = config["data"]

    frequency_sample = list(
        PackedTokenDataset(
            files=[Path(path) for path in data["validation_files"]],
            tokenizer=tokenizer,
            sequence_length=config["model"]["n_ctx"],
            sequences=32,
            seed=config["training"]["seed"],
        )
    )
    probe = repeated_token_batch(
        tokenizer,
        frequency_sample,
        seq_len=args.seq_len,
        batch=args.batch,
        seed=args.probe_seed,
    ).to(args.device)

    icl_sample = list(
        PackedTokenDataset(
            files=[Path(path) for path in data["validation_files"]],
            tokenizer=tokenizer,
            sequence_length=config["model"]["n_ctx"],
            sequences=args.icl_sequences,
            seed=config["training"]["seed"],
        )
    )
    icl_sample = [sequence.to(args.device) for sequence in icl_sample]

    _, cache = model.run_with_cache(probe)
    per_token_loss = model(probe, return_type="loss", loss_per_token=True).mean(dim=0)
    valid_loss = model(torch.stack(icl_sample[: args.ablate_sequences]), return_type="loss").item()
    score = icl_score(model, icl_sample)

    ablate_icl_sample = icl_sample[: args.ablate_sequences]
    baseline_icl = icl_score(model, ablate_icl_sample)
    baseline_second_half = per_token_loss[args.seq_len :].mean().item()

    heads = []
    for layer in range(n_layers):
        for head in range(n_heads):
            pattern = cache[f"blocks.{layer}.attn.hook_pattern"][:, head]
            scores = head_scores(pattern, args.seq_len)

            ablated_second_half = (
                ablate_head_loss(model, probe, layer, head).mean(dim=0)[args.seq_len :].mean().item()
            )
            with_ablation = ablated_icl_score(model, ablate_icl_sample, layer, head)

            heads.append(
                {
                    "layer": layer,
                    "head": head,
                    **scores,
                    "ablation_delta_repeat_loss": ablated_second_half - baseline_second_half,
                    "ablation_delta_icl": with_ablation - baseline_icl,
                }
            )

    result = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "step": _step_from_name(checkpoint),
        "model": config["model"],
        "valid_loss": valid_loss,
        "icl_score": score,
        "per_token_loss_repeated": per_token_loss.tolist(),
        "heads": heads,
    }
    if args.dump_patterns:
        top_heads = sorted(heads, key=lambda row: row["induction"], reverse=True)[:6]
        result["dumped_patterns"] = {
            "tokens": probe[0].tolist(),
            "heads": [
                {
                    "layer": row["layer"],
                    "head": row["head"],
                    "pattern": cache[f"blocks.{row['layer']}.attn.hook_pattern"][0, row["head"]].tolist(),
                }
                for row in top_heads
            ],
        }

    output_dir = (args.output_dir or args.run_dir / "reports" / "final").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "induction.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {output_dir / 'induction.json'}")


def _step_from_name(checkpoint) -> int | None:
    name = Path(checkpoint).stem
    if name.startswith("step_"):
        return int(name.removeprefix("step_"))
    return None


if __name__ == "__main__":
    main()
