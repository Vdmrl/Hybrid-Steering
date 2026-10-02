"""Weight-level induction circuit diagnostics: composition, copying, and the
full induction QK circuit, for a trained two-layer checkpoint.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from model import load_checkpoint
from train import PackedTokenDataset


def comp_score(a: torch.Tensor, b: torch.Tensor) -> float:
    """Frobenius-normalized composition score between two d_model x d_model maps."""
    return ((a @ b).norm() / (a.norm() * b.norm())).item()


def composition_baseline(
    d_model: int, d_head: int, trials: int, scale_a: float, scale_b: float, generator: torch.Generator
) -> float:
    """Mean comp_score over random rank-``d_head`` matrices at matched scale."""
    scores = []
    for _ in range(trials):
        a = torch.randn(d_model, d_head, generator=generator) @ torch.randn(
            d_head, d_model, generator=generator
        ) * scale_a
        b = torch.randn(d_model, d_head, generator=generator) @ torch.randn(
            d_head, d_model, generator=generator
        ) * scale_b
        scores.append(comp_score(a, b))
    return sum(scores) / trials


@torch.inference_mode()
def composition_matrices(model, *, trials: int = 20, seed: int = 0) -> dict:
    """Q/K/V composition scores (layer-0 head i -> layer-1 head j), baseline-subtracted."""
    n_heads, d_model, d_head = model.cfg.n_heads, model.cfg.d_model, model.cfg.d_head
    w_ov0 = model.W_V[0] @ model.W_O[0]  # [head, d_model, d_model]
    w_qk1 = model.W_Q[1] @ model.W_K[1].transpose(-1, -2)
    w_ov1 = model.W_V[1] @ model.W_O[1]

    generator = torch.Generator().manual_seed(seed)
    scale_ov = w_ov0.norm(dim=(-2, -1)).mean().item() / (d_model * d_head) ** 0.5
    scale_qk = w_qk1.norm(dim=(-2, -1)).mean().item() / (d_model * d_head) ** 0.5

    q_comp = torch.zeros(n_heads, n_heads)
    k_comp = torch.zeros(n_heads, n_heads)
    v_comp = torch.zeros(n_heads, n_heads)
    for i in range(n_heads):
        for j in range(n_heads):
            q_comp[i, j] = comp_score(w_ov0[i], w_qk1[j])
            k_comp[i, j] = comp_score(w_qk1[j], w_ov0[i].T)
            v_comp[i, j] = comp_score(w_ov0[i], w_ov1[j])

    q_baseline = composition_baseline(d_model, d_head, trials, scale_ov, scale_qk, generator)
    k_baseline = composition_baseline(d_model, d_head, trials, scale_qk, scale_ov, generator)
    v_baseline = composition_baseline(d_model, d_head, trials, scale_ov, scale_ov, generator)
    return {
        "q_comp": (q_comp - q_baseline).tolist(),
        "k_comp": (k_comp - k_baseline).tolist(),
        "v_comp": (v_comp - v_baseline).tolist(),
        "q_baseline": q_baseline,
        "k_baseline": k_baseline,
        "v_baseline": v_baseline,
    }


@torch.inference_mode()
def copying_scores(model) -> list[list[float]]:
    """Exact OV copying score per head: sum(Re eig) / sum(|eig|) of the rank-d_head product."""
    scores = []
    for layer in range(model.cfg.n_layers):
        row = []
        for head in range(model.cfg.n_heads):
            m = (model.W_O[layer, head] @ model.W_U) @ (model.W_E @ model.W_V[layer, head])
            eigenvalues = torch.linalg.eigvals(m)
            row.append((eigenvalues.real.sum() / eigenvalues.abs().sum()).item())
        scores.append(row)
    return scores


@torch.inference_mode()
def induction_qk_circuit(
    model, prev_head: tuple[int, int], induction_head: tuple[int, int], candidates: torch.Tensor
) -> dict:
    """Full K-composition induction circuit restricted to frequent tokens.

    Returns the diagonal-dominance accuracy of ``W_E' W_QK1 W_OV0' W_E'^T``: query
    token t should prefer keys whose previous token is t.
    """
    prev_layer, prev_idx = prev_head
    ind_layer, ind_idx = induction_head
    w_ov0 = model.W_V[prev_layer, prev_idx] @ model.W_O[prev_layer, prev_idx]
    w_qk1 = model.W_Q[ind_layer, ind_idx] @ model.W_K[ind_layer, ind_idx].T
    e_sub = model.W_E[candidates]
    m = e_sub @ w_qk1 @ w_ov0.T @ e_sub.T
    ranks = m.argsort(dim=-1, descending=True)
    targets = torch.arange(len(candidates))
    top1 = (ranks[:, 0] == targets).float().mean().item()
    top5 = (ranks[:, :5] == targets[:, None]).any(dim=-1).float().mean().item()
    return {"top1_diagonal_accuracy": top1, "top5_diagonal_accuracy": top5}


def frequent_candidates(tokenizer, sequences, top_n: int) -> list[int]:
    frequency = Counter()
    for sequence in sequences:
        frequency.update(sequence.tolist())
    special = set(tokenizer.all_special_ids)
    return [token for token, _ in frequency.most_common() if token not in special][:top_n]


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# Two-layer induction circuit report",
        "",
        f"Checkpoint: `{report['checkpoint']}`",
        f"Selected heads: prev-token L{report['prev_head'][0]}H{report['prev_head'][1]}, "
        f"induction L{report['induction_head'][0]}H{report['induction_head'][1]}",
        "",
        f"K-composition (prev -> induction), baseline-subtracted: "
        f"{report['composition']['k_comp'][report['prev_head'][1]][report['induction_head'][1]]:.4f} "
        f"(baseline {report['composition']['k_baseline']:.4f})",
        f"OV copying score (prev head): {report['copying'][report['prev_head'][0]][report['prev_head'][1]]:.4f}",
        f"OV copying score (induction head): "
        f"{report['copying'][report['induction_head'][0]][report['induction_head'][1]]:.4f}",
        f"Induction QK circuit top-1 diagonal accuracy: {report['induction_circuit']['top1_diagonal_accuracy']:.4f}",
        f"Induction QK circuit top-5 diagonal accuracy: {report['induction_circuit']['top5_diagonal_accuracy']:.4f}",
        "",
        "## Interpretation limits",
        "",
        "- LayerNorm is omitted from the OV/QK products; it cannot be folded exactly at layer 1 the",
        "  way it can at layer 0, so this is the standard linear approximation.",
        "- Shortformer positional terms are dropped from the QK analysis.",
        "- Composition baselines are random rank-matched matrices at the real weights' scale, not a",
        "  causal ablation.",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--candidate-vocab", type=int, default=2000)
    parser.add_argument("--baseline-trials", type=int, default=20)
    parser.add_argument("--prev-head", type=str, help="layer,head override")
    parser.add_argument("--induction-head", type=str, help="layer,head override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer, config, checkpoint = load_checkpoint(args.run_dir, args.checkpoint, args.device)
    if any(layer != "attn" for layer in config["model"]["layer_types"]):
        raise ValueError("circuit_lens.py reads QK/OV attention weights; it does not support GDN layers")

    if args.prev_head and args.induction_head:
        prev_head = tuple(int(x) for x in args.prev_head.split(","))
        induction_head = tuple(int(x) for x in args.induction_head.split(","))
    else:
        induction_report = json.loads((args.run_dir / "reports" / "final" / "induction.json").read_text())
        prev_row = max(induction_report["heads"], key=lambda row: row["prev_token"])
        induction_row = max(induction_report["heads"], key=lambda row: row["induction"])
        prev_head = (prev_row["layer"], prev_row["head"])
        induction_head = (induction_row["layer"], induction_row["head"])

    data = config["data"]
    sequences = list(
        PackedTokenDataset(
            files=[Path(path) for path in data["validation_files"]],
            tokenizer=tokenizer,
            sequence_length=config["model"]["n_ctx"],
            sequences=32,
            seed=config["training"]["seed"],
        )
    )
    candidates = torch.tensor(frequent_candidates(tokenizer, sequences, args.candidate_vocab))

    report = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "model": config["model"],
        "prev_head": prev_head,
        "induction_head": induction_head,
        "composition": composition_matrices(model, trials=args.baseline_trials),
        "copying": copying_scores(model),
        "induction_circuit": induction_qk_circuit(model, prev_head, induction_head, candidates),
    }

    output_dir = (args.output_dir or args.run_dir / "reports").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "circuits.json").write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(output_dir / "circuits.md", report)
    print(f"wrote {output_dir / 'circuits.md'}")


if __name__ == "__main__":
    main()
