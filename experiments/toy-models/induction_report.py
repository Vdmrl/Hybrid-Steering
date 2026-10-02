"""Render the two-layer induction-head experiment as a self-contained HTML explainer."""

import argparse
import json
import math
from pathlib import Path

from fasthtml.common import Div, H1, H2, P, Span, Strong, Style, Titled
from research_viz import (
    Chart,
    CircuitsVis,
    DataTable,
    Panel,
    ScrollZoomToggle,
    heatmap_option,
    line_option,
    render,
)

from train import cosine_schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def stat(value: str, label: str, detail: str):
    return Div(
        Span(value, cls="toy-stat-value"),
        Strong(label, cls="toy-stat-label"),
        Span(detail, cls="toy-stat-detail"),
        cls="toy-stat",
    )


def load_sweep(run_dir: Path) -> list[dict]:
    steps = sorted(int(p.name.removeprefix("step_")) for p in (run_dir / "reports").glob("step_*"))
    reports = []
    for step in steps:
        path = run_dir / "reports" / f"step_{step}" / "induction.json"
        if path.exists():
            reports.append(json.loads(path.read_text()))
    return reports


def head_matrix(reports: list[dict], score: str) -> tuple[list[str], list[str], list[list[float]]]:
    if not reports:
        return [], [], []
    n_layers = len(reports[0]["model"]["layer_types"])
    n_heads = reports[0]["model"]["n_heads"]
    labels = [f"L{layer}H{head}" for layer in range(n_layers) for head in range(n_heads)]
    steps = [str(r["step"]) for r in reports]
    values = [[0.0] * len(reports) for _ in labels]
    for j, report in enumerate(reports):
        for row in report["heads"]:
            index = row["layer"] * n_heads + row["head"]
            values[index][j] = row[score]
    return labels, steps, values


def load_data(run_dir: Path) -> dict:
    from omegaconf import OmegaConf

    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    config = OmegaConf.to_container(OmegaConf.load(run_dir / "checkpoints" / "config.yaml"), resolve=True)
    sweep = load_sweep(run_dir)
    final_path = run_dir / "reports" / "final" / "induction.json"
    final = json.loads(final_path.read_text()) if final_path.exists() else (sweep[-1] if sweep else None)
    circuits_path = run_dir / "reports" / "circuits.json"
    circuits = json.loads(circuits_path.read_text()) if circuits_path.exists() else None
    return {
        "metrics": metrics,
        "config": config,
        "sweep": sweep,
        "final": final,
        "circuits": circuits,
    }


def index(data: dict):
    config = data["config"]
    training = config["training"]
    metrics = data["metrics"]
    sweep = data["sweep"]
    final = data["final"]
    circuits = data["circuits"]

    train_points = [
        (record["optimizer_step"], record["train_loss"])
        for record in metrics
        if record.get("optimizer_step", 0) and "train_loss" in record
    ]
    schedule = cosine_schedule(
        training["steps"], training["warmup_steps"], training["minimum_learning_rate"] / training["learning_rate"]
    )
    lr_points = [
        (record["optimizer_step"], training["learning_rate"] * schedule(record["optimizer_step"]))
        for record in metrics
        if record.get("optimizer_step", 0)
    ]

    icl_points = [(r["step"], r["icl_score"]) for r in sweep]
    induction_labels, induction_steps, induction_values = head_matrix(sweep, "induction")

    per_token_series = {}
    for report in sweep[:: max(1, len(sweep) // 4)] if sweep else []:
        loss = report["per_token_loss_repeated"]
        per_token_series[f"step {report['step']}"] = list(enumerate(loss))

    ablation_rows = []
    if final:
        for row in final["heads"]:
            ablation_rows.append(
                {
                    "id": f"L{row['layer']}H{row['head']}",
                    "head": f"L{row['layer']}H{row['head']}",
                    "induction": f"{row['induction']:.3f}",
                    "prev_token": f"{row['prev_token']:.3f}",
                    "delta_repeat_loss": f"{row['ablation_delta_repeat_loss']:+.3f}",
                    "delta_icl": f"{row['ablation_delta_icl']:+.3f}",
                }
            )

    verdicts = []
    if circuits:
        prev = f"L{circuits['prev_head'][0]}H{circuits['prev_head'][1]}"
        induction = f"L{circuits['induction_head'][0]}H{circuits['induction_head'][1]}"
        k = circuits["composition"]["k_comp"][circuits["prev_head"][1]][circuits["induction_head"][1]]
        copying_prev = circuits["copying"][circuits["prev_head"][0]][circuits["prev_head"][1]]
        copying_induction = circuits["copying"][circuits["induction_head"][0]][circuits["induction_head"][1]]
        top1 = circuits["induction_circuit"]["top1_diagonal_accuracy"]
        verdicts = [
            f"Previous-token head: {prev}",
            f"Induction head: {induction}",
            f"K-composition (baseline-subtracted): {k:.4f}",
            f"OV copying score — prev head: {copying_prev:.4f}, induction head: {copying_induction:.4f}",
            f"Induction QK circuit top-1 diagonal accuracy: {top1:.4f}",
        ]

    dumped = final.get("dumped_patterns") if final else None

    return Titled(
        "Two-layer induction circuits",
        Style(
            """
            .toy-hero { padding: 4rem 0 2.25rem; max-width: 800px; }
            .toy-kicker { color: #9c4a2b; font: 700 .75rem/1.2 ui-monospace, monospace; letter-spacing: .11em; text-transform: uppercase; }
            .toy-hero h1 { font-size: clamp(2.7rem, 7vw, 5.4rem); line-height: .94; letter-spacing: -.07em; margin: .45rem 0 1.25rem; }
            .toy-lede { color: var(--rv-muted); font-size: 1.18rem; line-height: 1.6; max-width: 700px; }
            .toy-stat-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .8rem; margin: 1.5rem 0 3rem; }
            .toy-stat { border-top: 2px solid #c15f3c; padding: .75rem 0; }
            .toy-stat-value { display:block; font: 700 1.7rem/1.1 ui-monospace, monospace; letter-spacing:-.08em; }
            .toy-stat-label { display:block; margin-top:.35rem; font-size:.82rem; }
            .toy-stat-detail { display:block; margin-top:.15rem; color:var(--rv-muted); font-size:.74rem; }
            .toy-verdict { padding: .5rem 0; border-bottom: 1px solid var(--rv-border); font: 600 .95rem ui-monospace, monospace; }
            @media(max-width:700px){.toy-stat-grid{grid-template-columns:repeat(2,minmax(0,1fr));}.toy-hero{padding-top:2.5rem;}}
            """
        ),
        Div(
            Span("Transformer Circuits · Toy Models", cls="toy-kicker"),
            H1("A two-layer model learns to look back further."),
            P(
                "A two-layer attention-only transformer, trained the same way as the one-layer "
                "model. This explainer follows in-context learning ability, per-head induction "
                "scores, and the weight-level composition circuit behind them.",
                cls="toy-lede",
            ),
            cls="toy-hero",
        ),
        Div(
            stat(f"{training['steps']:,}", "optimizer steps", f"{training['target_tokens']:,} predicted tokens"),
            stat(f"{len(config['model']['layer_types'])}", "layers", ",".join(config["model"]["layer_types"])),
            stat(f"{final['valid_loss']:.3f}" if final else "—", "held-out loss", f"ppl {math.exp(final['valid_loss']):.1f}" if final else ""),
            stat(f"{len(sweep)}", "snapshots swept", "log-spaced over training"),
            cls="toy-stat-grid",
        ),
        H2("Training and learning-rate schedule"),
        P("The cosine schedule is computed analytically so a reader can tell a phase change from the warmup ramp."),
        Panel(
            Chart(line_option({"training loss": train_points}, x_label="optimizer step", y_label="cross-entropy loss", y_log=True), height="100%", controls=[ScrollZoomToggle()]),
            Chart(line_option({"learning rate": lr_points}, x_label="optimizer step", y_label="learning rate"), height="100%"),
            layout="grid", columns=2, height="420px",
        ),
        H2("In-context learning score"),
        P("Mean(loss at token 500) − mean(loss at token 50) over held-out real text. The 2022 paper's phase-change signature: a sharp drop over a narrow step window."),
        Chart(line_option({"ICL score": icl_points}, x_label="optimizer step", y_label="loss[500] − loss[50]"), height="380px"),
        H2("Per-head induction score across training"),
        P("Attention paid from the second copy of a repeated random sequence to the token one before its previous occurrence, averaged over a fixed probe batch."),
        Chart(heatmap_option(induction_labels, induction_steps, induction_values, x_label="checkpoint", y_label="head"), height="480px", show_zoom_controls=False),
        H2("Loss on the repeated sequence"),
        P("Per-token loss over `[eos] + rand(n) + rand(n)`; the drop on the second half is the induction signature."),
        Chart(line_option(per_token_series, x_label="position", y_label="loss"), height="420px"),
        *(
            [
                H2("Composition scores"),
                P("Layer-0 head (row) composing into layer-1 head (column), Frobenius-normalized and baseline-subtracted. K-composition is the induction signature."),
                Panel(
                    Chart(heatmap_option([f"H{i}" for i in range(len(circuits["composition"]["q_comp"]))], [f"H{j}" for j in range(len(circuits["composition"]["q_comp"][0]))], circuits["composition"]["q_comp"], x_label="layer-1 head", y_label="layer-0 head"), height="100%", show_zoom_controls=False),
                    Chart(heatmap_option([f"H{i}" for i in range(len(circuits["composition"]["k_comp"]))], [f"H{j}" for j in range(len(circuits["composition"]["k_comp"][0]))], circuits["composition"]["k_comp"], x_label="layer-1 head", y_label="layer-0 head"), height="100%", show_zoom_controls=False),
                    Chart(heatmap_option([f"H{i}" for i in range(len(circuits["composition"]["v_comp"]))], [f"H{j}" for j in range(len(circuits["composition"]["v_comp"][0]))], circuits["composition"]["v_comp"], x_label="layer-1 head", y_label="layer-0 head"), height="100%", show_zoom_controls=False),
                    layout="grid", columns=3, height="420px",
                ),
            ]
            if circuits
            else []
        ),
        H2("Zero-ablation per head"),
        P("Zeroing one head's output at every position; positive Δ means the head causally supports repeated-token prediction or in-context learning."),
        DataTable(
            ablation_rows,
            [
                {"key": "head", "label": "Head"},
                {"key": "induction", "label": "Induction score"},
                {"key": "prev_token", "label": "Prev-token score"},
                {"key": "delta_repeat_loss", "label": "Δ repeat loss"},
                {"key": "delta_icl", "label": "Δ ICL score"},
            ],
            id="ablation-table", title="Per-head ablation", height="420px",
        ),
        *(
            [
                H2("Attention on the repeated sequence"),
                P("The top induction-scoring heads at the final checkpoint, on the fixed probe sequence. The striped off-diagonal band is the induction signature."),
                *(
                    Div(
                        H2(f"L{head['layer']}H{head['head']}", cls="toy-caption"),
                        CircuitsVis("AttentionPatterns", {"tokens": [str(t) for t in dumped["tokens"]], "attention": [head["pattern"]]}),
                    )
                    for head in dumped["heads"]
                ),
            ]
            if dumped
            else []
        ),
        *(
            [H2("Verdict"), *(P(v, cls="toy-verdict") for v in verdicts)]
            if verdicts
            else []
        ),
    )


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output = args.output or run_dir / "induction_report.html"
    render(lambda: index(load_data(run_dir)), output, title="Two-layer induction circuits")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
