"""Render the one-layer toy-model experiment as a self-contained HTML explainer."""

import argparse
import csv
import json
import math
from pathlib import Path

from fasthtml.common import Div, H1, H2, P, Span, Strong, Style, Titled
from research_viz import (
    Badge,
    Chart,
    CircuitsVis,
    ClearFiltersButton,
    DataTable,
    FilterBar,
    Panel,
    ScrollZoomToggle,
    SearchInput,
    SelectFilter,
    heatmap_option,
    line_option,
    render,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_checkpoint_heatmap(run_dir: Path) -> tuple[list[str], list[str], list[list[float]]]:
    """Average OV logit boost to the expected output token, per head, per checkpoint, across
    that head's top-60 candidate skip-trigrams - shows which heads develop a skip-trigram
    circuit and when, not just at the final step. Averaging (rather than taking the best
    artifact-filtered candidate) means every head/checkpoint cell is always defined, even for
    heads whose top candidates are dominated by a garbage-token shortcut."""
    steps = sorted(
        int(p.name.removeprefix("step_")) for p in (run_dir / "reports").glob("step_*")
    )
    n_heads = 12
    values = [[0.0] * len(steps) for _ in range(n_heads)]
    for j, step in enumerate(steps):
        rows = json.loads((run_dir / "reports" / f"step_{step}" / "skip_trigrams.json").read_text())["rows"]
        sums = [0.0] * n_heads
        counts = [0] * n_heads
        for row in rows:
            sums[row["head"]] += row["ov_logit_delta"]
            counts[row["head"]] += 1
        for head in range(n_heads):
            values[head][j] = sums[head] / counts[head] if counts[head] else 0.0
    return [f"H{i}" for i in range(n_heads)], [f"step {s}" for s in steps], values


def load_skip_trigram_examples(run_dir: Path, top_n: int = 6) -> list[dict]:
    """Render the selected source keys in the paper's QK/OV table shape."""
    steps = sorted(
        int(p.name.removeprefix("step_")) for p in (run_dir / "reports").glob("step_*")
    )
    rows = json.loads(
        (run_dir / "reports" / f"step_{steps[-1]}" / "skip_trigrams.json").read_text()
    )["rows"]

    groups: dict[tuple[int, int], dict] = {}
    for row in rows:
        key = (row["head"], row["source_id"])
        group = groups.setdefault(
            key,
            {"head": row["head"], "source": row["source"], "destinations": {}, "outputs": {}, "score": row.get("key_score", 0.0)},
        )
        group["destinations"][row["destination"]] = max(
            group["destinations"].get(row["destination"], -math.inf), row["qk_score"]
        )
        group["outputs"][row["output"]] = max(
            group["outputs"].get(row["output"], -math.inf), row["ov_logit_delta"]
        )
        group["score"] = max(group["score"], row.get("key_score", 0.0))

    best_per_source: dict[int, dict] = {}
    for (head, source_id), group in groups.items():
        best = best_per_source.get(source_id)
        if best is None or group["score"] > best["score"]:
            best_per_source[source_id] = group

    ranked = sorted(
        best_per_source.values(),
        key=lambda g: g["score"],
        reverse=True,
    )[:top_n]
    return [
        {
            "head": f"H{g['head']}",
            "source": g["source"],
            "destinations": [
                f"{token} ({score:.2f})"
                for token, score in sorted(g["destinations"].items(), key=lambda item: item[1], reverse=True)[:4]
            ],
            "outputs": [
                f"{token} ({score:.2f})"
                for token, score in sorted(g["outputs"].items(), key=lambda item: item[1], reverse=True)[:4]
            ],
            "score": f"{g['score']:.2f}",
        }
        for g in ranked
    ]


def load_data(run_dir: Path) -> dict:
    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    final = metrics[-1]
    causal = list(csv.DictReader((run_dir / "reports" / "causal_probe.csv").open()))
    return {
        "metrics": metrics,
        "final": final,
        "causal": causal,
        "heatmap": load_checkpoint_heatmap(run_dir),
        "skip_trigram_examples": load_skip_trigram_examples(run_dir),
        "attention_patterns": json.loads(
            (run_dir / "reports" / "attention_patterns.json").read_text()
        )["probes"] if (run_dir / "reports" / "attention_patterns.json").exists() else [],
    }


def num(value: str) -> float:
    return float(value)


def stat(value: str, label: str, detail: str):
    return Div(
        Span(value, cls="toy-stat-value"),
        Strong(label, cls="toy-stat-label"),
        Span(detail, cls="toy-stat-detail"),
        cls="toy-stat",
    )


def circuit_path(head: str, source: str, destination: str, output: str):
    return Div(
        Div(source, cls="toy-token toy-source"),
        Span("…", cls="toy-ellipsis"),
        Div(destination, cls="toy-token toy-destination"),
        Span("QK", cls="toy-arrow"),
        Div(head, cls="toy-head"),
        Span("OV", cls="toy-arrow"),
        Div(output, cls="toy-token toy-output"),
        cls="toy-path",
    )


CIRCUIT_EXAMPLES = [
    ("H0", "' York'", "' New'", "' York'", "Functionally confirmed"),
    ("H5", "' for'", "' a'", "' lot'", "Learned head path; canceled elsewhere"),
    ("H6", "' No'", "'.'", "' 1'", "Functionally confirmed; peaked early"),
    ("H8", "' can'", "' can'", "\"'t\"", "Learned head path; canceled elsewhere"),
]


def index(data: dict):
    final = data["final"]
    perplexity = math.exp(final["valid_loss"])
    train_points = [
        (record["optimizer_step"], record["train_loss"])
        for record in data["metrics"]
        if record.get("optimizer_step", 0) and "train_loss" in record
    ]
    attention = {}
    ablation = {}
    gain = {}
    for row in data["causal"]:
        name = f"H{row['head']} · {row['trigram']}"
        attention.setdefault(name, []).append((int(row["step"]), num(row["attention_to_source"])))
        ablation.setdefault(name, []).append((int(row["step"]), num(row["target_head_ablation_drop"])))
        gain.setdefault(name, []).append((int(row["step"]), num(row["source_logprob_gain"])))
    heatmap_heads, heatmap_steps, heatmap_values = data["heatmap"]

    causal_rows = [
        {
            "id": f"{row['step']}-{row['head']}",
            "step": row["step"],
            "head": f"H{row['head']}",
            "trigram": row["trigram"],
            "attention": f"{num(row['attention_to_source']):.3f}",
            "ablation": f"{num(row['target_head_ablation_drop']):+.3f}",
            "gain": f"{num(row['source_logprob_gain']):+.3f}",
            "search": f"{row['step']} {row['head']} {row['trigram']}",
            "filters": {"head": f"H{row['head']}"},
        }
        for row in data["causal"]
    ]

    return Titled(
        "One-layer attention circuits",
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
            .toy-callout { border-left: 3px solid #c15f3c; padding: 1rem 1.15rem; background: color-mix(in srgb, #c15f3c 7%, transparent); margin: 1.25rem 0 2rem; }
            .toy-path { display:flex; align-items:center; gap:.65rem; flex-wrap:wrap; padding:1rem; border:1px solid var(--rv-border); border-radius:.7rem; background:var(--rv-card); }
            .toy-token,.toy-head { padding:.35rem .6rem; border-radius:.4rem; font:600 .9rem ui-monospace, monospace; }
            .toy-source { background:#e8f0fc; color:#24548a; }.toy-destination{background:#fae8df;color:#984321;}.toy-output{background:#e7f3e8;color:#256b3a;}.toy-head{background:#282c34;color:#fff;}
            .toy-arrow { color: var(--rv-muted); font:700 .75rem ui-monospace, monospace; }.toy-ellipsis{color:var(--rv-muted);font-size:1.35rem;}
            .toy-caption { color:var(--rv-muted); font-size:.88rem; margin-top:.65rem; }
            .toy-badge-row { display:flex; flex-wrap:wrap; gap:.3rem; }
            @media(max-width:700px){.toy-stat-grid{grid-template-columns:repeat(2,minmax(0,1fr));}.toy-hero{padding-top:2.5rem;}}
            """
        ),
        Div(
            Span("Transformer Circuits · Toy Models", cls="toy-kicker"),
            H1("A one-layer model learns to look back."),
            P(
                "A 81M-parameter attention-only transformer was trained for 1.31B tokens. "
                "This explainer follows the evidence from loss curves to QK/OV paths to causal head ablations.",
                cls="toy-lede",
            ),
            cls="toy-hero",
        ),
        Div(
            stat("10,000", "optimizer steps", "1.310B predicted tokens"),
            stat(f"{final['valid_loss']:.3f}", "held-out loss", f"ppl {perplexity:.1f}"),
            stat("2:05:40", "H200 runtime", "physical GPU 3"),
            stat("81.2M", "parameters", "1 layer · 12 heads"),
            cls="toy-stat-grid",
        ),
        H2("The model learned the task"),
        P("Zoom or expand the curve. The final held-out result is the red reference line."),
        Chart(
            line_option(
                {"training loss": train_points},
                x_label="optimizer step",
                y_label="cross-entropy loss",
                styles={"training loss": {"color": "#4e79a7"}},
                mark_lines=[{"x": 10_000, "label": f"validation {final['valid_loss']:.3f}", "color": "#c15f3c"}],
                y_log=True,
            ),
            height="420px",
            controls=[ScrollZoomToggle()],
        ),
        H2("A circuit in one sentence"),
        P("At the destination token, a head can use its QK path to select an earlier source token, then use its OV path to promote a likely next token."),
        circuit_path("H0", "' York'", "' New'", "' York'"),
        P("This is the strongest whole-model probe: the earlier token becomes useful after training, and ablating head 0 removes 1.78 logits from the target.", cls="toy-caption"),
        Div(
            Strong("Interpretation discipline. "),
            "A top QK/OV score is only a hypothesis. The causal section below tests selected hypotheses by changing the source token and zeroing the claimed head.",
            cls="toy-callout",
        ),
        H2("Which heads learn a skip-trigram circuit, and when"),
        P("Not attention weight - the average OV-path logit boost to the expected output token, across each head's top-60 candidate skip-trigrams at that checkpoint."),
        Chart(
            heatmap_option(heatmap_heads, heatmap_steps, heatmap_values, x_label="checkpoint", y_label="head"),
            height="480px", show_zoom_controls=False,
        ),
        H2("Large entries in the QK/OV circuit"),
        P(
            "A source-pivoted, compact QK/OV dump. QK is normalized against the end-of-text "
            "key, OV logits are mean-centered, and rows are ranked by the paper's automated heuristic."
        ),
        DataTable(
            [
                {
                    "id": f"skip-{i}",
                    "head": row["head"],
                    "source": row["source"],
                    "destinations": row["destinations"],
                    "outputs": row["outputs"],
                    "score": row["score"],
                    "search": f"{row['head']} {row['source']} {' '.join(row['destinations'])} {' '.join(row['outputs'])}",
                }
                for i, row in enumerate(data["skip_trigram_examples"])
            ],
            [
                {"key": "head", "label": "Head"},
                {"key": "source", "label": "Source Token"},
                {
                    "key": "destinations",
                    "label": "Destination Token",
                    "render": lambda r: Div(*(Badge(d) for d in r["destinations"]), cls="toy-badge-row"),
                },
                {
                    "key": "outputs",
                    "label": "Out Token",
                    "render": lambda r: Div(*(Badge(o, kind="positive") for o in r["outputs"]), cls="toy-badge-row"),
                },
                {"key": "score", "label": "Key Score"},
            ],
            id="skip-trigram-examples", title="Top QK/OV entries", height="420px",
        ),
        H2("Formation across checkpoints"),
        P("Each series starts from the identically seeded untrained model. Hover for exact values; use the chart controls to zoom into the early phase."),
        Panel(
            Chart(line_option(attention, x_label="optimizer step", y_label="attention to earlier source", legend="bottom"), height="100%"),
            Chart(line_option(ablation, x_label="optimizer step", y_label="target logit lost on head ablation", legend="bottom"), height="100%"),
            Chart(line_option(gain, x_label="optimizer step", y_label="source log-prob gain", legend="bottom"), height="100%"),
            layout="grid", columns=3, height="430px",
        ),
        H2("Causal probe explorer"),
        P("The probe context is `[source] + eight copies of ' the' + [destination]`. Positive ablation drop means the claimed head causally supports the target. Filter by head or search a trigram."),
        FilterBar(
            SelectFilter("causal-probes", "head", ["H0", "H5", "H6", "H8"], label="Head"),
            SearchInput("causal-probes", label="Search", placeholder="York, lot, No, can..."),
            ClearFiltersButton("causal-probes"),
        ),
        DataTable(
            causal_rows,
            [
                {"key": "step", "label": "Step"},
                {"key": "head", "label": "Head"},
                {"key": "trigram", "label": "Source … destination → output"},
                {"key": "attention", "label": "Attention"},
                {"key": "ablation", "label": "Head Δlogit"},
                {"key": "gain", "label": "Source Δlog p"},
            ],
            id="causal-probes", title="Causal trajectory", height="430px",
        ),
        *(
            Div(
                H2(f"{probe['head']} attention: {probe['label']}"),
                P(
                    "At the final checkpoint, this is the full attention pattern over the "
                    "same fixed context used by the causal probe. Select the claimed head "
                    "to inspect whether the destination reads the earlier source."
                ),
                CircuitsVis(
                    "AttentionPatterns",
                    {"tokens": probe["tokens"], "attention": probe["attention"]},
                ),
            )
            for probe in data["attention_patterns"]
        ),
        H2("The four tested circuits"),
        P("Every skip-trigram hypothesis that was causally probed above, with its final verdict."),
        *(
            Div(
                circuit_path(head, source, destination, output),
                P(verdict, cls="toy-caption"),
                style="margin-bottom:1.25rem;",
            )
            for head, source, destination, output, verdict in CIRCUIT_EXAMPLES
        ),
        H2("What this establishes"),
        P("The one-layer model contains learned, interpretable QK/OV pathways. Two probes are functionally confirmed at whole-model level (`York … New → York` and `No … . → 1`); two others have strong head-level pathways whose net effect is canceled elsewhere. This is a partial reproduction of the one-layer circuit claim, not every result in the original paper."),
    )


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output = args.output or run_dir / "interactive.html"
    render(lambda: index(load_data(run_dir)), output, title="One-layer attention circuits")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
