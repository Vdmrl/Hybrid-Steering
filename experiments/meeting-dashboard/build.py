"""Build a dependency-free HTML dashboard from experiment summaries."""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path

LABELS = {
    "candor": "Principled candor",
    "calm": "Calm composure",
    "concrete": "Concrete language",
    "casual": "Casualness",
    "optimism": "Optimism",
}
FEATURES = ("candor", "calm", "concrete", "casual")
COMPOSITION_ROWS = (
    ("candor",),
    ("candor", "calm"),
    ("candor", "calm", "concrete"),
    ("candor", "calm", "casual"),
    ("candor", "calm", "concrete", "casual"),
    ("candor", "concrete"),
    ("candor", "casual"),
    ("candor", "concrete", "casual"),
    ("calm",),
    ("calm", "concrete"),
    ("calm", "casual"),
    ("calm", "concrete", "casual"),
    ("concrete",),
    ("casual",),
    ("concrete", "casual"),
)


def load(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def effect_rows(data: dict | None, key: str, value_key: str) -> list[tuple]:
    if not data:
        return []
    return [
        (
            LABELS.get(name.removeprefix("main-"), name),
            float(values[value_key]),
            [float(item) for item in values["ci95"]],
        )
        for name, values in data.get(key, {}).items()
        if name.startswith("main-") or key == "main_effects"
    ]


def forest(title: str, rows: list[tuple]) -> str:
    if not rows:
        return ""
    width, left, right, row_height = 760, 190, 30, 42
    plot_width = width - left - right

    def x(value: float) -> float:
        return left + (max(-1, min(1, value)) + 1) * plot_width / 2

    parts = [
        f'<h2>{html.escape(title)}</h2><svg viewBox="0 0 {width} {55 + row_height * len(rows)}">'
    ]
    for tick in (-1, -0.5, 0, 0.5, 1):
        parts.append(
            f'<line x1="{x(tick)}" y1="20" x2="{x(tick)}" '
            f'y2="{35 + row_height * len(rows)}" class="grid"/>'
            f'<text x="{x(tick)}" y="{50 + row_height * len(rows)}" '
            f'class="tick">{tick:g}</text>'
        )
    for index, (label, value, interval) in enumerate(rows):
        y = 35 + index * row_height
        parts.extend(
            [
                f'<text x="4" y="{y + 5}" class="label">{html.escape(label)}</text>',
                (
                    f'<line x1="{x(interval[0])}" y1="{y}" '
                    f'x2="{x(interval[1])}" y2="{y}" class="ci"/>'
                ),
                f'<circle cx="{x(value)}" cy="{y}" r="6" class="point"/>',
                (
                    f'<text x="{width - 4}" y="{y + 5}" class="value">'
                    f"{value:+.2f} [{interval[0]:+.2f}, {interval[1]:+.2f}]</text>"
                ),
            ]
        )
    return "".join(parts) + "</svg>"


def interaction_table(data: dict | None) -> str:
    if not data:
        return ""
    interactions = data.get("factorial_interactions") or data.get("interactions") or {}
    features = sorted(
        {key.split("|", 1)[0] for key in interactions}
        | {key.split("context=", 1)[1] for key in interactions}
    )
    if not features:
        return ""
    cells = ["<h2>Interaction effects</h2><table><tr><th>Target ↓ / context →</th>"]
    cells += [f"<th>{html.escape(LABELS.get(name, name))}</th>" for name in features]
    cells.append("</tr>")
    for target in features:
        cells.append(f"<tr><th>{html.escape(LABELS.get(target, target))}</th>")
        for context in features:
            if target == context:
                cells.append("<td class='na'>—</td>")
                continue
            item = interactions.get(f"{target}|context={context}")
            if not item:
                cells.append("<td class='na'>pending</td>")
                continue
            value = float(item["difference_in_differences"])
            alpha = min(0.85, 0.15 + abs(value) * 0.7)
            color = (
                f"rgba(70,170,120,{alpha})"
                if value >= 0
                else f"rgba(224,95,95,{alpha})"
            )
            cells.append(
                f"<td style='background:{color}' title='95% CI: "
                f"{item['ci95'][0]:+.2f} … {item['ci95'][1]:+.2f}'>{value:+.2f}</td>"
            )
        cells.append("</tr>")
    return "".join(cells) + "</table><p class='hint'>Hover a cell for its 95% CI.</p>"


def composition_depth(data: dict | None) -> str:
    if not data or not data.get("effects_by_context_size"):
        return ""
    rows = data["effects_by_context_size"]
    cells = [
        "<h2>Does steering survive composition?</h2>",
        "<p class='hint'>Effect of each target with 0–3 other features already active.</p>",
        "<table><tr><th>Target</th><th>Alone</th><th>+1 other</th>",
        "<th>+2 others</th><th>+3 others (all four)</th></tr>",
    ]
    for feature, depths in rows.items():
        cells.append(f"<tr><th>{html.escape(LABELS.get(feature, feature))}</th>")
        for depth in ("0", "1", "2", "3"):
            item = depths[depth]
            low, high = item["ci95"]
            cells.append(
                f"<td title='95% CI: {low:+.2f} … {high:+.2f}'>"
                f"<strong>{item['effect']:+.2f}</strong><br>"
                f"<span class='small'>[{low:+.2f}, {high:+.2f}]</span></td>"
            )
        cells.append("</tr>")
    return "".join(cells) + "</table>"


def composition_matrix(data: dict | None) -> str:
    if not data or not data.get("effects_by_context"):
        return ""
    contexts = data["effects_by_context"]

    def value(active: tuple[str, ...], feature: str) -> str:
        context = (
            "+".join(name for name in FEATURES if name in active and name != feature)
            or "none"
        )
        item = contexts[feature][context]
        low, high = item["ci95"]
        status = "positive" if low > 0 else "negative" if high < 0 else "uncertain"
        return (
            f"<span class='{status}'><strong>{item['effect']:+.2f}</strong></span><br>"
            f"<span class='small'>[{low:+.2f}, {high:+.2f}]</span>"
        )

    rows = [
        "<h2>Composition results at a glance</h2>",
        (
            "<p class='hint'>One row per composition, grouped by the first and "
            "then the second feature. Each feature column is judged separately.</p>"
        ),
        "<table><tr><th>Active composition</th>",
    ]
    rows += [f"<th>{html.escape(LABELS[feature])}</th>" for feature in FEATURES]
    rows.append("</tr>")
    for active in COMPOSITION_ROWS:
        title = " + ".join(LABELS[feature] for feature in active)
        rows.append(f"<tr><th>{html.escape(title)}</th>")
        for feature in FEATURES:
            if feature in active:
                rows.append(f"<td>{value(active, feature)}</td>")
            else:
                rows.append("<td class='na'>—</td>")
        rows.append("</tr>")
    return "".join(rows) + "</table>"


def verdict_table(data: dict | None) -> str:
    if not data or not data.get("effects_by_context"):
        return ""
    contexts = data["effects_by_context"]
    rows = [
        "<h2>Final composition verdict</h2>",
        (
            "<p class='hint'>There are 8 contexts because the other three "
            "features have 2³ ON/OFF configurations. Each context contains "
            "up to 128 held-out prompts; 8/8 is not the sample size.</p>"
        ),
        "<table><tr><th>Feature</th>",
        "<th>Standalone</th><th>With all four</th>",
        "<th>Contexts with positive 95% CI (of 8)</th><th>Verdict</th></tr>",
    ]
    for feature in FEATURES:
        items = contexts[feature]
        standalone = items["none"]
        all_others = "+".join(name for name in FEATURES if name != feature)
        composed = items[all_others]
        reliable = sum(item["ci95"][0] > 0 for item in items.values())
        if reliable == len(items):
            verdict = "robustly composes"
        elif composed["ci95"][0] > 0:
            verdict = "composes, context-sensitive"
        elif reliable:
            verdict = "weak / context-dependent"
        else:
            verdict = "direction failed"
        rows.append(
            f"<tr><th>{html.escape(LABELS[feature])}</th>"
            f"<td>{standalone['effect']:+.2f}</td>"
            f"<td>{composed['effect']:+.2f}</td>"
            f"<td>{reliable}/{len(items)}</td><td>{verdict}</td></tr>"
        )
    return "".join(rows) + "</table>"


def exact_compositions(data: dict | None) -> str:
    if not data or not data.get("effects_by_context"):
        return ""
    blocks = [
        "<h2>Standalone vs. composed effects</h2>",
        (
            "<p class='hint'>Every active feature is judged separately. "
            "“Standalone” adds it to the neutral condition; “in composition” "
            "adds the same feature while all other listed features are active.</p>"
        ),
    ]
    for active in (row for row in COMPOSITION_ROWS if len(row) > 1):
        rows = []
        for feature in active:
            contexts = data["effects_by_context"][feature]
            standalone = contexts["none"]
            context = "+".join(
                name for name in FEATURES if name in active and name != feature
            )
            composed = contexts[context]
            low, high = composed["ci95"]
            status = "positive" if low > 0 else "negative" if high < 0 else "uncertain"
            delta = composed["effect"] - standalone["effect"]
            rows.append(
                f"<tr><th>{html.escape(LABELS[feature])}</th>"
                f"<td>{standalone['effect']:+.2f}<br>"
                f"<span class='small'>[{standalone['ci95'][0]:+.2f}, "
                f"{standalone['ci95'][1]:+.2f}]</span></td>"
                f"<td class='{status}'><strong>{composed['effect']:+.2f}</strong><br>"
                f"<span class='small'>[{low:+.2f}, {high:+.2f}]</span></td>"
                f"<td>{delta:+.2f}</td></tr>"
            )
        title = " + ".join(LABELS[feature] for feature in active)
        blocks.append(
            f"<details open><summary>{html.escape(title)}</summary>"
            "<table><tr><th>Feature judged separately</th>"
            "<th>Standalone effect</th><th>Effect in composition</th>"
            "<th>Change</th></tr>" + "".join(rows) + "</table></details>"
        )
    return "".join(blocks)


def composition_cards(title: str, data: dict | None) -> str:
    if not data:
        return f"<section><h2>{html.escape(title)}</h2><p class='pending'>Pending</p></section>"
    cards = []
    for name, item in data.items():
        if not isinstance(item, dict) or "trait_effect" not in item:
            continue
        ci = item["trait_ci95"]
        cards.append(
            f"<article><h3>{html.escape(name.replace('_', ' '))}</h3>"
            f"<strong>{item['trait_effect']:+.2f}</strong>"
            f"<span>95% CI {ci[0]:+.2f} … {ci[1]:+.2f}</span>"
            f"<span>quality {item['quality_effect']:+.2f}</span></article>"
        )
    return (
        f"<section><h2>{html.escape(title)}</h2><div class='cards'>"
        + "".join(cards)
        + "</div></section>"
    )


def concept_guide() -> str:
    return """<h2>Steering concepts / Что означают признаки</h2>
<p class="hint">Positive effects favor the first pole; negative effects favor
the opposite pole used to construct the direction.</p>
<table>
<tr><th>Feature</th><th>Положительный полюс</th><th>Противоположный полюс</th><th>α</th></tr>
<tr><th>Principled candor</th><td>Принципиальная прямота: вежливо указывать на
ошибки и не поддакивать</td><td>Sycophancy — угодливое согласие</td><td>8</td></tr>
<tr><th>Calm composure</th><td>Спокойствие и самообладание</td>
<td>Fear / panic — страх и паника</td><td>2</td></tr>
<tr><th>Concrete language</th><td>Конкретный, предметный язык</td>
<td>Abstract language — абстрактный язык</td><td>4</td></tr>
<tr><th>Casualness</th><td>Неформальный, разговорный стиль</td>
<td>Formality — формальный стиль</td><td>1</td></tr>
</table>"""


def build(args: argparse.Namespace) -> str:
    base = load(args.four_axis)
    optimism = load(args.optimism)
    main_data = optimism or base
    trait = effect_rows(
        main_data,
        "main_effects" if optimism else "pairwise_effects",
        "effect" if optimism else "signed_effect",
    )
    quality = effect_rows(
        main_data,
        "quality_effects",
        "effect" if optimism else "signed_effect",
    )
    complete = sum(
        item is not None
        for item in (base, load(args.calm_french), load(args.candor_french), optimism)
    )
    body = (
        f"<header><p>Hybrid Steering · Qwen3.5-9B</p><h1>Meeting dashboard</h1>"
        f"<p>{complete}/4 summaries available · generated "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p></header>"
        f"<main><section>{concept_guide()}</section>"
        f"<section>{forest('Main steering effects', trait)}</section>"
        f"<section>{verdict_table(base)}</section>"
        f"<section>{composition_depth(base)}</section>"
        f"<section>{composition_matrix(base)}</section>"
        f"<section>{exact_compositions(base)}</section>"
        f"<section>{forest('Answer-quality effects', quality)}</section>"
        f"<section>{interaction_table(main_data)}</section>"
        f"{composition_cards('Calm + French', load(args.calm_french))}"
        f"{composition_cards('Candor + French', load(args.candor_french))}</main>"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="refresh" content="60"><title>Hybrid Steering results</title>
<style>
:root{{--bg:#0d1117;--panel:#161b22;--text:#e6edf3;--muted:#8b949e;--accent:#8b8cff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);
font:15px/1.45 system-ui,sans-serif}}header,main{{max-width:1060px;margin:auto;padding:24px}}
header h1{{font-size:34px;margin:.1em 0}}header p{{color:var(--muted)}}section{{background:var(--panel);
border:1px solid #30363d;border-radius:12px;margin:16px 0;padding:20px;overflow:auto}}
h2{{font-size:18px;margin:0 0 16px}}svg{{width:100%;min-width:680px}}.grid{{stroke:#30363d}}
.ci{{stroke:#b7c0d8;stroke-width:4;stroke-linecap:round}}.point{{fill:var(--accent)}}
.label{{fill:var(--text)}}.tick{{fill:var(--muted);text-anchor:middle}}.value{{fill:var(--muted);text-anchor:end}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:11px;border:1px solid #30363d;text-align:center}}
th:first-child{{text-align:left}}.na,.pending,.hint{{color:var(--muted)}}.cards{{display:grid;
grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}article{{background:#0d1117;
border-radius:9px;padding:14px}}article h3{{font-size:14px;margin:0 0 10px;text-transform:capitalize}}
article strong{{display:block;font-size:28px}}article span{{display:block;color:var(--muted)}}
.positive{{color:#62d394}}.negative{{color:#ff7b72}}.uncertain{{color:#d2a85a}}
details{{margin:12px 0}}summary{{cursor:pointer;font-weight:700;margin-bottom:8px}}.small{{color:var(--muted)}}
</style></head><body>{body}</body></html>"""


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--four-axis", type=Path)
    parser.add_argument("--calm-french", type=Path)
    parser.add_argument("--candor-french", type=Path)
    parser.add_argument("--optimism", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build(args), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
