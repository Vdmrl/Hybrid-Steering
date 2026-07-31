"""Build a compact, presentation-ready dashboard from completed summaries."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP3 = ROOT / "experiments/meeting-dashboard-3-real/exp3_summary.json"
EXP4 = ROOT / "outputs/strong-composition-exp4/summary.json"
OUTPUT = ROOT / "outputs/meeting-dashboard-interim/index.html"

FEATURES = {
    "joy": "Радость",
    "concrete": "Конкретный язык",
    "optimism": "Оптимизм",
    "candor": "Принципиальная прямота",
}
METHODS = {
    "gdn_raw_r1": "Raw, rank 1",
    "gdn_rss_r1": "RSS, rank 1",
    "gdn_raw_r4": "Raw, rank 4",
    "gdn_rss_r4": "RSS, rank 4",
}


def condition(summary: dict, name: str) -> dict:
    return next(item for item in summary["conditions"] if item["condition"] == name)


def contrast(summary: dict, name: str, mask: str = "1111") -> dict:
    return next(
        item
        for item in summary["contrasts"]
        if item["contrast"] == name and item["mask"] == mask
    )


def score(value: float) -> str:
    return f"{value:.3f}"


def signed(value: float) -> str:
    return f"{value:+.3f}"


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def pp(value: float) -> str:
    return f"{value * 100:+.1f} п.п."


def ci(metric: dict, *, rate: bool = False) -> str:
    formatter = percent if rate else score
    return f"[{formatter(metric['ci95_low'])}; {formatter(metric['ci95_high'])}]"


def status(metric: dict) -> tuple[str, str]:
    if metric["ci95_low"] > 0:
        return "Положительный", "positive"
    if metric["ci95_high"] < 0:
        return "Отрицательный", "negative"
    return "Неопределённый", "uncertain"


def feature_names(item: dict) -> str:
    return " + ".join(FEATURES.get(name, name) for name in item["active_features"])


def forest_row(label: str, metric: dict, scale: float = 0.4) -> str:
    def x(value: float) -> float:
        return max(0, min(100, (value + scale) / (2 * scale) * 100))

    left = x(metric["ci95_low"])
    width = x(metric["ci95_high"]) - left
    dot = x(metric["mean"])
    return f"""
    <div class="forest-row">
      <div>{html.escape(label)}</div>
      <div class="forest-track"><i class="zero"></i><i class="ci" style="left:{left:.2f}%;width:{width:.2f}%"></i><i class="dot" style="left:{dot:.2f}%"></i></div>
      <strong>{signed(metric["mean"])}</strong><span>{ci(metric)}</span>
    </div>"""


def profile_rows(exp4: dict) -> str:
    variants = [
        ("baseline", "Baseline", "base"),
        ("gdn_raw_r1_1111", "GDN raw, rank 1", "raw"),
        ("gdn_rss_r4_1111", "GDN RSS, rank 4", "rss"),
    ]
    rows = []
    for key, feature_label in FEATURES.items():
        cells = []
        for condition_id, label, color in variants:
            item = condition(exp4, condition_id)
            value = item["features"][key]["expected"]["mean"]
            cells.append(
                f'<div class="barline"><span>{html.escape(label)}</span><b class="bar {color}" style="width:{value / 5 * 100:.1f}%"></b><em>{score(value)}</em></div>'
            )
        rows.append(
            f'<article class="profile"><h3>{feature_label}</h3>{"".join(cells)}</article>'
        )
    return "".join(rows)


def pair_rows(exp4: dict) -> str:
    masks = ["0011", "0101", "0110", "1001", "1010", "1100"]
    rows = []
    for mask in masks:
        reference = condition(exp4, f"gdn_raw_r1_{mask}")
        cells = []
        for method in METHODS:
            item = condition(exp4, f"{method}_{mask}")
            cells.append(
                f"<td><b>{score(item['mean_minimum_expected']['mean'])}</b><small>joint {percent(item['all_active_ge4']['mean'])}</small></td>"
            )
        rows.append(f"<tr><th>{feature_names(reference)}</th>{''.join(cells)}</tr>")
    return "".join(rows)


def method_rows(exp4: dict) -> str:
    baseline = condition(exp4, "baseline")
    rows = [
        f"<tr><th>Baseline</th><td>{score(baseline['mean_minimum_expected']['mean'])}</td><td>{percent(baseline['all_active_ge4']['mean'])}</td><td>{score(baseline['quality']['mean'])}</td></tr>"
    ]
    for method, label in METHODS.items():
        item = condition(exp4, f"{method}_1111")
        rows.append(
            f"<tr><th>{label}</th><td>{score(item['mean_minimum_expected']['mean'])}</td><td>{percent(item['all_active_ge4']['mean'])}</td><td>{score(item['quality']['mean'])}</td></tr>"
        )
    return "".join(rows)


def interference_tables(exp4: dict) -> str:
    bits = {"joy": 1, "concrete": 2, "optimism": 4, "candor": 8}
    tables = []
    for method in ("gdn_raw_r1", "gdn_raw_r4"):
        rows = []
        for added, added_label in FEATURES.items():
            cells = []
            for target in FEATURES:
                if added == target:
                    cells.append('<td class="muted">—</td>')
                    continue
                pair_mask = f"{bits[added] + bits[target]:04b}"
                singleton_mask = f"{bits[target]:04b}"
                pair = condition(exp4, f"{method}_{pair_mask}")
                singleton = condition(exp4, f"{method}_{singleton_mask}")
                delta = (
                    pair["features"][target]["expected"]["mean"]
                    - singleton["features"][target]["expected"]["mean"]
                )
                css_class = "good" if delta > 0 else "bad"
                cells.append(f'<td class="{css_class}"><b>{signed(delta)}</b></td>')
            rows.append(f"<tr><th>{added_label}</th>{''.join(cells)}</tr>")
        tables.append(
            f"<article><h3>{METHODS[method]}</h3><table><tr><th>Добавили ↓ / измеряем →</th>"
            + "".join(f"<th>{label}</th>" for label in FEATURES.values())
            + f"</tr>{''.join(rows)}</table></article>"
        )
    return "".join(tables)


def build(exp3: dict, exp4: dict) -> str:
    gdn_vs_activation = contrast(exp3, "gdn_minus_activation_raw")["paired_difference"]
    exp3_rank = contrast(exp3, "gdn_rank4_minus_rank1")["paired_difference"]
    rss_r4 = contrast(exp4, "rss_minus_raw_rank4")["delta_mean_minimum_expected"]
    raw_rank = contrast(exp4, "rank4_minus_rank1_raw")["delta_mean_minimum_expected"]
    rss_rank = contrast(exp4, "rank4_minus_rank1_rss")["delta_mean_minimum_expected"]
    rss_r1 = contrast(exp4, "rss_minus_raw_rank1")["delta_mean_minimum_expected"]
    best_score = condition(exp4, "gdn_raw_r1_1111")
    best_joint = condition(exp4, "gdn_rss_r4_1111")
    baseline = condition(exp4, "baseline")

    forest = "".join(
        [
            forest_row("GDN raw r1 − activation raw", gdn_vs_activation),
            forest_row("RSS r4 − RSS r1", exp3_rank),
            forest_row("RSS r4 − raw r4", rss_r4),
            forest_row("Raw r4 − raw r1", raw_rank),
            forest_row("RSS r4 − RSS r1", rss_rank),
        ]
    )
    method_status = status(rss_r4)[0]
    return TEMPLATE.format(
        n=baseline["n"],
        baseline_joint=percent(baseline["all_active_ge4"]["mean"]),
        best_score=score(best_score["mean_minimum_expected"]["mean"]),
        best_score_joint=percent(best_score["all_active_ge4"]["mean"]),
        best_joint=percent(best_joint["all_active_ge4"]["mean"]),
        best_joint_score=score(best_joint["mean_minimum_expected"]["mean"]),
        gdn_activation=signed(gdn_vs_activation["mean"]),
        gdn_activation_ci=ci(gdn_vs_activation),
        rss_gain=signed(rss_r4["mean"]),
        rss_gain_ci=ci(rss_r4),
        method_status=method_status,
        method_rows=method_rows(exp4),
        forest=forest,
        profile_rows=profile_rows(exp4),
        pair_rows=pair_rows(exp4),
        method_headers="".join(f"<th>{label}</th>" for label in METHODS.values()),
        interference_tables=interference_tables(exp4),
        rss_r1=signed(rss_r1["mean"]),
        rss_r1_ci=ci(rss_r1),
        raw_rank=signed(raw_rank["mean"]),
        raw_rank_ci=ci(raw_rank),
        rss_rank=signed(rss_rank["mean"]),
        rss_rank_ci=ci(rss_rank),
        judge_cost=f"${exp4['judge_usage']['estimated_usd']:.2f}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp3", type=Path, default=EXP3)
    parser.add_argument("--exp4", type=Path, default=EXP4)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    exp3 = json.loads(args.exp3.read_text(encoding="utf-8"))
    exp4 = json.loads(args.exp4.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build(exp3, exp4), encoding="utf-8")
    print(args.output)


TEMPLATE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hybrid Steering — промежуточные результаты</title>
<style>
:root{{--bg:#07111d;--panel:#101d2c;--panel2:#0b1724;--line:#2b4058;--text:#edf5ff;--muted:#9fb1c6;--cyan:#58ded2;--blue:#8191ff;--green:#55dc94;--amber:#f1bd65;--red:#ff8585}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 Inter,system-ui,sans-serif}}header,main{{max-width:1380px;margin:auto;padding:24px}}header{{padding-top:38px}}h1{{font-size:38px;line-height:1.05;margin:5px 0 10px}}h2{{font-size:21px;margin:0 0 12px}}h3{{font-size:14px;margin:0 0 8px}}.eyebrow{{color:var(--cyan);font-weight:800;text-transform:uppercase;letter-spacing:.08em}}.muted,small{{color:var(--muted)}}.grid{{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}}.panel{{grid-column:span 12;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px;overflow:auto}}.half{{grid-column:span 6}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:14px}}.card,.profile{{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:13px}}.card strong{{display:block;font-size:27px;margin:4px 0}}.tag{{display:inline-block;border:1px solid var(--line);border-radius:99px;padding:4px 9px;margin:3px;color:#d8e5f5}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{font-size:12px;color:#cad8e8}}td small{{display:block}}.good{{color:var(--green)}}.warn{{color:var(--amber)}}.bad{{color:var(--red)}}.callout{{border-left:3px solid var(--cyan);padding:10px 13px;background:var(--panel2);border-radius:0 8px 8px 0}}.forest-row{{display:grid;grid-template-columns:210px 1fr 60px 145px;gap:9px;align-items:center;margin:12px 0}}.forest-track{{height:20px;position:relative;background:#07111b;border-radius:99px}}.forest-track .zero{{position:absolute;left:50%;height:100%;border-left:1px dashed var(--amber)}}.forest-track .ci{{position:absolute;top:8px;height:4px;background:var(--cyan);border-radius:99px}}.forest-track .dot{{position:absolute;top:4px;width:12px;height:12px;margin-left:-6px;background:var(--green);border:2px solid var(--panel2);border-radius:50%}}.profiles{{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}}.barline{{display:grid;grid-template-columns:130px 1fr 44px;gap:7px;align-items:center;margin:7px 0}}.barline span{{font-size:11px;color:var(--muted)}}.bar{{height:12px;border-radius:99px;min-width:2px}}.base{{background:#687b92}}.raw{{background:var(--blue)}}.rss{{background:var(--green)}}.barline em{{font-style:normal;font-size:12px}}.conclusions{{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}}.conclusions article{{border-top:3px solid var(--cyan);background:var(--panel2);border-radius:9px;padding:13px}}@media(max-width:900px){{.half{{grid-column:span 12}}.cards,.profiles,.conclusions{{grid-template-columns:1fr 1fr}}}}@media(max-width:620px){{header,main{{padding:15px}}h1{{font-size:30px}}.cards,.profiles,.conclusions{{grid-template-columns:1fr}}.forest-row{{grid-template-columns:130px 1fr 55px}}.forest-row span:last-child{{display:none}}}}
</style></head><body>
<header><div class="eyebrow">Hybrid Steering · Qwen3.5-9B</div><h1>Промежуточные результаты</h1><div><span class="tag">Радость</span><span class="tag">Конкретный язык</span><span class="tag">Оптимизм</span><span class="tag">Принципиальная прямота</span></div></header>
<main class="grid">
<section class="panel"><div class="cards">
<article class="card"><small>GDN против activation steering</small><strong class="good">{gdn_activation}</strong><span>балла, 95% CI {gdn_activation_ci}</span></article>
<article class="card"><small>Лучший средний минимум</small><strong>{best_score}</strong><span>Raw rank 1 · joint {best_score_joint}</span></article>
<article class="card"><small>Лучший joint ≥4</small><strong>{best_joint}</strong><span>RSS rank 4 · mean-min {best_joint_score}</span></article>
<article class="card"><small>RSS поверх raw rank 4</small><strong class="good">{rss_gain}</strong><span>балла, 95% CI {rss_gain_ci}</span></article>
</div><p class="muted">Все основные условия оценены на одних и тех же {n} промптах. Expected score и mean-minimum измеряются в баллах Judge 1–5; joint ≥4 — доля ответов, где каждый активный признак получил не менее 4.</p></section>

<section class="panel"><h2>Четыре признака одновременно</h2><table><tr><th>Метод</th><th>Mean minimum ↑</th><th>Все признаки ≥4 ↑</th><th>Качество ответа ↑</th></tr>{method_rows}</table>
<p class="callout"><b>Главное:</b> raw rank 1 даёт лучший непрерывный score, а RSS rank 4 — лучший строгий joint rate. Поэтому единственного победителя нет: выбор зависит от того, важнее средняя выраженность самого слабого признака или одновременное прохождение порога всеми признаками.</p></section>

<section class="panel"><h2>Поддержанные парные сравнения</h2><div>{forest}</div><p class="muted">Точка — средняя разница, линия — 95% paired bootstrap CI, вертикальная линия — отсутствие эффекта. Контраст считается направленным только когда CI не пересекает ноль.</p></section>

<section class="panel"><h2>Что происходит с каждым признаком</h2><div class="profiles">{profile_rows}</div><p class="callout"><b>Интерференция:</b> concrete уже близок к потолку baseline; joy и optimism усиливаются вместе, но семантически частично перекрываются; candor в полной композиции ослабевает, особенно при rank 4. Поэтому текущий joint rate нельзя трактовать как доказательство четырёх независимых управляемых осей.</p></section>

<section class="panel"><h2>Композиция по парам</h2><table><tr><th>Пара признаков</th>{method_headers}</tr>{pair_rows}</table><p class="muted">В ячейке: mean-minimum по активным признакам и строгая доля joint ≥4. Пары позволяют увидеть, где выигрыш общего метода скрывает конфликт конкретных направлений.</p></section>

<section class="panel half"><h2>Что именно даёт RSS и rank</h2><table><tr><th>Контраст</th><th>Δ mean-min</th><th>95% CI</th></tr>
<tr><th>RSS − raw, rank 1</th><td>{rss_r1}</td><td>{rss_r1_ci}</td></tr>
<tr><th>RSS − raw, rank 4</th><td class="good">{rss_gain}</td><td>{rss_gain_ci}</td></tr>
<tr><th>Rank 4 − rank 1, raw</th><td class="bad">{raw_rank}</td><td>{raw_rank_ci}</td></tr>
<tr><th>Rank 4 − rank 1, RSS</th><td>{rss_rank}</td><td>{rss_rank_ci}</td></tr></table>
<p class="callout"><b>Вывод:</b> RSS полезен как стабилизатор перегруженного rank 4, но не даёт подтверждённого выигрыша при rank 1. Сам rank 4 без RSS заметно хуже rank 1.</p></section>

<section class="panel"><h2>Как добавление одного признака меняет другой</h2><div class="profiles">{interference_tables}</div><p class="muted">Строка — добавленный вектор; столбец — уже присутствующий признак, изменение его expected score относительно singleton. Например, ячейка «добавили оптимизм → измеряем прямоту» показывает, насколько оптимизм изменил прямоту. Минус означает ухудшение.</p><p class="callout"><b>Важно:</b> это направленные описательные разницы для пар. В summary нет paired CI для этих 24 отдельных сравнений, поэтому таблица показывает размер и направление интерференции, но не статистическую значимость каждой ячейки.</p></section>

<section class="panel"><h2>Короткие выводы</h2><div class="conclusions">
<article><h3>GDN работает</h3><p>На matched сравнении GDN raw rank 1 превосходит обычный activation steering на <b>{gdn_activation} балла</b>; 95% CI полностью выше нуля.</p></article>
<article><h3>RSS нужен не всегда</h3><p>Подтверждённый выигрыш RSS найден для rank 4: <b>{rss_gain} балла</b>. Для rank 1 результат неопределённый.</p></article>
<article><h3>Композиция частичная</h3><p>Одновременный joint rate растёт с baseline {baseline_joint} до {best_joint}, но четыре независимые оси не доказаны из-за overlap joy/optimism и потери candor.</p></article>
<article><h3>Лучший текущий режим</h3><p>Для основного continuous endpoint — <b>raw rank 1</b>. Если нужен максимальный строгий joint ≥4 — <b>RSS rank 4</b>, но с контролем качества и retention.</p></article>
</div><p class="muted">Judge API для сборки страницы не вызывается. Стоимость использованной итоговой оценки: {judge_cost}.</p></section>
</main></body></html>"""


if __name__ == "__main__":
    main()
