"""Build the self-contained Russian Dashboard 5 page.

The page is deliberately dependency-free: a compact summary is embedded once,
then small HTML/CSS/JS helpers render tables, profiles, heatmaps, and paired
confidence-interval plots.  It does not call the GPU or a Judge provider.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = Path(__file__).with_name("exp5_summary.json")
DEFAULT_OUTPUT = ROOT / "outputs" / "meeting-dashboard-5" / "index.html"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build(summary: dict[str, Any] | None = None) -> str:
    payload = _json(summary) if summary is not None else "null"
    html = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dashboard 5 — five-concept clamp</title>
<style>
:root {{ --bg:#101521; --panel:#182233; --panel2:#202d42; --ink:#ecf2ff; --muted:#9eacc2;
  --line:#33435b; --accent:#8db7ff; --good:#54d19a; --bad:#ff8f8f; --warn:#ffd47c; }}
* {{ box-sizing:border-box }} body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 Inter,system-ui,-apple-system,"Segoe UI",sans-serif }}
main {{ max-width:1500px; margin:0 auto; padding:32px 24px 72px }} h1 {{ font-size:30px; margin:0 0 8px }} h2 {{ margin:34px 0 10px; font-size:22px }} h3 {{ margin:18px 0 8px; font-size:17px }} p {{ margin:7px 0 }} .lead {{ color:var(--muted); max-width:1050px }}
.status {{ border:1px solid var(--line); background:linear-gradient(135deg,#1d2a40,#172134); border-radius:14px; padding:18px 20px; margin:18px 0 20px }}
.status.done {{ border-color:#38765f }} .status.wait {{ border-color:#805c32 }} .status b {{ color:var(--accent) }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px }} .card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px 16px }}
.card .k {{ color:var(--muted); font-size:12px }} .card .v {{ font-size:23px; font-weight:700; margin-top:5px }}
table {{ width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:hidden; margin:10px 0 18px }} th,td {{ border-bottom:1px solid var(--line); padding:9px 10px; text-align:left; vertical-align:top }} th {{ color:#c7d8f5; font-weight:600; background:var(--panel2); position:sticky; top:0 }} tr:last-child td {{ border-bottom:0 }} td.num {{ white-space:nowrap; font-variant-numeric:tabular-nums }} .muted {{ color:var(--muted) }} .small {{ font-size:12px; color:var(--muted) }}
.pill {{ display:inline-block; border:1px solid var(--line); background:#202c40; border-radius:99px; padding:2px 8px; margin:2px 3px 2px 0; font-size:12px }} .positive {{ color:var(--good) }} .negative {{ color:var(--bad) }} .inconclusive {{ color:var(--warn) }}
.forest {{ background:var(--panel); border:1px solid var(--line); padding:10px 12px; border-radius:10px; margin:10px 0 18px }} .forest-row {{ display:grid; grid-template-columns:minmax(240px,1fr) 1.5fr 125px; gap:12px; align-items:center; min-height:34px; border-bottom:1px solid #2b3950 }} .forest-row:last-child {{ border:0 }} .forest-label {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap }} .axis {{ position:relative; height:20px; background:linear-gradient(#2a3952,#2a3952) center/100% 1px no-repeat }} .axis::after {{ content:""; position:absolute; left:50%; top:0; bottom:0; border-left:1px dashed #d8e2f6aa }} .ci {{ position:absolute; top:9px; height:2px; background:var(--accent) }} .dot {{ position:absolute; top:5px; width:10px; height:10px; margin-left:-5px; border-radius:50%; background:var(--accent); border:2px solid #eff5ff }}
.profile {{ display:grid; grid-template-columns:190px repeat(5,minmax(100px,1fr)); gap:8px; align-items:center; background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px; margin:10px 0 18px }} .profile .head {{ color:#c7d8f5; font-size:12px }} .barbox {{ height:22px; background:#29364b; border-radius:5px; overflow:hidden; position:relative }} .bar {{ height:100%; background:linear-gradient(90deg,#5f91ee,#9bbdff); min-width:1px }} .bartext {{ position:absolute; left:6px; top:1px; font-size:12px; color:#f5f8ff }}
.heatmap td {{ text-align:center; min-width:100px; font-variant-numeric:tabular-nums }} .heat-pos {{ background:#245b4a }} .heat-neg {{ background:#6b3039 }} .heat-zero {{ background:#2a374b }}
.note {{ background:#172235; border-left:3px solid var(--accent); padding:11px 14px; margin:9px 0; border-radius:4px }} ul {{ margin:7px 0 7px 22px }} .missing {{ color:var(--warn); font-style:italic }} code {{ color:#c8dcff }} .foot {{ color:var(--muted); font-size:12px; margin-top:30px }}
@media(max-width:800px) {{ main {{ padding:22px 12px 50px }} .forest-row {{ grid-template-columns:1fr; gap:3px; padding:6px 0 }} .profile {{ grid-template-columns:130px repeat(2,1fr) }} .profile .head:nth-child(n+4) {{ display:none }} }}
</style>
</head>
<body><main id="app"></main>
<script>
const SUMMARY = {payload};
const FEATURES = ["french_language","concrete_language","optimism","first_person_voice","bulleted_layout"];
const LABELS = {{french_language:"Французский язык", concrete_language:"Конкретный язык", optimism:"Оптимизм", first_person_voice:"Первое лицо", bulleted_layout:"Маркированная структура"}};
const SHORT = {{french_language:"Французский", concrete_language:"Конкретность", optimism:"Оптимизм", first_person_voice:"Первое лицо", bulleted_layout:"Буллеты"}};
const METHOD = {{baseline:"Бейзлайн", full_add_raw_r1:"GDN raw add r1", full_add_rss_r1:"GDN RSS add r1", full_add_rss_r4:"GDN RSS add r4", full_clamp_raw_r1:"GDN raw clamp r1", full_clamp_rss_r1:"GDN RSS clamp r1", full_clamp_rss_r4:"GDN RSS clamp r4", add:"RSS add r1", clamp:"RSS clamp r1", singleton:"Одиночный", loo_add:"LOO add", loo_clamp:"LOO clamp", flip:"Flip-one"}};
const esc = value => String(value ?? "—").replace(/[&<>]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;"}}[c]));
const num = (value, digits=2) => value == null || !Number.isFinite(Number(value)) ? "—" : Number(value).toFixed(digits);
const metric = m => m && m.mean != null ? Number(m.mean) : null;
function range(m, digits=2) {{ return !m || m.mean == null ? "—" : `${{num(m.mean,digits)}} [${{num(m.ci95_low,digits)}; ${{num(m.ci95_high,digits)}}]`; }}
function score(m) {{ return range(m,2) + " балла"; }}
function rate(m) {{ return !m || m.mean == null ? "—" : `${{num(100*m.mean,1)}}% [${{num(100*m.ci95_low,1)}%; ${{num(100*m.ci95_high,1)}}%]`; }}
function pp(m) {{ return !m || m.mean == null ? "—" : `${{num(100*m.mean,1)}} п.п. [${{num(100*m.ci95_low,1)}; ${{num(100*m.ci95_high,1)}}]`; }}
function status(m) {{ if(!m || m.mean == null) return "missing"; if(m.ci95_low > 0) return "positive"; if(m.ci95_high < 0) return "negative"; return "inconclusive"; }}
function cls(m) {{ return status(m); }}
function verdict(m) {{ return status(m)==="positive"?"поддержан CI":status(m)==="negative"?"отрицательный CI":status(m)==="inconclusive"?"неопределённый CI":"нет данных"; }}
function featureLabel(f) {{ return (SUMMARY.features && SUMMARY.features[f] && SUMMARY.features[f].label) || LABELS[f] || f; }}
function methodLabel(x) {{ return METHOD[x] || x; }}
function pills(fs) {{ return (fs||[]).map(f=>`<span class="pill">${{esc(featureLabel(f))}}</span>`).join(""); }}
function fmtN(n) {{ return n == null ? "—" : Number(n).toLocaleString("ru-RU"); }}
function forest(rows, scale=1) {{
  if(!rows.length) return `<div class="forest"><span class="missing">Нет контрастов в summary.</span></div>`;
  const extent = Math.max(scale, ...rows.flatMap(r=>[Math.abs(Number(r.metric?.ci95_low||0)),Math.abs(Number(r.metric?.ci95_high||0))]));
  const pos = value => Math.max(0,Math.min(100,50+50*Number(value)/extent));
  return `<div class="forest">${{rows.map(r=>{{const m=r.metric; const lo=pos(m?.ci95_low||0), hi=pos(m?.ci95_high||0), dot=pos(m?.mean||0); return `<div class="forest-row"><div class="forest-label">${{esc(r.label)}}<div class="small">${{esc(verdict(m))}}</div></div><div class="axis"><span class="ci" style="left:${{lo}}%;width:${{Math.max(0,hi-lo)}}%"></span><span class="dot ${{cls(m)}}" style="left:${{dot}}%"></span></div><div class="num ${{cls(m)}}">${{range(m,2)}}</div></div>`}}).join("")}}</div>`;
}}
function table(headers, rows, cls="") {{ return `<table class="${{cls}}"><thead><tr>${{headers.map(h=>`<th>${{h}}</th>`).join("")}}</tr></thead><tbody>${{rows.join("")}}</tbody></table>`; }}
function renderPending() {{
  return `<h1>Dashboard 5 — five-concept clamp</h1><div class="status wait"><b>Эксперимент ещё не завершён</b><p>Компактный summary отсутствует. Страница не подменяет отсутствующие блоки нулевыми значениями.</p></div><div class="card"><h3>План обработки</h3><p>После завершения Judge собрать baseline, одиночные признаки, пары RSS add/clamp, полные композиции, LOO и flip-one. Затем построить парные 95% bootstrap CI и отдельную оценку качества.</p></div>`;
}}
function renderSummary(S) {{
  const baseline=S.baseline; const full=S.full_five||[]; const single=S.singletons||[]; const pairs=S.pairs||[];
  const fullBest=full.find(x=>x.condition==="full_add_rss_r1") || full[0];
  const jointBaseline=baseline?.all_active_ge4; const jointFull=fullBest?.all_active_ge4;
  const strongest=[...single].sort((a,b)=>(metric(b.features?.[b.active_features?.[0]]?.delta_expected_vs_baseline)||-99)-(metric(a.features?.[a.active_features?.[0]]?.delta_expected_vs_baseline)||-99))[0];
  const cards=[
    ["Размер задачи",`${{fmtN(S.experiment?.prompts)}} промптов · ${{fmtN(S.experiment?.conditions)}} условий`],
    ["Judge",`${{fmtN(S.judge_usage?.judgments)}} оценок · ${{num(S.judge_usage?.estimated_usd,2)}} $`],
    ["Joint all-five",`${{rate(jointBaseline)}} → ${{rate(jointFull)}}`],
    ["Лучший одиночный сигнал",strongest?`${{featureLabel(strongest.active_features[0])}}: ${{score(strongest.features[strongest.active_features[0]].expected)}}`:"—"]
  ];
  let html=`<h1>Dashboard 5 — пять признаков и clamp</h1><p class="lead">Финальный разбор завершённого Exp5: Qwen3.5-9B, одна и та же Judge-оценка для всех условий. Числа ниже вычислены из компактного summary; большие генерации и raw Judge-файлы не встраиваются.</p><div class="status done"><b>Результаты загружены</b><p>Проверяем не только силу каждого признака, но и сохранение остальных признаков при композиции. Основной endpoint — целочисленный score и доля score ≥4; expected score показан как более чувствительный вторичный endpoint.</p></div><div class="grid">${{cards.map(c=>`<div class="card"><div class="k">${{c[0]}}</div><div class="v">${{esc(c[1])}}</div></div>`).join("")}}</div>`;
  html+=`<h2>1. Что именно посчитано</h2><div class="card"><p><b>Признаки:</b> ${{pills(S.experiment?.features||FEATURES)}}</p><p><b>Условия:</b> ${{fmtN(S.experiment?.conditions)}} (baseline, 5 singleton, 6 full-five, LOO, flip-one и 10 пар в двух вариантах).</p><p><b>Не смешиваем n:</b> trait n = ${{fmtN(S.trait_n)}}; quality n = ${{fmtN(S.quality_n)}}. Для каждого контраста используется paired bootstrap (${{fmtN(S.experiment?.bootstrap_reps)}} повторов, seed ${{S.experiment?.bootstrap_seed}}).</p></div>`;
  html+=`<h2>2. Жизнеспособность признаков: одиночные steering-векторы</h2><p class="lead">Дельта expected score относительно baseline — в баллах шкалы 1–5. Дельта p(score≥4) — в процентных пунктах. CI пересекает ноль — результат не считаем доказанным.</p>`;
  html+=table(["Признак","Одиночный expected","Δ expected","p(score≥4)","Δ p4","Качество Δ"], single.map(r=>{const f=r.active_features?.[0],x=r.features?.[f]; return `<tr><td><b>${{esc(featureLabel(f))}}</b><div class="small">против ${{esc(S.features?.[f]?.opposite||"")}}</div></td><td class="num">${{score(x?.expected)}}</td><td class="num ${{cls(x?.delta_expected_vs_baseline)}}">${{score(x?.delta_expected_vs_baseline)}}</td><td class="num">${{rate(x?.p_ge4)}}</td><td class="num ${{cls(x?.delta_p_ge4_vs_baseline)}}">${{pp(x?.delta_p_ge4_vs_baseline)}}</td><td class="num ${{cls(r.delta_quality_vs_baseline)}}">${{score(r.delta_quality_vs_baseline)}}</td></tr>`}));
  html+=forest(single.map(r=>{const f=r.active_features?.[0];return {{label:featureLabel(f),metric:r.features?.[f]?.delta_expected_vs_baseline}}}),.8);
  html+=`<h2>3. Полная композиция из пяти</h2><p class="lead">Это не доказательство независимой композициональности само по себе. Joint rate — описательное evidence: одновременно score≥4 у всех пяти. Смотрим также профиль каждого признака и качество.</p>`;
  html+=`<div class="profile"><div class="head">Метод</div>${{FEATURES.map(f=>`<div class="head">${{esc(SHORT[f])}}</div>`).join("")}}${{full.map(r=>`<div><b>${{esc(methodLabel(r.method))}}</b><div class="small">joint ${{rate(r.all_active_ge4)}}</div></div>${{FEATURES.map(f=>{const m=r.features?.[f]?.expected; const value=Math.max(0,Math.min(5,metric(m)||0)); return `<div class="barbox" title="${{esc(featureLabel(f))}}: ${{score(m)}}"><div class="bar" style="width:${{20*value}}%"></div><span class="bartext">${{num(metric(m),2)}}</span></div>`}).join("")}}`).join("")}}</div>`;
  html+=table(["Метод","Joint all-five","Δ joint","min expected","Δ min","Quality Δ"],full.map(r=>`<tr><td><b>${{esc(methodLabel(r.method))}}</b><div class="small">${{esc(r.condition)}}</div></td><td class="num">${{rate(r.all_active_ge4)}}</td><td class="num ${{cls(r.delta_all_active_ge4_vs_baseline)}}">${{pp(r.delta_all_active_ge4_vs_baseline)}}</td><td class="num">${{score(r.mean_minimum_expected)}}</td><td class="num ${{cls(r.delta_mean_minimum_expected_vs_baseline)}}">${{score(r.delta_mean_minimum_expected_vs_baseline)}}</td><td class="num ${{cls(r.delta_quality_vs_baseline)}}">${{score(r.delta_quality_vs_baseline)}}</td></tr>`));
  html+=table(["Метод","Признак","Expected","Δ expected","p(score≥4)","Δ p4"],full.flatMap(r=>FEATURES.map(f=>{const x=r.features?.[f]; return `<tr><td>${{esc(methodLabel(r.method))}}</td><td>${{esc(featureLabel(f))}}</td><td class="num">${{score(x?.expected)}}</td><td class="num ${{cls(x?.delta_expected_vs_baseline)}}">${{score(x?.delta_expected_vs_baseline)}}</td><td class="num">${{rate(x?.p_ge4)}}</td><td class="num ${{cls(x?.delta_p_ge4_vs_baseline)}}">${{pp(x?.delta_p_ge4_vs_baseline)}}</td></tr>`})));
  html+=`<h2>4. Interference: как одиночный признак меняет остальные</h2><p class="lead">В ячейке — expected-score delta одиночного условия относительно baseline. Диагональ показывает собственную цель; вне диагонали — off-target interference.</p><table class="heatmap"><thead><tr><th>Одиночный \\ оцениваемый</th>${{FEATURES.map(f=>`<th>${{esc(SHORT[f])}}</th>`).join("")}}</tr></thead><tbody>${{single.map(r=>{const active=r.active_features?.[0]; return `<tr><th>${{esc(SHORT[active])}}</th>${{FEATURES.map(f=>{const m=r.features?.[f]?.delta_expected_vs_baseline, v=metric(m); const clsx=v==null?"heat-zero":v>0.05?"heat-pos":v<-.05?"heat-neg":"heat-zero"; return `<td class="${{clsx}} ${{cls(m)}}">${{score(m)}}</td>`}).join("")}}</tr>`}).join("")}}</tbody></table>`;
  html+=`<h2>5. Все десять пар: RSS add против RSS clamp</h2><p class="lead">Пара проверяется двумя способами. Clamp-add — paired contrast, а не «процент правильных ответов»: он показывает изменение clamp относительно add.</p>`;
  html+=table(["Пара","Метод","Joint","Δ joint","Δ min","Quality Δ"],pairs.map(r=>`<tr><td><b>${{esc(r.pair)}}</b><div class="small">${{pills(r.features)}}</div></td><td>${{esc(methodLabel(r.method))}}</td><td class="num">${{rate(r.joint)}}</td><td class="num ${{cls(r.delta_joint_vs_baseline)}}">${{pp(r.delta_joint_vs_baseline)}}</td><td class="num">${{score(r.delta_mean_minimum_expected_vs_baseline)}}</td><td class="num ${{cls(r.delta_quality_vs_baseline)}}">${{score(r.delta_quality_vs_baseline)}}</td></tr>`));
  html+=`<div class="note missing">В Exp5 для пар сгенерированы только RSS add r1 и RSS clamp r1. Контрасты RSS−raw и rank4−rank1 на уровне пар отсутствуют и не подменяются нулём; rank4-контрасты доступны только для full-five.</div>`;
  const pairContrasts=(S.contrasts||[]).filter(x=>String(x.contrast).startsWith("clamp_minus_add:"));
  html+=`<h3>Парные clamp − add</h3>${{forest(pairContrasts.map(x=>({{label:x.pair,metric:x.delta_mean_minimum_expected}})),.35)}}`;
  html+=`<h2>6. Сводка технических абляций</h2><p class="lead">Лесной график использует дельту minimum expected. Нулевая линия — отсутствие различия; качество показано рядом в таблице.</p>`;
  const mainContrasts=(S.contrasts||[]).filter(x=>!String(x.contrast).startsWith("clamp_minus_add:"));
  html+=forest(mainContrasts.map(x=>({{label:x.contrast,metric:x.delta_mean_minimum_expected}})),.15);
  html+=table(["Контраст","Δ min expected","Δ joint","Δ quality","Интерпретация"],mainContrasts.map(x=>`<tr><td><b>${{esc(x.contrast)}}</b><div class="small">${{esc(x.left)}} − ${{esc(x.right)}} · n=${{fmtN(x.n)}}</div></td><td class="num ${{cls(x.delta_mean_minimum_expected)}}">${{score(x.delta_mean_minimum_expected)}}</td><td class="num ${{cls(x.delta_all_active_ge4)}}">${{pp(x.delta_all_active_ge4)}}</td><td class="num ${{cls(x.delta_quality)}}">${{score(x.delta_quality)}}</td><td class="${{cls(x.delta_quality)}}">${{esc(verdict(x.delta_quality))}}</td></tr>`));
  html+=`<h2>7. LOO и flip-one</h2><p class="lead">LOO убирает один компонент из полной композиции; flip-one меняет знак одного направления. Это проверка доминирования и совместимости, а не новая модель причинности.</p>`;
  html+=table(["Условие","Активные признаки","Joint","min expected","Quality Δ"],(S.loo||[]).concat(S.flip_one||[]).map(r=>`<tr><td><b>${{esc(r.condition)}}</b></td><td>${{pills(r.active_features)}}</td><td class="num">${{rate(r.all_active_ge4)}}</td><td class="num">${{score(r.mean_minimum_expected)}}</td><td class="num ${{cls(r.delta_quality_vs_baseline)}}">${{score(r.delta_quality_vs_baseline)}}</td></tr>`));
  html+=`<h2>8. Retention: full composition − singleton</h2>`;
  if((S.retention||[]).length) html+=table(["Метод","Признак","Full − singleton expected","Качество контраста"],S.retention.map(r=>`<tr><td>${{esc(methodLabel(r.method))}}</td><td>${{esc(featureLabel(r.feature))}}</td><td class="num ${{cls(r.full_minus_singleton)}}">${{score(r.full_minus_singleton)}}</td><td class="num">${{score(r.delta_quality)}}</td></tr>`));
  else html+=`<div class="note missing">Retention не записан в summary; не выводим его как нулевой.</div>`;
  html+=`<h2>9. Deterministic sanity checks</h2>`;
  const sanity=S.deterministic_sanity;
  if(sanity) html+=table(["Проверка","Результат"],[`<tr><td>Baseline: маркеры списка</td><td class="num">${{rate(sanity.baseline_bullet_rate)}}</td></tr>`,`<tr><td>Baseline: первое лицо</td><td class="num">${{rate(sanity.baseline_first_person_rate)}}</td></tr>`]);
  else html+=`<div class="note missing">Текстовые sanity-check не включены в summary. Judge API для их добавления не нужен, но исходные генерации должны быть доступны сборщику.</div>`;
  html+=`<h2>10. Что можно утверждать</h2><div class="note"><b>Поддержано:</b> одиночные сигналы можно сравнить с baseline; французский и оптимизм дают разные по силе изменения, а concrete имеет высокий baseline ceiling. Off-target таблица показывает, какие направления проседают при добавлении.</div><div class="note"><b>Не доказано:</b> что пять признаков складываются независимо. Joint all-five не растёт; пары с высоким joint частично наследуют уже высокий baseline. Clamp сам по себе не становится доказанно лучше add, а rank4-контрасты нужно трактовать вместе с quality.</div><div class="note"><b>Следующий честный шаг:</b> сначала калибровать новые признаки, затем повторить только независимые оси с тем же blind Judge, заранее зафиксировав primary endpoint и paired CI. Не делать вывод о композициональности по одной доле joint.</div>`;
  html+=`<h2>11. Provenance и ограничения</h2><div class="grid"><div class="card"><div class="k">Модель Judge</div><div class="v">${{esc(S.judge_usage?.judge_model)}}</div><p class="small">${{esc(S.judge_usage?.prompt_version)}} · rubric ${{esc(S.judge_usage?.rubric_version)}} · config ${{esc(S.judge_usage?.config_version)}}</p></div><div class="card"><div class="k">Токены / оценка</div><div class="v">${{fmtN(S.judge_usage?.input_tokens)}} + ${{fmtN(S.judge_usage?.output_tokens)}}</div><p class="small">estimated ${{num(S.judge_usage?.estimated_usd,4)}} $; reasoning ${{fmtN(S.judge_usage?.reasoning_tokens)}}</p></div><div class="card"><div class="k">Selection</div><div class="v">${{S.selection?.status?esc(S.selection.status):"missing"}}</div><p class="small">scale=${{S.selection?.scale??"missing"}}, beta=${{S.selection?.beta??"missing"}}; selected λ=${{S.selection?.selected_lambda??"missing"}}. Dev selection не выдумываем.</p></div></div><p class="foot">Сборка: data-driven summary · paired bootstrap CI · primary integer trait_score, secondary expected_score. Отсутствующие блоки помечены missing.</p>`;
  return html;
}}
document.getElementById("app").innerHTML = SUMMARY ? renderSummary(SUMMARY) : renderPending();
</script></body></html>"""
    # The template is plain HTML rather than a Python f-string.  Doubling
    # braces keeps the source readable alongside the previous dashboards; the
    # final pass restores CSS/JavaScript braces and injects the JSON once.
    return html.replace("{{", "{").replace("}}", "}").replace("{payload}", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = None
    if args.summary.exists():
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build(summary), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
