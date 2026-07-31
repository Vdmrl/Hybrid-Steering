"""Build the self-contained Russian dashboard for Experiment 4."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/meeting-dashboard-4/index.html"
SUMMARY = ROOT / "outputs/strong-composition-exp4/summary.json"


def build(summary: dict | None = None) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    data = json.dumps(
        {"summary": summary}, ensure_ascii=False, separators=(",", ":")
    ).replace("</", "<\\/")
    return HTML.replace("__GENERATED__", generated).replace("__DATA__", data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = None
    if args.summary.exists():
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
    args.output.write_text(build(summary), encoding="utf-8")
    print(args.output)


HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hybrid Steering — Experiment 4</title>
<style>
:root{--bg:#09111c;--panel:#111d2c;--panel2:#0d1724;--line:#2a3b52;--text:#edf4ff;--muted:#9fb0c4;--cyan:#64ddd4;--blue:#8794ff;--green:#5ee09b;--amber:#f2bd63;--red:#ff8585}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 Inter,system-ui,sans-serif}header,main{max-width:1400px;margin:auto;padding:24px}header{padding-top:36px}h1{font-size:36px;line-height:1.08;margin:6px 0 12px}h2{font-size:20px;margin:0 0 8px}h3{font-size:15px;margin:0 0 7px}.eyebrow{color:var(--cyan);font-weight:800;letter-spacing:.08em;text-transform:uppercase}.muted,.note{color:var(--muted)}.note{font-size:13px}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}.panel{grid-column:span 12;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px;overflow:auto}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:16px 0}.card{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:13px}.card strong{display:block;font-size:25px;margin:4px 0}.good{color:var(--green)}.warn{color:var(--amber)}.bad{color:var(--red)}table{width:100%;border-collapse:collapse;margin-top:14px}th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{color:#c9d6e7;font-size:12px;white-space:nowrap}tr:last-child td,tr:last-child th{border-bottom:0}.callout{border-left:3px solid var(--cyan);padding-left:12px;margin:14px 0}.callout.warn{border-color:var(--amber)}.callout.bad{border-color:var(--red)}.formula{font:13px ui-monospace,SFMono-Regular,Consolas,monospace;background:#08101a;border:1px solid var(--line);padding:12px;border-radius:8px;white-space:pre-wrap}.timeline{display:grid;grid-template-columns:repeat(7,1fr);gap:8px;margin-top:14px}.step{background:var(--panel2);border-top:3px solid var(--cyan);padding:11px;border-radius:8px}.step small{display:block;color:var(--muted);margin-top:5px}.insights{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.insight{background:var(--panel2);border-radius:10px;padding:13px;border-top:3px solid var(--cyan)}.insight.warn{border-top-color:var(--amber)}.source{font-size:12px;color:var(--muted);margin-top:16px}.mono{font:12px ui-monospace,SFMono-Regular,Consolas,monospace}.tag{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:2px 8px;color:var(--muted);font-size:11px;margin:2px}.status{display:inline-block;border-radius:999px;padding:2px 8px;font-size:11px;font-weight:700}.status.positive{color:var(--green);border:1px solid var(--green)}.status.inconclusive{color:var(--amber);border:1px solid var(--amber)}.status.negative{color:var(--red);border:1px solid var(--red)}.status.missing{color:var(--muted);border:1px solid var(--line)}.heat{min-width:150px;text-align:center}.heat small{display:block;color:#d7e4f5}.viz-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:16px}.viz-card{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:13px}.bar-row{display:grid;grid-template-columns:125px 1fr 48px;gap:7px;align-items:center;margin:5px 0}.bar-label{font-size:11px;color:var(--muted)}.bar-track{height:12px;background:#08101a;border-radius:99px;overflow:hidden}.bar-fill{height:100%;border-radius:99px}.bar-fill.base{background:#6b7b91}.bar-fill.raw{background:var(--blue)}.bar-fill.rss{background:var(--green)}.forest-row{display:grid;grid-template-columns:170px 1fr 82px;gap:8px;align-items:center;margin:10px 0}.forest-label{font-size:11px;color:var(--muted)}.forest-track{height:18px;position:relative;background:#08101a;border-radius:99px}.forest-zero{position:absolute;left:50%;top:0;height:100%;border-left:1px dashed var(--amber)}.forest-ci{position:absolute;top:7px;height:4px;background:var(--cyan);border-radius:99px}.forest-dot{position:absolute;top:3px;width:12px;height:12px;margin-left:-6px;background:var(--green);border-radius:50%;border:2px solid var(--panel2)}
@media(max-width:1050px){.cards,.insights,.viz-grid{grid-template-columns:1fr 1fr}.timeline{grid-template-columns:repeat(4,1fr)}}@media(max-width:600px){header,main{padding:16px}.cards,.insights,.viz-grid,.timeline{grid-template-columns:1fr}h1{font-size:29px}.bar-row{grid-template-columns:95px 1fr 42px}.forest-row{grid-template-columns:115px 1fr 68px}}
</style></head>
<body><header>
<div class="eyebrow">Hybrid Steering · Experiment 4 · Qwen3.5-9B</div>
<h1>Strong composition: rank × RSS-нормализация</h1>
<p class="muted" id="status">Это план эксперимента; итоговые результаты появятся после Judge. Построено __GENERATED__.</p>
<p class="callout">Главный вопрос: помогает ли дополнительная ёмкость направления и RSS-нормализация сохранить несколько поведенческих признаков одновременно, не ухудшая качество ответа?</p>
</header><main class="grid">
<section class="panel" id="why"><h2>Зачем нужен Exp4 после Exp3</h2><p>Exp3 дал сигнал для GDN raw rank 1 и RSS rank 4, но не закрыл raw rank 4 и не показал весь факториальный набор одним компактным отчётом. Exp4 сравнивает четыре метода на одинаковых prompt_id.</p><table><tr><th>Пробел Exp3</th><th>Что закрывает Exp4</th></tr><tr><td>Не разделены rank и RSS.</td><td>Raw/RSS для rank 1 и rank 4.</td></tr><tr><td>Не видна потеря отдельного признака в композиции.</td><td>Singleton retention и feature-level deltas.</td></tr><tr><td>Слабо видны взаимодействия.</td><td>Все шесть пар и all-four.</td></tr></table></section>
<section class="panel" id="features"></section>
<section class="panel" id="methods"><h2>Четыре сравниваемых метода</h2><table><tr><th>Метод</th><th>Математика</th><th>Вопрос</th></tr><tr><th>GDN raw, rank 1</th><td>Складываем rank-1 направления активных признаков.</td><td>Работает ли компактная композиция?</td></tr><tr><th>GDN RSS, rank 1</th><td>Складываем направления и нормализуем итог по RSS.</td><td>Помогает ли контроль нормы?</td></tr><tr><th>GDN raw, rank 4</th><td>Складываем rank-4 направления напрямую.</td><td>Нужна ли дополнительная ёмкость?</td></tr><tr><th>GDN RSS, rank 4</th><td>Складываем rank-4 направления и нормализуем итог.</td><td>Есть ли совместный выигрыш rank + RSS?</td></tr></table></section>
<section class="panel" id="matrix"><h2>Матрица запуска</h2><table><tr><th>Блок</th><th>Условия</th><th>Методы</th><th>Ответов</th><th>Назначение</th></tr><tr><th>All-four</th><td>1111</td><td>4</td><td>512</td><td>Главный endpoint: все четыре режима одновременно.</td></tr><tr><th>Singletons</th><td>0001, 0010, 0100, 1000</td><td>raw rank1 + raw rank4</td><td>1024</td><td>Retention: не ломается ли отдельный признак в полном составе?</td></tr><tr><th>All pairs</th><td>0011, 0101, 0110, 1001, 1010, 1100</td><td>4</td><td>3072</td><td>Основной тест взаимодействий.</td></tr><tr><th>Итого main</th><td>36 условий</td><td>4 метода там, где нужно</td><td><b>4608</b></td><td>Одинаковые 128 test prompts.</td></tr></table><p class="note">Биты: 0001 = joy, 0010 = concrete, 0100 = optimism, 1000 = candor. Подписи пар берутся из active_features в summary.</p></section>
<section class="panel" id="metrics"><h2>Метрики и единицы</h2><div class="cards"><div class="card"><b>Expected score</b><strong>1–5 баллов</strong><small>Вероятностно взвешенная оценка каждого активного признака.</small></div><div class="card"><b>Mean minimum</b><strong>1–5 баллов</strong><small>Средний минимум среди активных признаков в ответе.</small></div><div class="card"><b>Joint ≥4</b><strong>0–100%</strong><small>Доля ответов, где каждый активный признак получил не менее 4.</small></div><div class="card"><b>Quality</b><strong>1–5 баллов</strong><small>Отдельная оценка качества ответа; не смешивается с trait score.</small></div></div><p>Основные сравнения — paired bootstrap по одинаковым prompt_id. Для score показываем баллы, для долей — проценты и процентные пункты.</p><p class="note">Quality в текущем run.yaml оценивается на 32 ответах, trait-метрики — на 128. Dashboard показывает эти n раздельно.</p></section>
<section class="panel" id="queue"><h2>Порядок автономной очереди</h2><div class="timeline"><div class="step"><b>1. Self-test</b><small>Контракт и формулы.</small></div><div class="step"><b>2. GPU smoke</b><small>Один prompt, два режима.</small></div><div class="step"><b>3. Dev</b><small>Выбор λ на 32 prompts.</small></div><div class="step"><b>4. All-four</b><small>4 метода × 128.</small></div><div class="step"><b>5. Singletons</b><small>rank1/rank4.</small></div><div class="step"><b>6. Pairs</b><small>6 пар × 4 метода.</small></div><div class="step"><b>7. Judge + summary</b><small>CI, retention, report.</small></div></div><p class="callout" id="queue-status">Без summary здесь только план; Dashboard не запускает GPU или Judge.</p></section>
<section class="panel" id="results"></section>
<section class="panel" id="interpret"></section>
<section class="panel" id="limits"><h2>Что этот Exp4 всё ещё не проверяет</h2><table><tr><th>Вопрос</th><th>Статус</th><th>Почему отдельно</th></tr><tr><th>Свой orthogonal rank slot каждому признаку</th><td class="warn">Не входит</td><td>Это уже не простое сложение SVD-направлений, а другой протокол размещения компонент.</td></tr><tr><th>Заморозка или clamp отдельных компонент</th><td class="warn">Не входит</td><td>Сначала нужно измерить baseline composition и retention.</td></tr><tr><th>Классический activation steering</th><td class="warn">Не входит</td><td>Сравнение лучше делать отдельным matched-strength run на тех же парах.</td></tr><tr><th>French и новые признаки</th><td class="warn">Не входит</td><td>Exp4 изолирует rank × RSS на фиксированных четырёх признаках.</td></tr><tr><th>256 prompts</th><td class="warn">Не входит</td><td>Сначала закрываем методическую абляцию на 128 prompts.</td></tr></table><p class="source">Большие генерации и Judge artifacts остаются на сервере; dashboard читает только компактный summary.</p></section>
</main>
<script>
const featureNames={joy:'Joy — радость',concrete:'Concrete language — конкретный язык',optimism:'Optimism — оптимизм',candor:'Principled candor — принципиальная прямота'};
const FEATURES=['joy','concrete','optimism','candor'];
const DATA=__DATA__, RESULT=DATA.summary;
const condition=id=>RESULT?.conditions?.find(item=>item.condition===id);
const allConditions=()=>RESULT?.conditions||[];
const num=value=>Number.isFinite(Number(value))?Number(value):null;
const score=value=>num(value)===null?'—':num(value).toFixed(3);
const pct=value=>num(value)===null?'—':`${(num(value)*100).toFixed(1)}%`;
const pp=value=>num(value)===null?'—':`${(num(value)*100).toFixed(1)} п.п.`;
const scoreDelta=value=>num(value)===null?'—':`${num(value)>=0?'+':''}${num(value).toFixed(3)} балла`;
const scoreCI=value=>value?`[${score(value.ci95_low)}, ${score(value.ci95_high)}]`:'—';
const rateCI=value=>value?`[${pct(value.ci95_low)}, ${pct(value.ci95_high)}]`:'—';
const rateDeltaCI=value=>value?`[${pp(value.ci95_low)}, ${pp(value.ci95_high)}]`:'—';
const featureLabel=key=>featureNames[key]||key;
const featureList=item=>item?.active_features||[];
const methods=['gdn_raw_r1','gdn_rss_r1','gdn_raw_r4','gdn_rss_r4'];
const methodNames={gdn_raw_r1:'GDN raw, rank 1',gdn_rss_r1:'GDN RSS, rank 1',gdn_raw_r4:'GDN raw, rank 4',gdn_rss_r4:'GDN RSS, rank 4'};
const methodName=id=>methodNames[id]||id;
const allFourIds=methods.map(method=>`${method}_1111`);
const pairMasks=['0011','0101','0110','1001','1010','1100'];
const contrastNames=['rss_minus_raw_rank1','rss_minus_raw_rank4','rank4_minus_rank1_raw','rank4_minus_rank1_rss'];
const contrastLabels={rss_minus_raw_rank1:'RSS − raw, rank 1',rss_minus_raw_rank4:'RSS − raw, rank 4',rank4_minus_rank1_raw:'raw rank 4 − raw rank 1',rank4_minus_rank1_rss:'RSS rank 4 − RSS rank 1'};
const contrast=(name,mask)=>RESULT?.contrasts?.find(item=>item.contrast===name&&item.mask===mask);
const status=delta=>!delta?'missing':delta.ci95_low>0?'positive':delta.ci95_high<0?'negative':'inconclusive';
const statusClass=value=>`status ${status(value)}`;
const heatStyle=value=>{const opacity=Math.max(.08,Math.min(.75,(num(value)-2)/3*.75));return `background:rgba(94,224,155,${opacity})`};
const barWidth=value=>`${Math.max(0,Math.min(100,num(value)/5*100))}%`;
const forestLeft=value=>`${Math.max(0,Math.min(100,(num(value)+.2)/.4*100))}%`;
const forestWidth=(low,high)=>`${Math.max(0,Math.min(100,(num(high)-num(low))/.4*100))}%`;
const forestRow=(label,d)=>`<div class="forest-row"><span class="forest-label">${label}</span><div class="forest-track"><span class="forest-zero"></span><span class="forest-ci" style="left:${forestLeft(d.ci95_low)};width:${forestWidth(d.ci95_low,d.ci95_high)}"></span><span class="forest-dot" style="left:${forestLeft(d.mean)}"></span></div><span class="mono">${scoreDelta(d.mean)}</span></div>`;
const qualityN=RESULT?.quality_n??null;
const qualityNText=qualityN===null?'missing':qualityN;
const conditionFor=(method,mask)=>condition(`${method}_${mask}`);
const pairLabel=mask=>{const item=conditionFor('gdn_raw_r1',mask)||conditionFor('gdn_rss_r4',mask);return item?featureList(item).map(featureLabel).join(' + '):`mask ${mask}`};

document.querySelector('#status').textContent=RESULT?`Результаты загружены: ${allConditions().length} условий, ${(RESULT.contrasts||[]).length} контрастов. Построено __GENERATED__.`:`Это план эксперимента; итоговые результаты появятся после Judge. Построено __GENERATED__.`;
document.querySelector('#queue-status').innerHTML=RESULT?`<b>Summary загружен.</b> Готовые блоки показаны ниже; отсутствующие условия помечены как missing. Dashboard не запускает новые вызовы.`:`<b>Результаты ещё не подключены.</b> Очередь и Judge должны завершить свои блоки, после чего summary можно вставить через команду из README.`;
document.querySelector('#features').innerHTML=`<h2>Признаки и битовые маски</h2><p class="note">В коде бит 0 = joy, бит 1 = concrete, бит 2 = optimism, бит 3 = candor: <b>0001 = joy, 0010 = concrete, 0100 = optimism, 1000 = candor</b>. Для подписей результатов используется active_features из summary.</p><div class="cards">${Object.entries(featureNames).map(([key,value])=>`<div class="card"><b>${value.split(' — ')[0]}</b><strong>${value.split(' — ')[1]}</strong><small>Отдельное направление GDN для композиции.</small></div>`).join('')}</div><p class="note">Сильные quality-safe alpha: joy=4, concrete=4, optimism=8, candor=8. Общий λ выбирается только на dev из {0.5, 0.75, 1.0}.</p>`;

function conditionRow(id){
  const item=condition(id);
  if(!item)return `<tr><th>${methodNames[id.replace('_1111','')]||id}</th><td colspan="8" class="muted">missing</td></tr>`;
  const delta=item.delta_all_active_ge4_vs_baseline;
  return `<tr><th>${id==='baseline'?'Baseline':methodName(id.replace('_1111',''))}</th><td>${score(item.mean_minimum_expected?.mean)}<br><span class="muted">${scoreCI(item.mean_minimum_expected)}</span></td><td>${pct(item.all_active_ge4?.mean)}<br><span class="muted">${rateCI(item.all_active_ge4)}</span></td><td>${delta?pp(delta.mean):'—'}<br><span class="muted">${rateDeltaCI(delta)}</span></td><td>${score(item.quality?.mean)}<br><span class="muted">${scoreCI(item.quality)}</span></td><td>${item.n??'—'}</td><td>${qualityNText}</td><td>${featureList(item).map(featureLabel).join(' + ')||'—'}</td></tr>`;
}
function featureRows(){
  return allFourIds.flatMap(id=>{const item=condition(id);if(!item)return [`<tr><th colspan="6">${methodName(id.replace('_1111',''))}: missing</th></tr>`];return FEATURES.map(feature=>{const metric=item.features?.[feature];return `<tr><th>${methodName(id.replace('_1111',''))}</th><th>${featureLabel(feature)}</th><td>${score(metric?.expected?.mean)}<br><span class="muted">${scoreCI(metric?.expected)}</span></td><td>${scoreDelta(metric?.delta_expected_vs_baseline?.mean)}<br><span class="muted">${scoreCI(metric?.delta_expected_vs_baseline)}</span></td><td>${pct(metric?.p_ge4?.mean)}<br><span class="muted">${rateCI(metric?.p_ge4)}</span></td><td>${pp(metric?.delta_p_ge4_vs_baseline?.mean)}<br><span class="muted">${rateDeltaCI(metric?.delta_p_ge4_vs_baseline)}</span></td></tr>`})}).join('');
}
function pairHeatmap(){
  return `<table><tr><th>Пара</th>${methods.map(method=>`<th>${methodName(method)}</th>`).join('')}</tr>${pairMasks.map(mask=>`<tr><th>${pairLabel(mask)}<br><span class="muted">${mask}</span></th>${methods.map(method=>{const item=conditionFor(method,mask);return item?`<td class="heat" style="${heatStyle(item.mean_minimum_expected?.mean)}"><b>${score(item.mean_minimum_expected?.mean)}</b><small>joint ${pct(item.all_active_ge4?.mean)}</small></td>`:'<td class="muted">missing</td>'}).join('')}</tr>`).join('')}</table>`;
}
function contrastCell(item){const d=item?.delta_mean_minimum_expected,j=item?.delta_all_active_ge4;return item?`<span class="${statusClass(d)}">${status(d)}</span><br>${scoreDelta(d?.mean)}<br><span class="muted">${scoreCI(d)} · joint ${pp(j?.mean)} ${rateDeltaCI(j)}</span>`:'missing';}
function pairContrasts(){return `<table><tr><th>Пара</th>${contrastNames.map(name=>`<th>${contrastLabels[name]}</th>`).join('')}</tr>${pairMasks.map(mask=>`<tr><th>${pairLabel(mask)}</th>${contrastNames.map(name=>`<td>${contrastCell(contrast(name,mask))}</td>`).join('')}</tr>`).join('')}</table>`}
function retentionTable(){const rows=RESULT?.retention||[];if(!rows.length)return '<p class="muted">Retention ещё не появился в summary.</p>';return `<table><tr><th>Метод</th><th>Признак</th><th>n</th><th>Full − singleton</th><th>CI</th></tr>${rows.map(item=>{const d=item.full_minus_singleton;return `<tr><th>${methodName(item.method)}</th><th>${featureLabel(item.feature)}</th><td>${item.n??'—'}</td><td class="${statusClass(d)}">${scoreDelta(d?.mean)}</td><td>${scoreCI(d)}</td></tr>`}).join('')}</table>`}
function selectionPanel(){const selection=RESULT?.selection;if(!selection)return '<p class="muted">Dev selection ещё не подключён.</p>';return `<div class="cards"><div class="card"><b>Выбранный λ</b><strong>${score(selection.selected_lambda)}</strong><small>${selection.selection_reference||'dev reference'}</small></div><div class="card"><b>Baseline quality</b><strong>${score(selection.baseline_quality)}</strong><small>на dev</small></div></div><table><tr><th>λ</th><th>Mean minimum</th><th>Quality</th><th>Quality-safe</th></tr>${(selection.candidates||[]).map(item=>`<tr><th>${score(item.lambda)}</th><td>${score(item.mean_minimum_expected)}</td><td>${score(item.quality)}</td><td>${item.quality_safe?'yes':'no'}</td></tr>`).join('')}</table>`}

if(!RESULT){
  document.querySelector('#results').innerHTML='<h2>Результаты Exp4 ещё не подключены</h2><p>Когда очередь создаст summary.json, пересобери страницу:</p><div class="formula">python experiments/meeting-dashboard-4/build.py --summary outputs/strong-composition-exp4/summary.json</div><p class="callout warn">Пустая секция не означает нулевой эффект — это только отсутствие итогового Judge.</p>';
  document.querySelector('#interpret').innerHTML='<h2>Как читать будущий результат</h2><div class="insights"><article class="insight"><h3>Положительный CI</h3><p>Если CI paired difference полностью выше нуля, метод дал устойчивый прирост на одинаковых prompts.</p></article><article class="insight warn"><h3>Неопределённый CI</h3><p>Если CI пересекает ноль, directional claim делать нельзя.</p></article><article class="insight"><h3>Retention</h3><p>Отрицательный full − singleton показывает, что соседние направления ослабили отдельный признак.</p></article></div>';
}else{
  const requiredConditionIds=['baseline',...allFourIds,...pairMasks.flatMap(mask=>methods.map(method=>`${method}_${mask}`))];
  const missingConditions=requiredConditionIds.filter(id=>!condition(id));
  const missingContrasts=pairMasks.flatMap(mask=>contrastNames.map(name=>`${name}:${mask}`)).filter(key=>{const [name,mask]=key.split(':');return !contrast(name,mask)});
  const missingSections=[!RESULT.selection?'dev selection':null,qualityN===null?'quality n':null,!RESULT.retention?.length?'retention':null,!RESULT.judge_usage?'Judge usage':null].filter(Boolean);
  const missing=[...missingConditions,...missingContrasts,...missingSections];
  const full=condition('gdn_rss_r4_1111');
  const allFourContrasts=(RESULT.contrasts||[]).filter(item=>item.mask==='1111');
  const forest=allFourContrasts.map(item=>forestRow(contrastLabels[item.contrast]||item.contrast,item.delta_mean_minimum_expected)).join('');
  const primary=contrast('rank4_minus_rank1_rss','1111')||contrast('rank4_minus_rank1_raw','1111');
  const qualityDelta=full?.delta_quality_vs_baseline;
  const conclusions=[
    `<article class="insight"><h3>Совместная композиция</h3><p>${full?`Для RSS rank 4 mean minimum = ${score(full.mean_minimum_expected?.mean)}, joint = ${pct(full.all_active_ge4?.mean)}. Это описывает наблюдаемый результат, но не заменяет paired contrast.`:'All-four условие пока отсутствует.'}</p></article>`,
    `<article class="insight"><h3>Rank и RSS</h3><p>${primary?`Для ${contrastLabels[primary.contrast]} статус <span class="status ${status(primary.delta_mean_minimum_expected)}">${status(primary.delta_mean_minimum_expected)}</span>: ${scoreDelta(primary.delta_mean_minimum_expected?.mean)}.`:'Нужные rank-контрасты пока отсутствуют.'}</p></article>`,
    `<article class="insight ${qualityDelta&&qualityDelta.ci95_high<0?'warn':''}"><h3>Качество ответа</h3><p>${qualityDelta?`Δ quality = ${scoreDelta(qualityDelta.mean)}, CI ${scoreCI(qualityDelta)}.`:'Quality delta пока отсутствует.'}</p></article>`,
  ].join('');
  document.querySelector('#results').innerHTML=`<h2>Результаты Exp4</h2><p class="note">Загружено условий: ${allConditions().length}; контрастов: ${(RESULT.contrasts||[]).length}. Пропущенные условия не считаются нулевыми.</p><h3>Dev selection</h3>${selectionPanel()}<h3 style="margin-top:20px">All-four: итоговые условия</h3><table><tr><th>Условие</th><th>Mean minimum</th><th>Все активные ≥4</th><th>Δ joint vs baseline</th><th>Quality</th><th>Trait n</th><th>Quality n</th><th>Признаки</th></tr><tr><th>Baseline</th><td>${score(condition('baseline')?.mean_minimum_expected?.mean)}<br><span class="muted">${scoreCI(condition('baseline')?.mean_minimum_expected)}</span></td><td>${pct(condition('baseline')?.all_active_ge4?.mean)}<br><span class="muted">${rateCI(condition('baseline')?.all_active_ge4)}</span></td><td>—</td><td>${score(condition('baseline')?.quality?.mean)}<br><span class="muted">${scoreCI(condition('baseline')?.quality)}</span></td><td>${condition('baseline')?.n??'—'}</td><td>${qualityNText}</td><td>all four</td></tr>${allFourIds.map(conditionRow).join('')}</table><p class="note">Joint CI — доля и её 95% CI; Δ joint — процентные пункты. Score CI — баллы шкалы 1–5.</p><h3>Feature-level metrics</h3><table><tr><th>Метод</th><th>Признак</th><th>Expected</th><th>Δ expected</th><th>p(score ≥4)</th><th>Δ p(score ≥4)</th></tr>${featureRows()}</table><h3>Основные all-four paired contrasts</h3><div class="viz-grid"><article class="viz-card"><h3>Forest plot, Δ mean minimum</h3><p class="small">Ноль — отсутствие разницы; CI в баллах, шкала −0.2…+0.2.</p>${forest||'<p class="muted">Контрасты missing.</p>'}</article><article class="viz-card"><h3>Выбранный метод</h3><p>${full?`${methodName(full.condition)}: ${score(full.mean_minimum_expected?.mean)} балла, joint ${pct(full.all_active_ge4?.mean)}.`:'missing'}</p><p class="small">Это не автоматический выбор победителя: смотрим одновременно score, joint и quality.</p></article></div><h3>Все пары: heatmap mean minimum / joint rate</h3>${pairHeatmap()}<h3>Все pairwise contrasts</h3>${pairContrasts()}<h3>Singleton retention</h3>${retentionTable()}<p class="note">Judge usage: ${RESULT.judge_usage?.input_tokens??'—'} input, ${RESULT.judge_usage?.output_tokens??'—'} output, estimated $${RESULT.judge_usage?.estimated_usd===undefined?'—':Number(RESULT.judge_usage.estimated_usd).toFixed(2)}.</p>${missing.length?`<p class="callout warn">Missing blocks or fields: ${missing.join(', ')}.</p>`:''}`;
  document.querySelector('#interpret').innerHTML=`<h2>Нейтральные выводы по CI</h2><div class="insights">${conclusions}</div><p class="callout">Статус positive означает, что 95% CI полностью выше нуля; inconclusive — CI пересекает ноль; negative — полностью ниже нуля. Это сравнение на тех же prompts, а не вероятность истинности гипотезы.</p>`;
}
</script></body></html>"""


if __name__ == "__main__":
    main()
