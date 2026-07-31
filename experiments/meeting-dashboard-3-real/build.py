"""Build the corrected, self-contained Dashboard 3 from the real Exp3 summary."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = Path(__file__).with_name("exp3_summary.json")
SELECTION = Path(__file__).with_name("exp3_selection.json")
OUTPUT = ROOT / "outputs/meeting-dashboard-3/index.html"
FEATURES = ("joy", "concrete", "optimism", "candor")


def mask_features(mask: str) -> tuple[str, ...]:
    """Decode the run's bit mask (bit 0 is the first feature)."""
    value = int(mask, 2)
    return tuple(
        feature for index, feature in enumerate(FEATURES) if value & (1 << index)
    )


def build(summary: dict, selection: dict) -> str:
    payload = json.dumps(
        {"summary": summary, "selection": selection},
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    baseline = next(
        (
            item
            for item in summary.get("conditions", [])
            if item.get("condition") == "baseline"
        ),
        {},
    )
    return (
        HTML.replace("__DATA__", payload)
        .replace("__GENERATED__", generated)
        .replace("__CONDITION_COUNT__", str(len(summary.get("conditions", []))))
        .replace("__TEST_N__", str(baseline.get("n", "missing")))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        build(
            json.loads(args.summary.read_text(encoding="utf-8")),
            json.loads(args.selection.read_text(encoding="utf-8")),
        ),
        encoding="utf-8",
    )
    print(args.output)


HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hybrid Steering — Experiment 3 (real)</title>
<style>
:root{--bg:#09111c;--panel:#111d2c;--panel2:#0d1724;--line:#2a3b52;--text:#edf4ff;--muted:#9fb0c4;--cyan:#64ddd4;--blue:#8794ff;--green:#5ee09b;--amber:#f2bd63;--red:#ff8585}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 Inter,system-ui,sans-serif}header,main{max-width:1320px;margin:auto;padding:24px}header{padding-top:36px}h1{font-size:36px;line-height:1.08;margin:6px 0 12px}h2{font-size:20px;margin:0 0 8px}h3{font-size:15px;margin:0 0 7px}.eyebrow{color:var(--cyan);font-weight:800;letter-spacing:.08em;text-transform:uppercase}.muted,.note{color:var(--muted)}.note{font-size:13px}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}.panel{grid-column:span 12;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px;overflow:auto}.half{grid-column:span 6}.third{grid-column:span 4}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:16px 0}.card{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:13px}.card strong{display:block;font-size:27px;margin:4px 0}.good{color:var(--green)}.warn{color:var(--amber)}.bad{color:var(--red)}table{width:100%;border-collapse:collapse;margin-top:14px}th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}th:first-child,td:first-child{text-align:left}th{color:#c9d6e7;font-size:12px;white-space:nowrap}tr:last-child td,tr:last-child th{border-bottom:0}.callout{border-left:3px solid var(--cyan);padding-left:12px;margin:14px 0}.callout.warn{border-color:var(--amber)}.callout.bad{border-color:var(--red)}.small{font-size:12px;color:var(--muted)}.tag{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:2px 8px;font-size:11px;color:var(--muted);margin:2px}.formula{font:13px ui-monospace,SFMono-Regular,Consolas,monospace;background:#08101a;border:1px solid var(--line);padding:11px;border-radius:8px;white-space:pre-wrap}.insights{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.insight{background:var(--panel2);border-radius:10px;padding:13px;border-top:3px solid var(--cyan)}.insight.warn{border-top-color:var(--amber)}.source{font-size:12px;color:var(--muted);margin-top:16px}.mono{font:12px ui-monospace,SFMono-Regular,Consolas,monospace}.missing{color:var(--muted);font-style:italic}.sticky th{position:sticky;top:0;background:var(--panel)}.viz-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:16px}.viz-card{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:13px}.profile-group{margin:10px 0}.profile-title{font-size:12px;color:#c9d6e7;margin-bottom:5px}.bar-row{display:grid;grid-template-columns:120px 1fr 44px;gap:7px;align-items:center;margin:4px 0}.bar-label{font-size:11px;color:var(--muted)}.bar-track{height:12px;background:#08101a;border-radius:99px;overflow:hidden}.bar-fill{height:100%;border-radius:99px}.bar-fill.base{background:#6b7b91}.bar-fill.raw{background:var(--blue)}.bar-fill.rss{background:var(--green)}.forest-row{display:grid;grid-template-columns:150px 1fr 68px;gap:8px;align-items:center;margin:10px 0}.forest-label{font-size:11px;color:var(--muted)}.forest-track{height:18px;position:relative;background:#08101a;border-radius:99px}.forest-zero{position:absolute;left:50%;top:0;height:100%;border-left:1px dashed var(--amber)}.forest-ci{position:absolute;top:7px;height:4px;background:var(--cyan);border-radius:99px}.forest-dot{position:absolute;top:3px;width:12px;height:12px;margin-left:-6px;background:var(--green);border-radius:50%;border:2px solid var(--panel2)}
@media(max-width:950px){.half,.third{grid-column:span 12}.cards,.insights,.viz-grid{grid-template-columns:1fr 1fr}}@media(max-width:560px){header,main{padding:16px}.cards,.insights,.viz-grid{grid-template-columns:1fr}h1{font-size:29px}.bar-row{grid-template-columns:95px 1fr 40px}.forest-row{grid-template-columns:110px 1fr 60px}}
</style></head>
<body><header>
<div class="eyebrow">Hybrid Steering · Experiment 3 · Qwen3.5-9B</div>
<h1>Реальные результаты composition-normalization-v3</h1>
<p class="muted">Результаты Exp3: __CONDITION_COUNT__ условий, __TEST_N__ тестовых промптов и Judge с вероятностным expected score. Построено __GENERATED__.</p>
<p class="callout warn"><b>Важно:</b> это matched-strength эксперимент. Для каждого признака alpha подбиралась на dev, поэтому результат отвечает на вопрос «как ведут себя методы при сопоставимой силе», а не «какой метод можно раскрутить сильнее всего».</p>
</header><main class="grid">
<section class="panel" id="presentation"></section>
<section class="panel" id="overview"></section>
<section class="panel" id="features"></section>
<section class="panel" id="allfour"></section>
<section class="panel" id="matrix"></section>
<section class="panel" id="profiles"></section>
<section class="panel" id="combinations"></section>
<section class="panel" id="contrasts"></section>
<section class="panel" id="triples"></section>
<section class="panel" id="conclusions"></section>
</main>
<script>
const D=__DATA__, S=D.summary, SEL=D.selection;
const FEATURES=['joy','concrete','optimism','candor'];
const N={joy:['Joy','Радость'],concrete:['Concrete language','Конкретный язык'],optimism:['Optimism','Оптимизм'],candor:['Principled candor','Принципиальная прямота']};
const M={
 baseline:'Без steering',act_raw_1111:'Activation steering, raw',act_rss_1111:'Activation steering, RSS',
 gdn_raw_r1_1111:'GDN raw, rank 1',gdn_rss_r1_1111:'GDN RSS, rank 1',gdn_rss_r4_1111:'GDN RSS, rank 4'
};
const F=x=>N[x]?.[1]||x; const fmt=x=>Number(x).toFixed(3); const pct=x=>(100*Number(x)).toFixed(1)+'%';
const signed=x=>(x>=0?'+':'')+Number(x).toFixed(3); const scoreDelta=x=>signed(x)+' балла';
const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const cls=x=>x>0?'good':x<0?'bad':'warn';
const cond=id=>S.conditions.find(x=>x.condition===id);
const byId={}; S.conditions.forEach(x=>byId[x.condition]=x);
const maskFeatures=mask=>{const value=parseInt(mask,2);return FEATURES.filter((_,i)=>value&(1<<i))};
const featuresFor=item=>item?.active_features||maskFeatures(item?.mask||'');
const labels=fs=>fs.map(F).join(' + ');
function ci(x){return `[${fmt(x.ci95_low)}, ${fmt(x.ci95_high)}]`}
function rateCI(p,n){const z=1.96,den=1+z*z/n,center=(p+z*z/(2*n))/den,half=z*Math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return [Math.max(0,center-half),Math.min(1,center+half)]}
function rowName(id){return M[id]||id.replaceAll('_',' ')}
function scoreCells(x, only=FEATURES){return only.map(f=>`<td>${fmt(x.feature_expected_means[f])}</td>`).join('')}
const paired=(name,mask='1111')=>S.contrasts.find(x=>x.contrast===name&&x.mask===mask);
const primary=paired('gdn_minus_activation_raw'); const rankSignal=paired('gdn_rank4_minus_rank1'); const rssSignal=paired('gdn_rss_r1_minus_raw');
const barWidth=x=>`${Math.max(0,Math.min(100,Number(x)/5*100))}%`;
const forestLeft=x=>`${Math.max(0,Math.min(100,(Number(x)+.2)/.4*100))}%`;
const forestWidth=(lo,hi)=>`${Math.max(0,Math.min(100,(Number(hi)-Number(lo))/.4*100))}%`;
const forestRow=(label,d)=>`<div class="forest-row"><span class="forest-label">${label}</span><div class="forest-track"><span class="forest-zero"></span><span class="forest-ci" style="left:${forestLeft(d.ci95_low)};width:${forestWidth(d.ci95_low,d.ci95_high)}"></span><span class="forest-dot" style="left:${forestLeft(d.mean)}"></span></div><span class="mono">${scoreDelta(d.mean)}</span></div>`;

const base=byId.baseline, full=byId.gdn_rss_r4_1111;
const profileSources=[['Baseline','base',base],['GDN raw r1','raw',byId.gdn_raw_r1_1111],['GDN RSS r4','rss',full]];
const profileChart=FEATURES.map(f=>`<div class="profile-group"><div class="profile-title">${F(f)}</div>${profileSources.map(([label,kind,item])=>`<div class="bar-row"><span class="bar-label">${label}</span><div class="bar-track"><span class="bar-fill ${kind}" style="width:${barWidth(item.feature_expected_means[f])}"></span></div><span class="mono">${fmt(item.feature_expected_means[f])}</span></div>`).join('')}</div>`).join('');
const forestData=[['GDN raw r1 − activation raw',primary?.paired_difference],['RSS r4 − RSS r1',rankSignal?.paired_difference],['RSS r1 − raw r1',rssSignal?.paired_difference]].filter(([,d])=>d);
const forestChart=forestData.map(([label,d])=>forestRow(label,d)).join('');
document.querySelector('#presentation').innerHTML=`<h2>Тезисы для показа</h2><div class="insights">
<article class="insight"><h3>1. Наблюдаемое совместное проявление</h3><p>Для четырёх признаков joint endpoint составляет ${pct(base.all_active_ge4)} у baseline и ${pct(full.all_active_ge4)} у RSS rank 4. Это descriptive evidence: совместные ответы наблюдаются, но одного этого роста недостаточно для строгого доказательства композициональности.</p></article>
<article class="insight"><h3>2. Поддержанные CI-контрасты</h3><p>GDN raw rank 1 − activation raw = ${scoreDelta(primary.paired_difference.mean)}, CI [${scoreDelta(primary.paired_difference.ci95_low)}, ${scoreDelta(primary.paired_difference.ci95_high)}]. RSS rank 4 − RSS rank 1 = ${scoreDelta(rankSignal.paired_difference.mean)}, CI [${scoreDelta(rankSignal.paired_difference.ci95_low)}, ${scoreDelta(rankSignal.paired_difference.ci95_high)}].</p></article>
<article class="insight warn"><h3>3. Граница Exp3</h3><p>RSS rank 4 — лучший абсолютный результат, но Exp3 не содержал raw rank 4 и не давал paired CI для baseline joint rate. Поэтому разделить вклад rank и RSS можно только в Exp4.</p></article>
</div><h3>Feature-level delta: RSS rank 4 − baseline</h3><table><tr><th>Признак</th><th>Baseline</th><th>RSS rank 4</th><th>Δ, баллы</th></tr>${FEATURES.map(f=>`<tr><th>${F(f)}</th><td>${fmt(base.feature_expected_means[f])}</td><td>${fmt(full.feature_expected_means[f])}</td><td class="${cls(full.feature_expected_means[f]-base.feature_expected_means[f])}">${scoreDelta(full.feature_expected_means[f]-base.feature_expected_means[f])}</td></tr>`).join('')}</table><div class="viz-grid"><article class="viz-card"><h3>Профиль признаков, шкала 1–5</h3>${profileChart}<p class="small">Полоса показывает expected score; это не доля ответов.</p></article><article class="viz-card"><h3>Forest plot paired-контрастов</h3><p class="small">Ноль — отсутствие разницы; диапазон графика −0.2…+0.2 балла.</p>${forestChart}</article></div>`;
document.querySelector('#overview').innerHTML=`<h2>Короткий ответ</h2><div class="cards">
<div class="card"><span>Baseline: все 4 ≥4</span><strong>${pct(base.all_active_ge4)}</strong><small>N=${base.n}; качество ${fmt(base.quality_mean)}/5</small></div>
<div class="card"><span>GDN raw rank 1</span><strong>${pct(byId.gdn_raw_r1_1111.all_active_ge4)}</strong><small>mean minimum ${fmt(byId.gdn_raw_r1_1111.mean_minimum_expected)}/5</small></div>
<div class="card"><span>GDN RSS rank 4</span><strong class="good">${pct(full.all_active_ge4)}</strong><small>mean minimum ${fmt(full.mean_minimum_expected)}/5</small></div>
<div class="card"><span>Лучший доказанный контраст</span><strong class="good">${scoreDelta(primary.paired_difference.mean)}</strong><small>GDN raw rank1 − activation raw, all-four</small></div>
</div><p class="callout"><b>Главный вывод:</b> в Exp3 наблюдаются ответы, где все четыре порога проходят одновременно: ${pct(base.all_active_ge4)} у baseline и ${pct(full.all_active_ge4)} у RSS rank 4. Это описательный результат, а не отдельное доказательство композициональности. Самый устойчивый сигнал — GDN против activation и переход к rank 4; RSS на rank 1 преимуществ не показал.</p>`;

document.querySelector('#features').innerHTML=`<h2>Что именно проверяли</h2><p class="note">Биты задаются числом условия: <b>0001 = joy, 0010 = concrete, 0100 = optimism, 1000 = candor</b>; поэтому маски в таблицах не следует читать как обычный порядок слов слева направо. В каждой строке Judge независимо ставит каждому признаку expected score на шкале 1–5 и отдельно оценивает качество ответа. «Все ≥4» означает, что у одного ответа все активные признаки получили минимум 4.</p><div class="cards">
${FEATURES.map(f=>`<div class="card"><b>${N[f][0]}</b><strong>${F(f)}</strong><small>Оценивается поведение ответа, а не его полезность или длина.</small></div>`).join('')}</div>
<h3>Выбранные alpha для matched-strength</h3><table><tr><th>Признак</th><th>Целевой dev score</th><th>Выбранный GDN alpha</th><th>Почему</th></tr>${FEATURES.map(f=>`<tr><th>${F(f)}</th><td>${fmt(SEL[f].matched_target)}</td><td>${SEL[f].gdn.selected_alpha}</td><td>Ближайший к target при quality-safe выборе</td></tr>`).join('')}</table>
<div class="formula">mean_minimum = среднее по ответам минимального expected score среди активных признаков
joint_ge4 = доля ответов, где каждый активный признак получил score ≥ 4
expected_score = probability-weighted среднее Judge; это чувствительнее целого trait_score</div>`;

const fullIds=['baseline','act_raw_1111','act_rss_1111','gdn_raw_r1_1111','gdn_rss_r1_1111','gdn_rss_r4_1111'];
let all=`<h2>Все четыре признака одновременно</h2><p class="note">Это главный joint endpoint. Для baseline активных признаков нет, но мы всё равно показываем, как часто естественный ответ случайно проходит тот же порог по четырём шкалам.</p><table class="sticky"><tr><th>Условие</th><th>Все 4 ≥4</th><th>95% CI</th><th>Mean minimum</th><th>Качество</th>${FEATURES.map(f=>`<th>${N[f][0]}</th>`).join('')}</tr>`;
fullIds.forEach(id=>{const x=byId[id],ciRate=rateCI(x.all_active_ge4,x.n); all+=`<tr><th>${rowName(id)}</th><td class="${x.all_active_ge4>=.1?'good':'warn'}"><b>${pct(x.all_active_ge4)}</b></td><td>${pct(ciRate[0])} … ${pct(ciRate[1])}</td><td>${fmt(x.mean_minimum_expected)}</td><td>${fmt(x.quality_mean)}</td>${scoreCells(x)}</tr>`});
all+=`</table><p class="note">CI для joint rate здесь — ориентировочный Wilson 95% CI по ${base.n} независимым ответам. Основные сравнения ниже используют paired bootstrap CI по тем же prompt_id.</p>`;
document.querySelector('#allfour').innerHTML=all;

const methods=[['gdn_raw_r1_1111','Raw','Rank 1'],['gdn_rss_r1_1111','RSS','Rank 1'],['gdn_raw_r4_1111','Raw','Rank 4'],['gdn_rss_r4_1111','RSS','Rank 4']];
let mat=`<h2>Матрица «rank × нормализация»</h2><p class="note">Rank 1 у каждого признака — отдельное rank-1 направление; при сложении четырёх таких направлений общая сумма может иметь ранг до 4. RSS — нормализация суммы по root-sum-square, а не SVD-обрезка.</p><table><tr><th>Метод</th><th>Все 4 ≥4</th><th>Mean minimum</th><th>Качество</th><th>Что сравниваем</th></tr>`;
methods.forEach(([id,n,r])=>{const x=byId[id];if(!x){mat+=`<tr><th>${n}, ${r}</th><td colspan="4" class="missing">В Exp3 не посчитано</td></tr>`;return}mat+=`<tr><th>${n}, ${r}</th><td><b>${pct(x.all_active_ge4)}</b></td><td>${fmt(x.mean_minimum_expected)}</td><td>${fmt(x.quality_mean)}</td><td>${n==='RSS'?'направления сначала складываются, затем нормализуются':'направления складываются напрямую'}</td></tr>`});
mat+=`</table><p class="callout"><b>Честная интерпретация:</b> raw rank 4 отсутствует — это пробел Exp3. Поэтому нельзя говорить, что RSS «победил rank 4 raw» или что rank 4 всегда лучше: в Exp3 есть только RSS rank 4. Самый надёжный доступный контраст — RSS rank 4 − RSS rank 1: ${scoreDelta(rankSignal.paired_difference.mean)}, CI [${scoreDelta(rankSignal.paired_difference.ci95_low)}, ${scoreDelta(rankSignal.paired_difference.ci95_high)}] по mean minimum.</p>`;
document.querySelector('#matrix').innerHTML=mat;

let prof=`<h2>Какие признаки реально меняются в полном составе</h2><p class="note">Числа — expected score. Delta считается относительно baseline; это не процент ответов, а изменение средней оценки на шкале 1–5.</p><table><tr><th>Признак</th><th>Baseline</th><th>GDN raw r1</th><th>GDN RSS r1</th><th>GDN RSS r4</th><th>RSS r4 − baseline</th></tr>`;
FEATURES.forEach(f=>{const b=base.feature_expected_means[f], r1=byId.gdn_raw_r1_1111.feature_expected_means[f], rr1=byId.gdn_rss_r1_1111.feature_expected_means[f], rr4=full.feature_expected_means[f];prof+=`<tr><th>${F(f)}</th><td>${fmt(b)}</td><td>${fmt(r1)}</td><td>${fmt(rr1)}</td><td>${fmt(rr4)}</td><td class="${cls(rr4-b)}">${scoreDelta(rr4-b)}</td></tr>`});
prof+=`</table><p class="callout warn"><b>Почему all-four rate низкий:</b> joy на matched alpha слабее остальных, а concrete и candor уже высоки в baseline — у них есть ceiling effect. Низкий joint rate нельзя приписывать только плохому смешиванию.</p>`;
document.querySelector('#profiles').innerHTML=prof;

const rank1=S.conditions.filter(x=>x.condition.startsWith('gdn_raw_r1_')&&x.condition!=='gdn_raw_r1_1111').sort((a,b)=>a.active_features.length-b.active_features.length||a.condition.localeCompare(b.condition));
let comb=`<h2>Комбинации: raw GDN rank 1</h2><p class="note">Здесь виден каждый одиночный, парный и тройной состав, реально посчитанный в Exp3. Не путайте joint rate с «силой одного признака»: joint rate требует, чтобы все активные признаки были ≥4 в одном и том же ответе.</p><table><tr><th>Состав</th><th>Mean minimum</th><th>Все активные ≥4</th><th>Активные expected scores</th></tr>`;
rank1.forEach(x=>{const fs=featuresFor(x);comb+=`<tr><th>${labels(fs)}</th><td>${fmt(x.mean_minimum_expected)}</td><td><b>${pct(x.all_active_ge4)}</b></td><td>${fs.map(f=>`${F(f)}: ${fmt(x.feature_expected_means[f])}`).join(' · ')}</td></tr>`});
comb+=`</table><p class="callout">Сочетания с optimism чаще дают положительный GDN−activation contrast, особенно optimism+candor и concrete+optimism. Это гипотеза о совместимости направления, а не доказательство универсальной композициональности.</p>`;
document.querySelector('#combinations').innerHTML=comb;

const contrastIds=['gdn_rss_r1_minus_raw|1111','gdn_rank4_minus_rank1|1111','activation_rss_minus_raw|1111','gdn_minus_activation_rss|1111','gdn_minus_activation_raw|1111','gdn_rank4_minus_rank1|1100','gdn_minus_activation_raw|1100','gdn_minus_activation_raw|0110'];
const contrast=(name,mask)=>S.contrasts.find(x=>x.contrast===name&&x.mask===mask);
let ct=`<h2>Парные контрасты и доверительные интервалы</h2><p class="note">Это уже разность на тех же prompt_id, поэтому она информативнее простого сравнения процентов. CI, пересекающий ноль, не даёт уверенного directional claim.</p><table><tr><th>Контраст</th><th>Состав</th><th>Δ mean minimum</th><th>95% CI</th><th>Вывод</th></tr>`;
contrastIds.forEach(key=>{const [n,mask]=key.split('|'),x=contrast(n,mask);if(!x)return;const d=x.paired_difference;const verdict=d.ci95_low>0?'CI выше нуля':d.ci95_high<0?'CI ниже нуля':'CI пересекает ноль';ct+=`<tr><th>${n.replaceAll('_',' ')}</th><td>${labels(featuresFor(x))}</td><td class="${cls(d.mean)}">${scoreDelta(d.mean)}</td><td>[${scoreDelta(d.ci95_low)}, ${scoreDelta(d.ci95_high)}]</td><td>${verdict}</td></tr>`});
ct+=`</table><p class="callout"><b>Что здесь лучше всего видно:</b> GDN raw rank 1 выше activation raw на all-four на ${scoreDelta(primary.paired_difference.mean)} (CI полностью выше нуля). Для RSS rank 1 преимущество над raw не доказано. Rank 4 RSS выше rank 1 RSS на ${scoreDelta(rankSignal.paired_difference.mean)}.</p>`;
document.querySelector('#contrasts').innerHTML=ct;

const tripleMasks=['0111','1011','1101','1110'];
let triples=`<h2>Тройки: сила и сохранение качества</h2><p class="note">Для каждой тройки рядом показаны joint endpoint и отдельное качество ответа. Так слабую выраженность отдельного признака можно отличить от общего падения качества.</p><table><tr><th>Состав</th><th>Raw rank1<br>mean min</th><th>Raw rank1<br>все ≥4</th><th>Raw rank1<br>quality</th><th>RSS rank4<br>mean min</th><th>RSS rank4<br>все ≥4</th><th>RSS rank4<br>quality</th><th>Δ quality</th>`;
tripleMasks.forEach(mask=>{const raw=byId['gdn_raw_r1_'+mask],rss=byId['gdn_rss_r4_'+mask],dq=rss.quality_mean-raw.quality_mean;triples+=`<tr><th>${labels(featuresFor(raw))}</th><td>${fmt(raw.mean_minimum_expected)}</td><td><b>${pct(raw.all_active_ge4)}</b></td><td>${fmt(raw.quality_mean)}</td><td>${fmt(rss.mean_minimum_expected)}</td><td><b>${pct(rss.all_active_ge4)}</b></td><td>${fmt(rss.quality_mean)}</td><td class="${cls(dq)}">${scoreDelta(dq)}</td></tr>`});
triples+=`</table><p class="callout"><b>Что видно:</b> тройка concrete + optimism + candor — сильный рабочий случай: raw rank1 даёт ${pct(byId.gdn_raw_r1_1110.all_active_ge4)}, RSS rank4 — ${pct(byId.gdn_rss_r4_1110.all_active_ge4)}, при quality ${fmt(byId.gdn_rss_r4_1110.quality_mean)}/5. Тройки с joy заметно слабее (пример RSS rank4: ${pct(byId.gdn_rss_r4_0111.all_active_ge4)}, ${pct(byId.gdn_rss_r4_1011.all_active_ge4)}, ${pct(byId.gdn_rss_r4_1101.all_active_ge4)}), но их quality остаётся около ${fmt(byId.gdn_rss_r4_0111.quality_mean)}–${fmt(byId.gdn_rss_r4_1101.quality_mean)}. Это похоже на слабую выраженность joy, а не на разрушение ответа.</p>`;
document.querySelector('#triples').innerHTML=triples;

document.querySelector('#conclusions').innerHTML=`<h2>Выводы и границы вывода</h2><div class="insights">
<article class="insight"><h3>1. GDN умеет складывать признаки, но не идеально</h3><p>У full-four GDN raw rank1 joint rate ${pct(byId.gdn_raw_r1_1111.all_active_ge4)}, RSS rank4 — ${pct(full.all_active_ge4)}. Это значит, что часть ответов одновременно сохраняет четыре режима, но не все.</p></article>
<article class="insight"><h3>2. Rank 4 помогает в RSS</h3><p>Контраст RSS rank4 − rank1 равен ${scoreDelta(rankSignal.paired_difference.mean)}, CI [${scoreDelta(rankSignal.paired_difference.ci95_low)}, ${scoreDelta(rankSignal.paired_difference.ci95_high)}] и не пересекает ноль. Это самый ясный rank-сигнал Exp3, но raw rank4 нужно добавить в Exp4 для полной матрицы.</p></article>
<article class="insight"><h3>3. RSS на rank1 не является доказанным улучшением</h3><p>RSS rank1 − raw rank1 = ${scoreDelta(rssSignal.paired_difference.mean)}, CI [${scoreDelta(rssSignal.paired_difference.ci95_low)}, ${scoreDelta(rssSignal.paired_difference.ci95_high)}] пересекает ноль. Нельзя называть нормализацию универсально лучшей без rank/strength контроля.</p></article>
<article class="insight"><h3>4. Baseline и ceiling важны</h3><p>Concrete и candor уже близки к 4 в baseline, поэтому их рост ограничен. Joy заметно слабее; joint rate штрафуется одним слабым признаком.</p></article>
<article class="insight warn"><h3>5. Чего Exp3 не доказал</h3><p>Нет raw rank4, нет ортогональных rank slots «по одному рангу на признак», нет независимого holdout для всех factorial-комбинаций и нет человеческой калибровки Judge.</p></article>
<article class="insight"><h3>6. Зачем Exp4</h3><p>Exp4 добирает raw rank4, сравнивает raw/RSS на all-four и всех парах, а затем отдельно показывает singleton retention. Это закрывает главный пробел Exp3.</p></article>
</div><p class="source">Judge usage Exp3: ${Number(S.judge_usage.input_tokens).toLocaleString('ru-RU')} input tokens, ${Number(S.judge_usage.output_tokens).toLocaleString('ru-RU')} output tokens, оценка API $${Number(S.judge_usage.estimated_usd).toFixed(2)}.</p>`;
</script></body></html>"""


if __name__ == "__main__":
    main()
