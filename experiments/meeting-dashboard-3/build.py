"""Build a self-contained Russian dashboard for Experiment #3."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = ROOT / "experiments/composition-generation-queue/results/summary.json"
DEFAULT_COMPARISONS = (
    ROOT / "experiments/composition-generation-queue/results/comparisons.json"
)
DEFAULT_ANALYSIS = (
    ROOT / "experiments/composition-generation-queue/results/composition-analysis.json"
)
DEFAULT_EXP2 = ROOT / "experiments/meeting-dashboard-3/dashboard1_key_results.json"
DEFAULT_OUTPUT = ROOT / "outputs/meeting-dashboard-3/index.html"


def build(summary: dict, comparisons: dict, analysis: dict, exp2: dict) -> str:
    payload = json.dumps(
        {
            "summary": summary,
            "comparisons": comparisons,
            "analysis": analysis,
            "exp2": exp2,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return HTML.replace("__DATA__", payload).replace("__GENERATED__", generated)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--comparisons", type=Path, default=DEFAULT_COMPARISONS)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--exp2", type=Path, default=DEFAULT_EXP2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        build(
            json.loads(args.summary.read_text(encoding="utf-8")),
            json.loads(args.comparisons.read_text(encoding="utf-8")),
            json.loads(args.analysis.read_text(encoding="utf-8")),
            json.loads(args.exp2.read_text(encoding="utf-8")),
        ),
        encoding="utf-8",
    )
    print(args.output)


HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hybrid Steering — Experiment #3</title>
<style>
:root{--bg:#09101a;--panel:#111c2b;--panel2:#0d1724;--line:#26364b;--text:#edf4ff;--muted:#9bacbf;--cyan:#62d8d0;--blue:#8791ff;--green:#58d391;--amber:#f2bd63;--red:#fa7d7d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 Inter,ui-sans-serif,system-ui,sans-serif}header,main{max-width:1280px;margin:auto;padding:24px}header{padding-top:38px}h1{font-size:37px;line-height:1.08;margin:6px 0 12px}h2{font-size:20px;margin:0 0 8px}h3{font-size:15px;margin:0 0 8px}.eyebrow{color:var(--cyan);font-weight:800;letter-spacing:.08em;text-transform:uppercase}.muted,.note{color:var(--muted)}.note{font-size:13px}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}.panel{grid-column:span 12;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px;overflow:auto}.half{grid-column:span 6}.third{grid-column:span 4}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:16px}.card{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:13px}.card strong{display:block;font-size:27px;margin:4px 0}.good{color:var(--green)}.warn{color:var(--amber)}.bad{color:var(--red)}table{width:100%;border-collapse:collapse;margin-top:14px}th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}th:first-child,td:first-child{text-align:left}th{color:#c6d3e4;font-size:12px;white-space:nowrap}tr:last-child td,tr:last-child th{border:0}.tag{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:2px 8px;font-size:11px;color:var(--muted);margin:2px}.barrow{display:grid;grid-template-columns:205px 1fr 85px;gap:10px;align-items:center;margin:11px 0}.track{height:23px;background:#09111c;border-radius:5px;position:relative}.bar{height:100%;border-radius:5px;background:linear-gradient(90deg,var(--blue),var(--cyan))}.ci{position:absolute;top:10px;height:2px;background:white}.ci:before,.ci:after{content:"";position:absolute;top:-3px;width:2px;height:8px;background:#fff}.ci:after{right:0}.value{text-align:right}.callout{border-left:3px solid var(--cyan);padding-left:12px;margin:14px 0}.callout.warn{border-color:var(--amber)}.callout.bad{border-color:var(--red)}.legend{color:var(--muted);font-size:12px;margin-top:8px}.heat td{text-align:center;min-width:72px;font-weight:700}.heat th{text-align:center}.heat th:first-child{text-align:left}.small{font-size:12px;color:var(--muted)}.source{font-size:12px;color:var(--muted);margin-top:16px}.insights{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.insight{background:var(--panel2);border-radius:10px;padding:13px;border-top:3px solid var(--cyan)}.insight h3{margin-bottom:5px}.nav{display:flex;gap:12px;flex-wrap:wrap;margin-top:14px}.nav a{color:var(--cyan);text-decoration:none;border:1px solid var(--line);border-radius:999px;padding:5px 10px}.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px}@media(max-width:900px){.half,.third{grid-column:span 12}.cards,.insights{grid-template-columns:1fr 1fr}}@media(max-width:560px){header,main{padding:16px}.cards,.insights{grid-template-columns:1fr}.barrow{grid-template-columns:130px 1fr 65px}h1{font-size:29px}}
</style>
</head>
<body>
<header>
  <div class="eyebrow">Hybrid Steering · Experiment #3 · Qwen3.5-9B</div>
  <h1>Что происходит, когда мы складываем несколько концептов?</h1>
  <p class="muted">Аналитический дашборд: не журнал запусков, а попытка ответить на вопрос,
  какие признаки реально сохраняются вместе, где появляется конфликт и помогает ли
  нормализация состояния GDN. Построено __GENERATED__.</p>
  <div class="nav"><a href="../meeting-dashboard/dashboard.html">Dashboard 1 · строгий A/B</a><a href="../meeting-dashboard-2/index.html">Dashboard 2 · исходная сводка</a></div>
</header>
<main class="grid">
  <section class="panel" id="overview"></section>
  <section class="panel" id="method"></section>
  <section class="panel" id="composition"></section>
  <section class="panel" id="profile"></section>
  <section class="panel half" id="activation"></section>
  <section class="panel" id="joy"></section>
  <section class="panel" id="exp2"></section>
  <section class="panel" id="insights"></section>
  <section class="panel" id="limits"></section>
</main>
<script>
const D=__DATA__,S=D.summary,C=D.comparisons,A=D.analysis,E=D.exp2;
const names={
  principled_candor:{en:'Principled candor',ru:'Принципиальная прямота',short:'Candor'},
  calm_composure:{en:'Calm composure',ru:'Спокойствие',short:'Calm'},
  concrete_language:{en:'Concrete language',ru:'Конкретный язык',short:'Concrete'},
  casualness:{en:'Casualness',ru:'Разговорный стиль',short:'Casualness'},
  optimism:{en:'Optimism',ru:'Оптимизм',short:'Optimism'},
  joy:{en:'Joy',ru:'Радость',short:'Joy'},
  french_language:{en:'French language',ru:'Французский язык',short:'French'}
};
const methodNames={gdn_svd_per_rank1:'GDN: отдельный rank 1 для каждого признака',gdn_svd_per_rank4:'GDN: отдельный rank 4 для каждого признака',gdn_svd_post_rank1:'GDN: rank 1 после сложения',gdn_svd_post_rank4:'GDN: rank 4 после сложения',gdn_norm_controlled:'GDN: RSS-нормализация'};
const featureOrder=['principled_candor','concrete_language','casualness','optimism'];
const pct=x=>(100*x).toFixed(1)+'%';
const pp=x=>(100*x).toFixed(1)+' п.п.';
const signed=x=>(x>=0?'+':'')+x.toFixed(2);
const ci=(x,low,high,scale=1)=>`[${signed(low*scale)}, ${signed(high*scale)}]`;
const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const cclass=x=>x>0?'good':x<0?'bad':'warn';
const find=(phase,condition)=>S.conditions.find(x=>x.phase===phase&&x.condition===condition);
const labelFeatures=fs=>fs.map(f=>names[f]?.ru||f).join(' + ');
const countClass=x=>x>=4?'good':x<=2?'bad':'warn';
function bar(label,value,low,high){return `<div class="barrow"><span>${esc(label)}</span><div class="track"><div class="bar" style="width:${Math.max(0,Math.min(1,value))*100}%"></div><i class="ci" style="left:${Math.max(0,low)*100}%;width:${Math.max(0,high-low)*100}%"></i></div><b class="value">${pct(value)}</b></div>`}
function cellScore(x){return `<span class="${countClass(x)}">${x.toFixed(2)}</span>`}

const rank1=C.all4_full.gdn_svd_per_rank1,norm=C.all4_full.gdn_norm_controlled;
const hold=A.holdout.methods;
document.querySelector('#overview').innerHTML=`<h2>Короткий ответ</h2>
<div class="cards">
 <div class="card"><span>Лучший GDN в Exp3</span><strong class="good">${pct(norm.joint_ge4)}</strong><small>RSS-нормализация · все 4 признака ≥4/5 · N=${norm.n}</small></div>
 <div class="card"><span>GDN rank 1 без нормализации</span><strong>${pct(rank1.joint_ge4)}</strong><small>Те же 128 prompt'ов · all-four endpoint</small></div>
 <div class="card"><span>Главный bottleneck</span><strong class="warn">${norm.feature_means[2].toFixed(2)}/5</strong><small>Разговорный стиль в полном составе</small></div>
 <div class="card"><span>Joy + optimism, α=8</span><strong class="good">${pct(C.joy_optimism.joy_a8_optimism.joint_ge4)}</strong><small>Оба признака ≥4/5 · exploratory</small></div>
</div>
<p class="callout"><b>Главный вывод:</b> композиция не разваливается полностью, но она не линейна и неравномерна. Candor, concrete language и optimism обычно сохраняются; casualness остаётся слабым звеном. Для GDN самый убедительный выигрыш даёт контроль общей нормы направления, а не увеличение rank с 1 до 4.</p>
<p class="source">Exp3: 128 тестовых prompt'ов для SVD/norm/joy, 96 holdout prompt'ов для заранее выбранных activation-конфигураций. Все числа ниже — автоматическая оценка; API и GPU при построении не запускаются.</p>`;

const fullRows=[
 ['gdn_svd_per_rank1','svd','per_r1_1111'],['gdn_svd_per_rank4','svd','per_r4_1111'],['gdn_svd_post_rank1','svd','post_r1_1111'],['gdn_svd_post_rank4','svd','post_r4_1111'],['gdn_norm_controlled','norm','norm_1111']
];
let method=`<h2>Какой способ сложения работает лучше?</h2><p class="note">Endpoint строгий: один ответ считается успехом только если все четыре активных признака получили ≥4/5. Whisker — 95% CI для rate.</p>`;
fullRows.forEach(([id,phase,condition])=>{const x=C.all4_full[id],s=find(phase,condition);method+=bar(methodNames[id],x.joint_ge4,s.all_active_ge_4.ci95_low,s.all_active_ge_4.ci95_high)});
const normDelta=C.all4_vs_rank1.gdn_norm_controlled_minus_gdn_svd_per_rank1.joint_ge4;
method+=`<table><tr><th>Сравнение с отдельным rank 1</th><th>Δ all-four</th><th>95% CI</th><th>Что это значит</th></tr>`;
const deltaRows=[['gdn_svd_per_rank4','gdn_svd_per_rank4_minus_gdn_svd_per_rank1','больше rank не дал доказанного выигрыша'],['gdn_svd_post_rank1','gdn_svd_post_rank1_minus_gdn_svd_per_rank1','сжимать уже сложенное направление до rank 1 вредно'],['gdn_svd_post_rank4','gdn_svd_post_rank4_minus_gdn_svd_per_rank1','после сложения rank 4 почти возвращает baseline'],['gdn_norm_controlled','gdn_norm_controlled_minus_gdn_svd_per_rank1','единственное явное улучшение GDN']];
deltaRows.forEach(([id,key,note])=>{const d=C.all4_vs_rank1[key].joint_ge4;method+=`<tr><td>${methodNames[id]}</td><td class="${cclass(d.difference)}">${pp(d.difference)}</td><td>${pp(d.ci95_low)} … ${pp(d.ci95_high)}</td><td>${note}</td></tr>`});
document.querySelector('#method').innerHTML=method+`</table><p class="callout"><b>Не следует говорить «rank 4 лучше»:</b> +3.9 п.п. имеет CI, пересекающий ноль. Для RSS-нормализации +14.8 п.п. CI полностью выше нуля: [+5.5, +24.2] п.п.</p>`;

const rankRows=[...A.rank1_composition.conditions].sort((a,b)=>a.active_features.length-b.active_features.length||a.condition.localeCompare(b.condition));
let comp=`<h2>Комбинации признаков: GDN rank 1</h2><p class="note">Здесь видно не только «сколько средний score», но и совместный endpoint. <i>Independence product</i> — произведение одиночных вероятностей ≥4/5; это ориентир при независимости, а не доказательство причинной независимости.</p><table><tr><th>Активные признаки</th><th>N</th><th>Все ≥4/5</th><th>Independence</th><th>Разница</th><th>Средний минимум</th><th>Слабое место</th></tr>`;
rankRows.forEach(r=>{const s=find('svd',r.condition),joint=s.all_active_ge_4;const delta=r.joint_ge4-r.independence_product;const weakest=Object.entries(s.feature_scores).sort((a,b)=>a[1].mean-b[1].mean)[0];comp+=`<tr><th>${esc(labelFeatures(r.active_features))}</th><td>${r.n}</td><td class="${countClass(r.joint_ge4*5)}"><b>${pct(r.joint_ge4)}</b><br><span class="small">${pct(joint.ci95_low)} … ${pct(joint.ci95_high)}</span></td><td>${pct(r.independence_product)}</td><td class="${cclass(delta)}">${pp(delta)}</td><td>${r.mean_minimum.toFixed(2)}</td><td>${names[weakest[0]]?.ru||weakest[0]} (${weakest[1].mean.toFixed(2)})</td></tr>`});
comp+=`</table><h3 style="margin-top:20px">Что меняет RSS-нормализация на самих комбинациях?</h3><p class="note">Ниже уже не rank-1 baseline, а нормализованный GDN. У него нет полной factorial-таблицы, поэтому показываем только реально посчитанные составы.</p><table><tr><th>Активные признаки</th><th>N</th><th>Все ≥4/5</th><th>Средний минимум</th><th>Средние scores</th></tr>`;
S.conditions.filter(x=>x.phase==='norm').sort((a,b)=>a.active_features.length-b.active_features.length).forEach(x=>{comp+=`<tr><th>${esc(labelFeatures(x.active_features))}</th><td>${x.n_joint}</td><td><b>${pct(x.all_active_ge_4.rate)}</b><br><span class="small">${pct(x.all_active_ge_4.ci95_low)} … ${pct(x.all_active_ge_4.ci95_high)}</span></td><td>${x.minimum_active_score.mean.toFixed(2)}</td><td>${x.active_features.map(f=>`${names[f]?.short||f}: ${x.feature_scores[f].mean.toFixed(2)}`).join(' · ')}</td></tr>`});
document.querySelector('#composition').innerHTML=comp+`</table><p class="callout"><b>Как читать:</b> для candor + concrete получается 96.1% joint при ожидаемых 96.1%; для всех четырёх rank-1 — 31.3% при ориентире 28.5%. Нормализованный all-four поднимается до 46.1%, но bottleneck casualness всё равно остаётся.</p>`;

const profiles=[['GDN rank 1','svd','per_r1_1111'],['GDN rank 4','svd','per_r4_1111'],['GDN RSS norm','norm','norm_1111']];
let profile=`<h2>Что именно теряется или сохраняется?</h2><p class="note">Средний score по четырём активным признакам в полном составе. Это уже уровень выраженности (1–5), не бинарная вероятность.</p><table class="heat"><tr><th>Метод</th>${featureOrder.map(f=>`<th>${names[f].short}</th>`).join('')}<th>Минимум</th></tr>`;
profiles.forEach(([label,phase,condition])=>{const s=find(phase,condition);profile+=`<tr><th>${label}</th>${featureOrder.map(f=>`<td>${cellScore(s.feature_scores[f].mean)}<br><span class="small">[${s.feature_scores[f].ci95_low.toFixed(2)}, ${s.feature_scores[f].ci95_high.toFixed(2)}]</span></td>`).join('')}<td>${s.minimum_active_score.mean.toFixed(2)}</td></tr>`});
profile+=`</table><h3 style="margin-top:18px">Полный rank-1 состав против одиночного признака</h3><table><tr><th>Признак</th><th>Одиночный mean</th><th>Полный состав mean</th><th>Δ score</th><th>Интерпретация</th></tr>`;
const fullS=find('svd','per_r1_1111');
featureOrder.forEach((f,i)=>{const single=find('svd',`per_r1_${(1<<i).toString(2).padStart(4,'0')}`);const d=A.rank1_four_way_retention[f];profile+=`<tr><th>${names[f].ru}</th><td>${single.feature_scores[f].mean.toFixed(2)}</td><td>${fullS.feature_scores[f].mean.toFixed(2)}</td><td class="${cclass(d.difference)}">${signed(d.difference)} [${signed(d.ci95_low)}, ${signed(d.ci95_high)}]</td><td>${f==='casualness'?'score растёт, но остаётся ниже 3/5 — это bottleneck, а не успех 4/5':f==='principled_candor'?'в полном составе немного слабее, но остаётся выше 4/5':'сохраняется или не меняется заметно'}</td></tr>`});
document.querySelector('#profile').innerHTML=profile+`</table><p class="note">Δ score — парная разница 1–5 между полным и одиночным запуском на тех же prompt'ах. Её нельзя читать как «процент ответов с признаком».</p>`;

const holdLabels={baseline:'Без steering',gdn_rank1:'GDN rank 1',gdn_rank4:'GDN rank 4',gdn_norm:'GDN RSS norm',activation_l10_a4:'Activation L10 α=4',activation_l20_a1:'Activation L20 α=1'};
let act=`<h2>GDN против классического activation steering</h2><p class="note">Это честнее сравнивать на 96 holdout prompt'ах. Activation L10/α=4 и L20/α=1 были выбраны на dev-части из 32 prompt'ов, затем проверены на holdout.</p><h3>Dev sweep layer × α</h3><table class="heat"><tr><th>Layer ↓ / α →</th><th>0.5</th><th>1</th><th>2</th><th>4</th></tr>`;
[[10,'l10'],[20,'l20'],[30,'l30']].forEach(([layer,prefix])=>{act+=`<tr><th>${layer}</th>`;[0.5,1,2,4].forEach(alpha=>{const x=find('activation',`${prefix}_a${alpha}_all4`);act+=`<td>${pct(x.all_active_ge_4.rate)}<br><span class="small">N=${x.n_joint}</span></td>`});act+='</tr>'});
act+=`</table><p class="note">Sweep — только dev, поэтому его нельзя читать как честную оценку выбранного максимума. Ниже — заранее выделенный 96-prompt holdout.</p><table><tr><th>Метод</th><th>Все 4 ≥4/5</th><th>Средний минимум</th><th>Candor</th><th>Concrete</th><th>Casual</th><th>Optimism</th></tr>`;
Object.entries(hold).forEach(([id,x])=>{act+=`<tr><th>${holdLabels[id]||id}</th><td><b>${pct(x.joint_ge4.rate)}</b><br><span class="small">[${pct(x.joint_ge4.ci95_low)}, ${pct(x.joint_ge4.ci95_high)}]</span></td><td>${x.minimum.mean.toFixed(2)}</td>${featureOrder.map(f=>`<td>${x.feature_means[f].toFixed(2)}</td>`).join('')}</tr>`});
const aNorm=A.holdout.comparisons.activation_l10_a4_minus_gdn_norm.joint_ge4;
document.querySelector('#activation').innerHTML=act+`</table><p class="callout"><b>Activation L10/α=4 выглядит выше GDN norm:</b> +${pp(aNorm.difference)} на holdout, но CI [${pp(aNorm.ci95_low)}, ${pp(aNorm.ci95_high)}] пересекает ноль. Поэтому это сигнал в пользу дальнейшего сравнения, а не доказательство превосходства классического steering.</p>`;

const joyRows=[1,2,4,8];let joy=`<h2>Отдельная проверка: joy + optimism</h2><p class="note">Здесь α увеличивали для joy. Это наглядная композиция двух эмоциональных направлений, но α=8 выбран по этой же серии, поэтому результат exploratory.</p><table><tr><th>α joy</th><th>Joy alone ≥4/5</th><th>Joy + optimism: оба ≥4/5</th><th>Mean joy + optimism</th></tr>`;
joyRows.forEach(a=>{const j=C.joy_optimism[`joy_a${a}`],jo=C.joy_optimism[`joy_a${a}_optimism`];joy+=`<tr><th>${a}</th><td>${pct(j.joint_ge4)}</td><td class="${a===8?'good':''}"><b>${pct(jo.joint_ge4)}</b></td><td>${jo.feature_means[0].toFixed(2)} / ${jo.feature_means[1].toFixed(2)}</td></tr>`});
document.querySelector('#joy').innerHTML=joy+`</table><p class="callout">При α=8 оба признака проходят порог в 85.2% ответов. Это самый красивый результат по совместному endpoint, но его нужно повторить на заранее отложенном holdout, прежде чем считать confirmatory.</p>`;

let exp2=`<h2>Что добавляет предыдущий factorial-тест?</h2><p class="note">Dashboard 1 измерял не достигнутый score, а строгий signed A/B effect: ON выигрывает у OFF только при согласии двух порядков ответа. Поэтому числа ниже не надо складывать с 1–5 score из Exp3.</p><table><tr><th>Признак</th><th>Standalone effect</th><th>В полном составе</th><th>Вывод</th></tr>`;
E.main_effects.forEach(x=>{const full=E.full_context_effects.find(y=>y.feature===x.feature);exp2+=`<tr><th>${names[x.feature]?.ru||x.label}</th><td class="${cclass(x.effect)}">${signed(x.effect)}<br><span class="small">[${signed(x.ci_low)}, ${signed(x.ci_high)}]</span></td><td class="${cclass(full.effect)}">${signed(full.effect)}<br><span class="small">[${signed(full.ci_low)}, ${signed(full.ci_high)}]</span></td><td>${x.verdict}</td></tr>`});
exp2+=`</table><h3 style="margin-top:18px">Interaction-сигналы</h3><table><tr><th>Target</th><th>Context</th><th>Δ эффекта</th><th>Смысл</th></tr>`;
E.interactions.forEach(x=>{exp2+=`<tr><th>${names[x.target]?.ru||x.target}</th><td>${names[x.context]?.ru||x.context}</td><td class="${cclass(x.effect)}">${signed(x.effect)} [${signed(x.ci_low)}, ${signed(x.ci_high)}]</td><td>${x.verdict}</td></tr>`});
exp2+=`</table><h3 style="margin-top:18px">Языковые комбинации из старого теста</h3><table><tr><th>Комбинация</th><th>Результат</th><th>Интерпретация</th></tr>`;
E.pair_tests.forEach(p=>{exp2+=`<tr><th>${p.name}</th><td>${p.rows.map(r=>`${names[r.feature]?.ru||r.feature}: ${signed(r.standalone)} → <b>${signed(r.combined)}</b>`).join('<br>')}</td><td>${p.note}</td></tr>`});
document.querySelector('#exp2').innerHTML=exp2+`</table><p class="callout">Согласующаяся картина: три сильных оси не исчезают при добавлении других, но взаимодействия не равны нулю. Особенно аккуратно нужно относиться к casualness и к неудачному calm-направлению.</p>`;

document.querySelector('#insights').innerHTML=`<h2>Инсайты для обсуждения</h2><div class="insights">
 <article class="insight"><h3>1. Нормализация — главный выигрыш GDN</h3><p>RSS-нормализация подняла joint rate с ${pct(rank1.joint_ge4)} до ${pct(norm.joint_ge4)}: ${pp(normDelta.difference)} [${pp(normDelta.ci95_low)}, ${pp(normDelta.ci95_high)}]. Это сильнее, чем просто оставить rank 4.</p></article>
 <article class="insight"><h3>2. Комбинация частично работает</h3><p>Для candor + concrete наблюдается ${pct(A.rank1_composition.conditions.find(x=>x.condition==='per_r1_0011').joint_ge4)} совместного успеха, а для всех четырёх — ${pct(A.rank1_composition.conditions.find(x=>x.condition==='per_r1_1111').joint_ge4)}. Это не «всё или ничего»: слабый признак снижает intersection.</p></article>
 <article class="insight"><h3>3. Casualness — узкое место</h3><p>Одиночный casualness даёт только ${pct(A.rank1_composition.conditions.find(x=>x.condition==='per_r1_0100').joint_ge4)} ≥4/5, тогда как candor — ${pct(A.rank1_composition.conditions.find(x=>x.condition==='per_r1_0001').joint_ge4)}. Поэтому низкий all-four rate нельзя автоматически приписывать плохому сложению всех направлений.</p></article>
 <article class="insight"><h3>4. Rank 1 полезен, но не магический</h3><p>Для candor + concrete rank 1 почти полностью сохраняет оба признака; для полного four-way состава joint rate уже ${pct(rank1.joint_ge4)}. Rank 4 здесь не имеет статистически подтверждённого преимущества.</p></article>
 <article class="insight"><h3>5. Classical steering пока только конкурент</h3><p>На holdout activation L10/α=4 даёт ${pct(hold.activation_l10_a4.joint_ge4.rate)} против ${pct(hold.gdn_norm.joint_ge4.rate)} у GDN norm, но CI разницы пересекает ноль. Нельзя заявлять превосходство ни одной стороны.</p></article>
 <article class="insight"><h3>6. Joy + optimism выглядит чисто</h3><p>При α=8 оба эмоциональных признака проходят 4/5 в ${pct(C.joy_optimism.joy_a8_optimism.joint_ge4)} ответов. Следующий шаг — независимый holdout, а не ещё один sweep на тех же prompt'ах.</p></article>
</div>`;

document.querySelector('#limits').innerHTML=`<h2>Ограничения и статус доказательности</h2><div class="grid">
 <div class="third"><h3 class="warn">Judge</h3><p class="note">На сервере эти результаты получены старым compact-путём: модель возвращала одну цифру 1–5, provenance — <span class="mono">trait_compact_v1</span>, модель <span class="mono">gpt-4o-mini</span>. JSONL — оболочка раннера; current v3 на этом наборе отдельно не прогонялся.</p></div>
 <div class="third"><h3 class="warn">Endpoint</h3><p class="note">All-four ≥4/5 очень строг: один недотянутый trait делает весь ответ неуспешным. Он хорошо показывает совместное присутствие, но не заменяет средний score или probability-weighted endpoint.</p></div>
 <div class="third"><h3 class="warn">Выбор α</h3><p class="note">Activation и joy α выбирались по dev/той же серии. Holdout для activation отделён, а joy+optimism пока нет. Поэтому красивые проценты joy и dev sweep остаются exploratory.</p></div>
 <div class="third"><h3 class="warn">Что не доказано</h3><p class="note">Нет человеческой калибровки, отдельной оценки answer quality, raw full-rank baseline для каждого нормализованного варианта и повторения на другой модели/распределении.</p></div>
 <div class="third"><h3 class="good">Что уже можно сказать</h3><p class="note">Нормализация — наиболее убедительный GDN-абляционный сигнал; rank 1 поддерживает часть комбинаций; casualness ограничивает full intersection; interactions не являются повсеместно нулевыми.</p></div>
 <div class="third"><h3 class="good">Рациональный следующий эксперимент</h3><p class="note">Заранее выбрать α на dev, затем сравнить raw GDN rank 1, RSS-normalized rank 1 и activation steering на одном независимом holdout для 2-, 3- и 4-feature комбинаций, используя текущий Judge v3 и отдельный quality endpoint.</p></div>
</div><p class="source">Дашборд не изменяет исходные JSON и не пересчитывает статистику: он только связывает сводки Exp3 с ключевыми результатами Dashboard 1 и объясняет, как их читать вместе.</p>`;
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
