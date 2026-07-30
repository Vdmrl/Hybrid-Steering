"""Build the second, analysis-first dashboard as one self-contained HTML file."""

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
DEFAULT_OUTPUT = ROOT / "outputs/meeting-dashboard-2/index.html"


def build(summary: dict, comparisons: dict) -> str:
    payload = json.dumps(
        {"summary": summary, "comparisons": comparisons},
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return HTML.replace("__DATA__", payload).replace("__GENERATED__", generated)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--comparisons", type=Path, default=DEFAULT_COMPARISONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        build(
            json.loads(args.summary.read_text(encoding="utf-8")),
            json.loads(args.comparisons.read_text(encoding="utf-8")),
        ),
        encoding="utf-8",
    )
    print(args.output)


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hybrid Steering · Analysis dashboard</title>
<style>
:root{--bg:#090d14;--panel:#111824;--line:#263245;--text:#eef3fb;--muted:#91a0b5;
--blue:#7c8cff;--cyan:#58d6d0;--green:#55ce91;--amber:#f3bd62;--red:#f47777}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:14px/1.45 Inter,ui-sans-serif,system-ui,sans-serif}header,main{max-width:1240px;
margin:auto;padding:24px}header{padding-top:38px}h1{font-size:34px;line-height:1.05;
margin:5px 0 12px}h2{font-size:19px;margin:0 0 6px}h3{font-size:14px;margin:0 0 10px}
p{margin:5px 0}.eyebrow{color:var(--cyan);font-weight:700;text-transform:uppercase;
letter-spacing:.08em}.muted,.note{color:var(--muted)}.note{font-size:13px}.grid{display:grid;
grid-template-columns:repeat(12,1fr);gap:14px}.panel{grid-column:span 12;background:var(--panel);
border:1px solid var(--line);border-radius:14px;padding:19px;overflow:auto}.half{grid-column:span 6}
.third{grid-column:span 4}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;
margin-top:16px}.card{background:#0b111b;border:1px solid var(--line);border-radius:10px;
padding:13px}.card strong{display:block;font-size:25px}.good{color:var(--green)}
.warn{color:var(--amber)}.bad{color:var(--red)}table{width:100%;border-collapse:collapse;
margin-top:14px}th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;
white-space:nowrap}th:first-child,td:first-child{text-align:left}th{color:#bdc8d8;font-size:12px}
tr:last-child td{border:0}.barrow{display:grid;grid-template-columns:185px 1fr 68px;gap:10px;
align-items:center;margin:11px 0}.track{height:22px;background:#0a1019;border-radius:5px;position:relative}
.bar{height:100%;border-radius:5px;background:linear-gradient(90deg,var(--blue),var(--cyan))}
.ci{position:absolute;height:2px;background:#fff;top:10px}.ci:before,.ci:after{content:"";
position:absolute;width:2px;height:8px;background:#fff;top:-3px}.ci:after{right:0}.value{text-align:right}
.legend{display:flex;gap:15px;color:var(--muted);font-size:12px;margin:8px 0}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}
.heat td{font-weight:700;text-align:center;min-width:82px}.heat th{text-align:center}
.heat th:first-child{text-align:left}.svg{width:100%;min-width:540px;height:auto}.axis{stroke:#3b485c}
.tick{fill:var(--muted);font-size:11px}.line{fill:none;stroke-width:3}.point{stroke:#0b111b;
stroke-width:2}.feature-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:12px}
.feature{background:#0b111b;border-radius:8px;padding:10px}.feature b{display:block}
.feature span{color:var(--muted);font-size:12px}.tag{display:inline-block;border:1px solid var(--line);
border-radius:99px;padding:2px 8px;font-size:11px;color:var(--muted)}.callout{border-left:3px solid
var(--cyan);padding-left:12px;margin:12px 0}.ranked td:nth-child(3){font-weight:700}
@media(max-width:850px){.half,.third{grid-column:span 12}.cards,.feature-grid{grid-template-columns:1fr 1fr}}
@media(max-width:520px){header,main{padding:16px}.cards,.feature-grid{grid-template-columns:1fr}
.barrow{grid-template-columns:120px 1fr 55px}}
</style>
</head>
<body>
<header>
  <div class="eyebrow">Qwen3.5-9B · recurrent-state steering</div>
  <h1>Composition analysis dashboard · v2</h1>
  <p class="muted">Not a run log: each view answers one experimental question.
  Built __GENERATED__.</p>
  <div class="feature-grid">
    <div class="feature"><b>Principled candor</b><span>принципиальная прямота ↔ угодливость</span></div>
    <div class="feature"><b>Concrete language</b><span>конкретный язык ↔ абстрактный язык</span></div>
    <div class="feature"><b>Casualness</b><span>неформальность ↔ формальность</span></div>
    <div class="feature"><b>Optimism</b><span>оптимизм ↔ пессимизм</span></div>
  </div>
</header>
<main class="grid">
  <section class="panel" id="overview"></section>
  <section class="panel half" id="method-bars"></section>
  <section class="panel half" id="bottleneck"></section>
  <section class="panel half" id="svd"></section>
  <section class="panel half" id="activation"></section>
  <section class="panel half" id="composition"></section>
  <section class="panel half" id="joy"></section>
  <section class="panel" id="ranked"></section>
  <section class="panel" id="notes"></section>
</main>
<script>
const DATA=__DATA__;
const conditions=DATA.summary.conditions;
const comparisons=DATA.comparisons;
const FEATURES=["principled_candor","concrete_language","casualness","optimism"];
const F={principled_candor:"Candor",concrete_language:"Concrete",casualness:"Casualness",optimism:"Optimism",joy:"Joy"};
const METHODS={
  per_r1_1111:"Per-feature SVD rank 1",
  per_r4_1111:"Per-feature SVD rank 4",
  post_r1_1111:"Post-sum SVD rank 1",
  post_r4_1111:"Post-sum SVD rank 4",
  norm_1111:"Norm-controlled GDN"
};
const by=(phase,name)=>conditions.find(x=>x.phase===phase&&x.condition===name);
const pct=x=>(100*x).toFixed(1)+"%";
const esc=x=>String(x).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function confidenceTag(n){return n>=96?'<span class="tag good">confirmatory N='+n+'</span>':'<span class="tag warn">exploratory N='+n+'</span>'}
function heat(v,min=0,max=1){let t=Math.max(0,Math.min(1,(v-min)/(max-min)));return `rgba(${Math.round(244-160*t)},${Math.round(87+111*t)},${Math.round(105+45*t)},${.18+.55*t})`}
function bar(label,value,low,high){
  return `<div class="barrow"><span>${esc(label)}</span><div class="track"><div class="bar" style="width:${value*100}%"></div>
  <i class="ci" style="left:${low*100}%;width:${Math.max(0,(high-low)*100)}%"></i></div><b class="value">${pct(value)}</b></div>`}

const all4=Object.entries(METHODS).map(([name,label])=>({label,...by(name.startsWith("norm")?"norm":"svd",name)}));
const best=all4.reduce((a,b)=>a.all_active_ge_4.rate>b.all_active_ge_4.rate?a:b);
const casual=best.feature_scores.casualness.mean;
const activationRows=conditions.filter(x=>x.phase==="activation"&&x.condition.endsWith("_all4")&&x.condition!=="baseline");
const activationBest=activationRows.reduce((a,b)=>a.all_active_ge_4.rate>b.all_active_ge_4.rate?a:b);
const activationConfirmed=conditions.filter(x=>x.phase==="activation_holdout");
const activationWinner=activationConfirmed.filter(x=>x.condition!=="baseline").reduce((a,b)=>a.all_active_ge_4.rate>b.all_active_ge_4.rate?a:b,activationBest);
const joy8=by("joy","joy_a8_optimism");
document.querySelector("#overview").innerHTML=`<h2>What do the results say?</h2>
<div class="cards">
 <div class="card"><span>Best full GDN method</span><strong>${pct(best.all_active_ge_4.rate)}</strong><small>${METHODS[best.condition]} · all 4 ≥ 4/5</small></div>
 <div class="card"><span>Main bottleneck</span><strong class="warn">${casual.toFixed(2)}/5</strong><small>Casualness under ${METHODS[best.condition]}</small></div>
 <div class="card"><span>${activationConfirmed.length?"Best activation holdout":"Best activation sweep"}</span><strong>${pct(activationWinner.all_active_ge_4.rate)}</strong><small>${activationWinner.condition.replaceAll("_"," ")} · ${confidenceTag(activationWinner.n_joint)}</small></div>
 <div class="card"><span>Joy + optimism, α=8</span><strong>${pct(joy8.all_active_ge_4.rate)}</strong><small>both traits ≥ 4/5 · N=${joy8.n_joint}</small></div>
</div>
<p class="callout"><b>Strongest current result:</b> norm control improves four-way GDN joint success by
14.8 percentage points over per-feature rank 1; paired bootstrap 95% CI [+5.5, +24.2] pp.
Activation looks stronger, and ${activationConfirmed.length?"the separate 96-prompt holdout below preserves the exploratory/confirmatory boundary":"its best setting was selected on the same 32 prompts and still needs holdout confirmation"}.</p>`;

document.querySelector("#method-bars").innerHTML=`<h2>Which GDN composition method works best?</h2>
<p class="note">Strict joint endpoint: all four independently scored traits are at least 4/5 on the same answer.</p>
${all4.map(x=>bar(x.label,x.all_active_ge_4.rate,x.all_active_ge_4.ci95_low,x.all_active_ge_4.ci95_high)).join("")}
<div class="legend"><span><i class="dot" style="background:var(--cyan)"></i>rate</span><span>white whisker = 95% Wilson CI</span></div>`;

let bottleneck=`<h2>Where does composition fail?</h2><p class="note">Mean 1–5 score for each trait in four-way answers.</p>
<table class="heat"><tr><th>Method</th>${FEATURES.map(x=>`<th>${F[x]}</th>`).join("")}<th>minimum</th></tr>`;
all4.forEach(x=>{bottleneck+=`<tr><td>${METHODS[x.condition]}</td>${FEATURES.map(f=>`<td style="background:${heat(x.feature_scores[f].mean,1,5)}">${x.feature_scores[f].mean.toFixed(2)}</td>`).join("")}<td>${x.minimum_active_score.mean.toFixed(2)}</td></tr>`});
document.querySelector("#bottleneck").innerHTML=bottleneck+"</table><p class='note'>Casualness is consistently the weakest axis; a high aggregate rate cannot hide it here.</p>";

const deltas=comparisons.all4_vs_rank1;
let svd=`<h2>Does extra SVD rank help?</h2><p class="note">Paired differences against per-feature rank 1 on the same 128 prompts.</p>
<table><tr><th>Comparison</th><th>Δ joint success</th><th>95% CI</th><th>Interpretation</th></tr>`;
Object.entries(deltas).forEach(([name,x])=>{const d=x.joint_ge4;const clear=d.ci95_low>0||d.ci95_high<0;svd+=`<tr><td>${esc(name.replace("_minus_gdn_svd_per_rank1","").replaceAll("_"," "))}</td><td class="${d.difference>0?"good":"bad"}">${(100*d.difference).toFixed(1)} pp</td><td>[${(100*d.ci95_low).toFixed(1)}, ${(100*d.ci95_high).toFixed(1)}]</td><td>${clear?"clear difference":"inconclusive"}</td></tr>`});
document.querySelector("#svd").innerHTML=svd+`</table><p class="callout">Rank 4 is not reliably better than rank 1. Compressing only after summing to rank 1 is reliably worse. Norm control is the only clear improvement here.</p>`;

const layers=[10,20,30], alphas=[.5,1,2,4];
let act=`<h2>Classical activation steering: layer × α</h2><p class="note">Cell = four-way joint success. The original 32-prompt sweep is exploratory.</p>
<table class="heat"><tr><th>Layer ↓ / α →</th>${alphas.map(a=>`<th>${a}</th>`).join("")}</tr>`;
layers.forEach(l=>{act+=`<tr><th>${l}</th>`;alphas.forEach(a=>{const x=by("activation",`l${l}_a${a}_all4`);act+=`<td style="background:${heat(x.all_active_ge_4.rate)}">${pct(x.all_active_ge_4.rate)}<br><small>N=${x.n_joint}</small></td>`});act+="</tr>"});
act+="</table>";
if(activationConfirmed.length){act+=`<h3 style="margin-top:18px">Held-out confirmation</h3><p class="note">The untouched 96 prompts only; not pooled with the 32-prompt sweep.</p>${activationConfirmed.map(x=>bar(x.condition.replaceAll("_"," "),x.all_active_ge_4.rate,x.all_active_ge_4.ci95_low,x.all_active_ge_4.ci95_high)).join("")}`}
else act+=`<p class="callout warn">Holdout is pending for layer 10 / α=4 and layer 20 / α=1. Do not present 62.5% as confirmatory yet.</p>`;
document.querySelector("#activation").innerHTML=act;

function compositionRows(prefix,phase){
 return conditions.filter(x=>x.phase===phase&&x.condition.startsWith(prefix)&&x.active_features.length>=2)
}
const rank1=compositionRows("per_r1_","svd"), norm=conditions.filter(x=>x.phase==="norm");
function depth(rows,size){const xs=rows.filter(x=>x.active_features.length===size);return xs.length?xs.reduce((s,x)=>s+x.all_active_ge_4.rate,0)/xs.length:null}
const depths=[2,3,4];
let comp=`<h2>Does performance decay with more traits?</h2><p class="note">Mean joint success across all available combinations of each size.</p>
<table><tr><th>Method</th>${depths.map(x=>`<th>${x} traits</th>`).join("")}</tr>`;
[[METHODS.per_r1_1111,rank1],[METHODS.norm_1111,norm]].forEach(([label,rows])=>{comp+=`<tr><td>${label}</td>${depths.map(n=>{const value=depth(rows,n);return value===null?'<td>—</td>':`<td style="background:${heat(value)}">${pct(value)}</td>`}).join("")}</tr>`});
document.querySelector("#composition").innerHTML=comp+`</table><p class="callout">Composition is not uniformly hard: combinations containing Casualness dominate failures. Use the ranked table below before attributing decay only to the number of traits.</p>`;

const joys=[1,2,4,8].map(a=>({a,single:by("joy",`joy_a${a}`),pair:by("joy",`joy_a${a}_optimism`)}));
function lineChart(series){
 const W=600,H=250,L=45,R=15,T=15,B=35,x=i=>L+i*(W-L-R)/3,y=v=>T+(1-v)*(H-T-B);
 let out=`<svg class="svg" viewBox="0 0 ${W} ${H}">`;
 [0,.25,.5,.75,1].forEach(v=>out+=`<line class="axis" x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}"/><text class="tick" x="5" y="${y(v)+4}">${pct(v)}</text>`);
 series.forEach((s,j)=>{let pts=s.values.map((v,i)=>`${x(i)},${y(v)}`).join(" ");out+=`<polyline class="line" stroke="${s.color}" points="${pts}"/>`;s.values.forEach((v,i)=>out+=`<circle class="point" fill="${s.color}" cx="${x(i)}" cy="${y(v)}" r="5"/>`)});
 joys.forEach((d,i)=>out+=`<text class="tick" text-anchor="middle" x="${x(i)}" y="${H-8}">α=${d.a}</text>`);
 return out+"</svg>";
}
document.querySelector("#joy").innerHTML=`<h2>Can a simple emotional pair compose?</h2><p class="note">This sweep is a useful positive control: Joy becomes much more detectable as α grows while Optimism remains present.</p>
${lineChart([{values:joys.map(x=>x.single.all_active_ge_4.rate),color:"#7c8cff"},{values:joys.map(x=>x.pair.all_active_ge_4.rate),color:"#55ce91"}])}
<div class="legend"><span><i class="dot" style="background:#7c8cff"></i>Joy alone ≥4</span><span><i class="dot" style="background:#55ce91"></i>Joy + Optimism both ≥4</span></div>`;

const rankedRows=[...rank1,...norm].filter(x=>x.active_features.length<4).sort((a,b)=>b.all_active_ge_4.rate-a.all_active_ge_4.rate);
let ranked=`<h2>Which concrete combinations work?</h2><p class="note">This separates “four traits are hard” from “one particular trait is weak”. Sorted by strict joint success.</p>
<table class="ranked"><tr><th>Method</th><th>Active traits</th><th>All ≥4/5</th><th>95% CI</th><th>Mean minimum</th><th>N</th></tr>`;
rankedRows.forEach(x=>ranked+=`<tr><td>${x.phase==="norm"?"Norm-controlled":"SVD rank 1"}</td><td>${x.active_features.map(f=>F[f]).join(" + ")}</td><td>${pct(x.all_active_ge_4.rate)}</td><td>[${pct(x.all_active_ge_4.ci95_low)}, ${pct(x.all_active_ge_4.ci95_high)}]</td><td>${x.minimum_active_score.mean.toFixed(2)}</td><td>${x.n_joint}</td></tr>`);
document.querySelector("#ranked").innerHTML=ranked+"</table>";

document.querySelector("#notes").innerHTML=`<h2>Reading rules and current limits</h2>
<div class="grid"><div class="third"><h3>Primary endpoint</h3><p class="note">Each active trait is judged independently on an anchored 1–5 scale. Joint success means every active trait scored at least 4 on the same answer.</p></div>
<div class="third"><h3>Uncertainty</h3><p class="note">Rates show Wilson 95% intervals. Method differences in the SVD table use paired prompt bootstrap intervals.</p></div>
<div class="third"><h3>Not measured yet</h3><p class="note">Answer quality, inactive-trait leakage, and human agreement are not part of this dashboard. Activation sweep winners require held-out confirmation.</p></div></div>`;
</script>
</body></html>"""


if __name__ == "__main__":
    main()
