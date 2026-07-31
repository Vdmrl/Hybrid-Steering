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


def build(summary: dict, selection: dict) -> str:
    payload = json.dumps(
        {"summary": summary, "selection": selection},
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return HTML.replace("__DATA__", payload).replace("__GENERATED__", generated)


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
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 Inter,system-ui,sans-serif}header,main{max-width:1320px;margin:auto;padding:24px}header{padding-top:36px}h1{font-size:36px;line-height:1.08;margin:6px 0 12px}h2{font-size:20px;margin:0 0 8px}h3{font-size:15px;margin:0 0 7px}.eyebrow{color:var(--cyan);font-weight:800;letter-spacing:.08em;text-transform:uppercase}.muted,.note{color:var(--muted)}.note{font-size:13px}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}.panel{grid-column:span 12;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px;overflow:auto}.half{grid-column:span 6}.third{grid-column:span 4}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:16px 0}.card{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:13px}.card strong{display:block;font-size:27px;margin:4px 0}.good{color:var(--green)}.warn{color:var(--amber)}.bad{color:var(--red)}table{width:100%;border-collapse:collapse;margin-top:14px}th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}th:first-child,td:first-child{text-align:left}th{color:#c9d6e7;font-size:12px;white-space:nowrap}tr:last-child td,tr:last-child th{border-bottom:0}.callout{border-left:3px solid var(--cyan);padding-left:12px;margin:14px 0}.callout.warn{border-color:var(--amber)}.callout.bad{border-color:var(--red)}.small{font-size:12px;color:var(--muted)}.tag{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:2px 8px;font-size:11px;color:var(--muted);margin:2px}.formula{font:13px ui-monospace,SFMono-Regular,Consolas,monospace;background:#08101a;border:1px solid var(--line);padding:11px;border-radius:8px;white-space:pre-wrap}.insights{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.insight{background:var(--panel2);border-radius:10px;padding:13px;border-top:3px solid var(--cyan)}.insight.warn{border-top-color:var(--amber)}.source{font-size:12px;color:var(--muted);margin-top:16px}.mono{font:12px ui-monospace,SFMono-Regular,Consolas,monospace}.missing{color:var(--muted);font-style:italic}.sticky th{position:sticky;top:0;background:var(--panel)}
@media(max-width:950px){.half,.third{grid-column:span 12}.cards,.insights{grid-template-columns:1fr 1fr}}@media(max-width:560px){header,main{padding:16px}.cards,.insights{grid-template-columns:1fr}h1{font-size:29px}}
</style></head>
<body><header>
<div class="eyebrow">Hybrid Steering · Experiment 3 · Qwen3.5-9B</div>
<h1>Реальные результаты composition-normalization-v3</h1>
<p class="muted">Это исправленная версия Dashboard 3: она читает именно завершённый summary эксперимента 3, а не старый factorial-run. Здесь 64 условия, 128 тестовых промптов и Judge с вероятностным expected score. Построено __GENERATED__.</p>
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
const signed=x=>(x>=0?'+':'')+Number(x).toFixed(3); const pp=x=>(100*Number(x)).toFixed(1)+' п.п.';
const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const cls=x=>x>0?'good':x<0?'bad':'warn';
const cond=id=>S.conditions.find(x=>x.condition===id);
const byId={}; S.conditions.forEach(x=>byId[x.condition]=x);
const maskFeatures=mask=>FEATURES.filter((_,i)=>mask[i]==='1');
const labels=fs=>fs.map(F).join(' + ');
function ci(x){return `[${fmt(x.ci95_low)}, ${fmt(x.ci95_high)}]`}
function rateCI(p,n){const z=1.96,den=1+z*z/n,center=(p+z*z/(2*n))/den,half=z*Math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return [Math.max(0,center-half),Math.min(1,center+half)]}
function rowName(id){return M[id]||id.replaceAll('_',' ')}
function scoreCells(x, only=FEATURES){return only.map(f=>`<td>${fmt(x.feature_expected_means[f])}</td>`).join('')}

const base=byId.baseline, full=byId.gdn_rss_r4_1111;
document.querySelector('#presentation').innerHTML=`<h2>Тезисы для показа</h2><div class="insights">
<article class="insight"><h3>1. GDN-композиция возможна</h3><p>Для четырёх признаков joint endpoint вырос с ${pct(base.all_active_ge4)} у baseline до ${pct(full.all_active_ge4)} у RSS rank 4. Это частичное, но наблюдаемое совместное сохранение режимов.</p></article>
<article class="insight"><h3>2. Самый чистый методический сигнал</h3><p>GDN raw rank 1 − activation raw = +0.092 по mean minimum; 95% CI [+0.016, +0.172]. Rank 4 RSS выше rank 1 RSS на +0.106.</p></article>
<article class="insight warn"><h3>3. Что ещё нельзя утверждать</h3><p>RSS rank 4 — лучший абсолютный результат, но Exp3 не содержал raw rank 4 и не закрывал full-rank control. Это проверяет текущий Exp4.</p></article>
</div><p class="source">Если Exp4 не завершится к показу, этот Dashboard 3 уже содержит самостоятельный завершённый результат: baseline, GDN/activation, rank, RSS, пары, тройки, all-four и quality.</p>`;
document.querySelector('#overview').innerHTML=`<h2>Короткий ответ</h2><div class="cards">
<div class="card"><span>Baseline: все 4 ≥4</span><strong>${pct(base.all_active_ge4)}</strong><small>N=${base.n}; качество ${fmt(base.quality_mean)}/5</small></div>
<div class="card"><span>GDN raw rank 1</span><strong>${pct(byId.gdn_raw_r1_1111.all_active_ge4)}</strong><small>mean minimum ${fmt(byId.gdn_raw_r1_1111.mean_minimum_expected)}/5</small></div>
<div class="card"><span>GDN RSS rank 4</span><strong class="good">${pct(full.all_active_ge4)}</strong><small>mean minimum ${fmt(full.mean_minimum_expected)}/5</small></div>
<div class="card"><span>Лучший доказанный контраст</span><strong class="good">${pp(0.0918334765625)}</strong><small>GDN raw rank1 − activation raw, all-four</small></div>
</div><p class="callout"><b>Главный вывод:</b> комбинация признаков работает не как «всё или ничего». Для всех четырёх признаков одновременно строгий endpoint вырос с ${pct(base.all_active_ge4)} до ${pct(full.all_active_ge4)} у RSS rank 4, но это частичный эффект. Самый устойчивый сигнал — GDN против activation и переход к rank 4; RSS на rank 1 преимуществ не показал.</p>`;

document.querySelector('#features').innerHTML=`<h2>Что именно проверяли</h2><p class="note">Битовая маска читается слева направо как <b>joy · concrete · optimism · candor</b>. В каждой строке Judge независимо ставит каждому признаку expected score на шкале 1–5 и отдельно оценивает качество ответа. «Все ≥4» означает, что у одного ответа все активные признаки получили минимум 4.</p><div class="cards">
${FEATURES.map(f=>`<div class="card"><b>${N[f][0]}</b><strong>${F(f)}</strong><small>Оценивается поведение ответа, а не его полезность или длина.</small></div>`).join('')}</div>
<h3>Выбранные alpha для matched-strength</h3><table><tr><th>Признак</th><th>Целевой dev score</th><th>Выбранный GDN alpha</th><th>Почему</th></tr>${FEATURES.map(f=>`<tr><th>${F(f)}</th><td>${fmt(SEL[f].matched_target)}</td><td>${SEL[f].gdn.selected_alpha}</td><td>Ближайший к target при quality-safe выборе</td></tr>`).join('')}</table>
<div class="formula">mean_minimum = среднее по ответам минимального expected score среди активных признаков
joint_ge4 = доля ответов, где каждый активный признак получил score ≥ 4
expected_score = probability-weighted среднее Judge; это чувствительнее целого trait_score</div>`;

const fullIds=['baseline','act_raw_1111','act_rss_1111','gdn_raw_r1_1111','gdn_rss_r1_1111','gdn_rss_r4_1111'];
let all=`<h2>Все четыре признака одновременно</h2><p class="note">Это главный joint endpoint. Для baseline активных признаков нет, но мы всё равно показываем, как часто естественный ответ случайно проходит тот же порог по четырём шкалам.</p><table class="sticky"><tr><th>Условие</th><th>Все 4 ≥4</th><th>95% CI</th><th>Mean minimum</th><th>Качество</th>${FEATURES.map(f=>`<th>${N[f][0]}</th>`).join('')}</tr>`;
fullIds.forEach(id=>{const x=byId[id],ciRate=rateCI(x.all_active_ge4,x.n); all+=`<tr><th>${rowName(id)}</th><td class="${x.all_active_ge4>=.1?'good':'warn'}"><b>${pct(x.all_active_ge4)}</b></td><td>${pct(ciRate[0])} … ${pct(ciRate[1])}</td><td>${fmt(x.mean_minimum_expected)}</td><td>${fmt(x.quality_mean)}</td>${scoreCells(x)}</tr>`});
all+=`</table><p class="note">CI для joint rate здесь — ориентировочный Wilson 95% CI по 128 независимым ответам. Основные сравнения ниже используют paired bootstrap CI по тем же prompt_id.</p>`;
document.querySelector('#allfour').innerHTML=all;

const methods=[['gdn_raw_r1_1111','Raw','Rank 1'],['gdn_rss_r1_1111','RSS','Rank 1'],['gdn_raw_r4_1111','Raw','Rank 4'],['gdn_rss_r4_1111','RSS','Rank 4']];
let mat=`<h2>Матрица «rank × нормализация»</h2><p class="note">Rank 1 у каждого признака — отдельное rank-1 направление; при сложении четырёх таких направлений общая сумма может иметь ранг до 4. RSS — нормализация суммы по root-sum-square, а не SVD-обрезка.</p><table><tr><th>Метод</th><th>Все 4 ≥4</th><th>Mean minimum</th><th>Качество</th><th>Что сравниваем</th></tr>`;
methods.forEach(([id,n,r])=>{const x=byId[id];if(!x){mat+=`<tr><th>${n}, ${r}</th><td colspan="4" class="missing">В Exp3 не посчитано</td></tr>`;return}mat+=`<tr><th>${n}, ${r}</th><td><b>${pct(x.all_active_ge4)}</b></td><td>${fmt(x.mean_minimum_expected)}</td><td>${fmt(x.quality_mean)}</td><td>${n==='RSS'?'направления сначала складываются, затем нормализуются':'направления складываются напрямую'}</td></tr>`});
mat+=`</table><p class="callout"><b>Честная интерпретация:</b> raw rank 4 отсутствует — это пробел Exp3. Поэтому нельзя говорить, что RSS «победил rank 4 raw» или что rank 4 всегда лучше: в Exp3 есть только RSS rank 4. Самый надёжный доступный контраст — RSS rank 4 − RSS rank 1: ${pp(0.10628953125)} [${pp(0.0300067623)}, ${pp(0.190550226)}] по mean minimum.</p>`;
document.querySelector('#matrix').innerHTML=mat;

let prof=`<h2>Какие признаки реально меняются в полном составе</h2><p class="note">Числа — expected score. Delta считается относительно baseline; это не процент ответов, а изменение средней оценки на шкале 1–5.</p><table><tr><th>Признак</th><th>Baseline</th><th>GDN raw r1</th><th>GDN RSS r1</th><th>GDN RSS r4</th><th>RSS r4 − baseline</th></tr>`;
FEATURES.forEach(f=>{const b=base.feature_expected_means[f], r1=byId.gdn_raw_r1_1111.feature_expected_means[f], rr1=byId.gdn_rss_r1_1111.feature_expected_means[f], rr4=full.feature_expected_means[f];prof+=`<tr><th>${F(f)}</th><td>${fmt(b)}</td><td>${fmt(r1)}</td><td>${fmt(rr1)}</td><td>${fmt(rr4)}</td><td class="${cls(rr4-b)}">${signed(rr4-b)}</td></tr>`});
prof+=`</table><p class="callout warn"><b>Почему all-four rate низкий:</b> joy на matched alpha слабее остальных, а concrete и candor уже высоки в baseline — у них есть ceiling effect. Низкий joint rate нельзя приписывать только плохому смешиванию.</p>`;
document.querySelector('#profiles').innerHTML=prof;

const rank1=S.conditions.filter(x=>x.condition.startsWith('gdn_raw_r1_')&&x.condition!=='gdn_raw_r1_1111').sort((a,b)=>a.active_features.length-b.active_features.length||a.condition.localeCompare(b.condition));
let comb=`<h2>Комбинации: raw GDN rank 1</h2><p class="note">Здесь виден каждый одиночный, парный и тройной состав, реально посчитанный в Exp3. Не путайте joint rate с «силой одного признака»: joint rate требует, чтобы все активные признаки были ≥4 в одном и том же ответе.</p><table><tr><th>Состав</th><th>Mean minimum</th><th>Все активные ≥4</th><th>Активные expected scores</th></tr>`;
rank1.forEach(x=>{const bits=x.condition.slice(-4), fs=maskFeatures(bits);comb+=`<tr><th>${labels(fs)}</th><td>${fmt(x.mean_minimum_expected)}</td><td><b>${pct(x.all_active_ge4)}</b></td><td>${fs.map(f=>`${F(f)}: ${fmt(x.feature_expected_means[f])}`).join(' · ')}</td></tr>`});
comb+=`</table><p class="callout">Сочетания с optimism чаще дают положительный GDN−activation contrast, особенно optimism+candor и concrete+optimism. Это гипотеза о совместимости направления, а не доказательство универсальной композициональности.</p>`;
document.querySelector('#combinations').innerHTML=comb;

const contrastIds=['gdn_rss_r1_minus_raw|1111','gdn_rank4_minus_rank1|1111','activation_rss_minus_raw|1111','gdn_minus_activation_rss|1111','gdn_minus_activation_raw|1111','gdn_rank4_minus_rank1|1100','gdn_minus_activation_raw|1100','gdn_minus_activation_raw|0110'];
const contrast=(name,mask)=>S.contrasts.find(x=>x.contrast===name&&x.mask===mask);
let ct=`<h2>Парные контрасты и доверительные интервалы</h2><p class="note">Это уже разность на тех же prompt_id, поэтому она информативнее простого сравнения процентов. CI, пересекающий ноль, не даёт уверенного directional claim.</p><table><tr><th>Контраст</th><th>Состав</th><th>Δ mean minimum</th><th>95% CI</th><th>Вывод</th></tr>`;
contrastIds.forEach(key=>{const [n,mask]=key.split('|'),x=contrast(n,mask);if(!x)return;const d=x.paired_difference;const verdict=d.ci95_low>0?'CI выше нуля':d.ci95_high<0?'CI ниже нуля':'CI пересекает ноль';ct+=`<tr><th>${n.replaceAll('_',' ')}</th><td>${labels(maskFeatures(mask))}</td><td class="${cls(d.mean)}">${signed(d.mean)}</td><td>[${signed(d.ci95_low)}, ${signed(d.ci95_high)}]</td><td>${verdict}</td></tr>`});
ct+=`</table><p class="callout"><b>Что здесь лучше всего видно:</b> GDN raw rank 1 выше activation raw на all-four на ${signed(0.0918334765625)} (CI полностью выше нуля). Для RSS rank 1 преимущество над raw не доказано. Rank 4 RSS выше rank 1 RSS на ${signed(0.10628953125)}.</p>`;
document.querySelector('#contrasts').innerHTML=ct;

const tripleMasks=['0111','1011','1101','1110'];
let triples=`<h2>Тройки: сила и сохранение качества</h2><p class="note">Эта таблица добавляет то, чего не хватало в старой сводке: для каждой тройки рядом видны joint endpoint и отдельное качество ответа. Поэтому слабый joy можно отличить от общего падения качества.</p><table><tr><th>Состав</th><th>Raw rank1<br>mean min</th><th>Raw rank1<br>все ≥4</th><th>Raw rank1<br>quality</th><th>RSS rank4<br>mean min</th><th>RSS rank4<br>все ≥4</th><th>RSS rank4<br>quality</th><th>Δ quality</th></tr>`;
tripleMasks.forEach(mask=>{const raw=byId['gdn_raw_r1_'+mask],rss=byId['gdn_rss_r4_'+mask],dq=rss.quality_mean-raw.quality_mean;triples+=`<tr><th>${labels(maskFeatures(mask))}</th><td>${fmt(raw.mean_minimum_expected)}</td><td><b>${pct(raw.all_active_ge4)}</b></td><td>${fmt(raw.quality_mean)}</td><td>${fmt(rss.mean_minimum_expected)}</td><td><b>${pct(rss.all_active_ge4)}</b></td><td>${fmt(rss.quality_mean)}</td><td class="${cls(dq)}">${signed(dq)}</td></tr>`});
triples+=`</table><p class="callout"><b>Что видно:</b> тройка concrete + optimism + candor — сильный рабочий случай: raw rank1 даёт ${pct(byId.gdn_raw_r1_1110.all_active_ge4)}, RSS rank4 — ${pct(byId.gdn_rss_r4_1110.all_active_ge4)}, при quality ${fmt(byId.gdn_rss_r4_1110.quality_mean)}/5. Тройки с joy заметно слабее (пример RSS rank4: ${pct(byId.gdn_rss_r4_0111.all_active_ge4)}, ${pct(byId.gdn_rss_r4_1011.all_active_ge4)}, ${pct(byId.gdn_rss_r4_1101.all_active_ge4)}), но их quality остаётся около ${fmt(byId.gdn_rss_r4_0111.quality_mean)}–${fmt(byId.gdn_rss_r4_1101.quality_mean)}. Это похоже на слабую выраженность joy, а не на разрушение ответа.</p>`;
document.querySelector('#triples').innerHTML=triples;

document.querySelector('#conclusions').innerHTML=`<h2>Выводы и границы вывода</h2><div class="insights">
<article class="insight"><h3>1. GDN умеет складывать признаки, но не идеально</h3><p>У full-four GDN raw rank1 joint rate ${pct(byId.gdn_raw_r1_1111.all_active_ge4)}, RSS rank4 — ${pct(full.all_active_ge4)}. Это значит, что часть ответов одновременно сохраняет четыре режима, но не все.</p></article>
<article class="insight"><h3>2. Rank 4 помогает в RSS</h3><p>Контраст RSS rank4 − rank1 равен ${pp(0.10628953125)} и его CI не пересекает ноль. Это самый ясный rank-сигнал Exp3, но raw rank4 нужно добавить в Exp4 для полной матрицы.</p></article>
<article class="insight"><h3>3. RSS на rank1 не является улучшением</h3><p>RSS rank1 − raw rank1 = ${pp(-0.03362107)}; CI пересекает ноль. Нельзя называть нормализацию универсально лучшей без rank/strength контроля.</p></article>
<article class="insight"><h3>4. Baseline и ceiling важны</h3><p>Concrete и candor уже близки к 4 в baseline, поэтому их рост ограничен. Joy заметно слабее; joint rate штрафуется одним слабым признаком.</p></article>
<article class="insight warn"><h3>5. Чего Exp3 не доказал</h3><p>Нет raw rank4, нет ортогональных rank slots «по одному рангу на признак», нет независимого holdout для всех factorial-комбинаций и нет человеческой калибровки Judge.</p></article>
<article class="insight"><h3>6. Зачем Exp4</h3><p>Exp4 добирает raw rank4, сравнивает raw/RSS на all-four и всех парах, а затем отдельно показывает singleton retention. Это закрывает главный пробел Exp3.</p></article>
</div><p class="source">Judge usage Exp3: ${Number(S.judge_usage.input_tokens).toLocaleString('ru-RU')} input tokens, ${Number(S.judge_usage.output_tokens).toLocaleString('ru-RU')} output tokens, оценка API $${Number(S.judge_usage.estimated_usd).toFixed(2)}. Источник: <span class="mono">composition-normalization-v3/summary.json</span>. Dashboard только визуализирует summary и не делает новых API/GPU вызовов.</p>`;
</script></body></html>"""


if __name__ == "__main__":
    main()
