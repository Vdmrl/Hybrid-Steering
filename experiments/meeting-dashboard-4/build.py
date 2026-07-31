"""Build a Russian, self-contained design dashboard for Experiment 4."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/meeting-dashboard-4/index.html"


def build() -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return HTML.replace("__GENERATED__", generated)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build(), encoding="utf-8")
    print(args.output)


HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hybrid Steering — Experiment 4</title>
<style>
:root{--bg:#09111c;--panel:#111d2c;--panel2:#0d1724;--line:#2a3b52;--text:#edf4ff;--muted:#9fb0c4;--cyan:#64ddd4;--blue:#8794ff;--green:#5ee09b;--amber:#f2bd63;--red:#ff8585}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 Inter,system-ui,sans-serif}header,main{max-width:1320px;margin:auto;padding:24px}header{padding-top:36px}h1{font-size:36px;line-height:1.08;margin:6px 0 12px}h2{font-size:20px;margin:0 0 8px}h3{font-size:15px;margin:0 0 7px}.eyebrow{color:var(--cyan);font-weight:800;letter-spacing:.08em;text-transform:uppercase}.muted,.note{color:var(--muted)}.note{font-size:13px}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}.panel{grid-column:span 12;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px;overflow:auto}.half{grid-column:span 6}.third{grid-column:span 4}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:16px 0}.card{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:13px}.card strong{display:block;font-size:25px;margin:4px 0}.good{color:var(--green)}.warn{color:var(--amber)}.bad{color:var(--red)}table{width:100%;border-collapse:collapse;margin-top:14px}th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{color:#c9d6e7;font-size:12px}tr:last-child td,tr:last-child th{border-bottom:0}.callout{border-left:3px solid var(--cyan);padding-left:12px;margin:14px 0}.callout.warn{border-color:var(--amber)}.callout.bad{border-color:var(--red)}.formula{font:13px ui-monospace,SFMono-Regular,Consolas,monospace;background:#08101a;border:1px solid var(--line);padding:12px;border-radius:8px;white-space:pre-wrap}.timeline{display:grid;grid-template-columns:repeat(7,1fr);gap:8px;margin-top:14px}.step{background:var(--panel2);border-top:3px solid var(--cyan);padding:11px;border-radius:8px}.step small{display:block;color:var(--muted);margin-top:5px}.insights{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.insight{background:var(--panel2);border-radius:10px;padding:13px;border-top:3px solid var(--cyan)}.insight.warn{border-top-color:var(--amber)}.source{font-size:12px;color:var(--muted);margin-top:16px}.mono{font:12px ui-monospace,SFMono-Regular,Consolas,monospace}.tag{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:2px 8px;color:var(--muted);font-size:11px;margin:2px}
@media(max-width:950px){.half,.third{grid-column:span 12}.cards,.insights{grid-template-columns:1fr 1fr}.timeline{grid-template-columns:repeat(3,1fr)}}@media(max-width:560px){header,main{padding:16px}.cards,.insights,.timeline{grid-template-columns:1fr}h1{font-size:29px}}
</style></head>
<body><header>
<div class="eyebrow">Hybrid Steering · Experiment 4 · Qwen3.5-9B</div>
<h1>Strong composition: rank × нормализация</h1>
<p class="muted">Это не отчёт с результатами, а русскоязычная карта запущенного эксперимента: что именно сравниваем, зачем нужны блоки и как по ним принимать решение. Очередь уже запущена автономно; этот экран можно перечитать до появления итогового Judge. Построено __GENERATED__.</p>
<p class="callout"><b>Одной фразой:</b> мы проверяем, помогает ли увеличенный ранг и RSS-нормализация сохранить сразу несколько поведенческих признаков, а не только один самый сильный.</p>
</header><main class="grid">
<section class="panel" id="why"></section>
<section class="panel" id="features"></section>
<section class="panel" id="methods"></section>
<section class="panel" id="matrix"></section>
<section class="panel" id="metrics"></section>
<section class="panel" id="queue"></section>
<section class="panel" id="interpret"></section>
<section class="panel" id="limits"></section>
</main>
<script>
const featureNames={joy:'Joy — радость',concrete:'Concrete language — конкретный язык',optimism:'Optimism — оптимизм',candor:'Principled candor — принципиальная прямота'};
document.querySelector('#why').innerHTML=`<h2>Зачем нужен Exp4 после Exp3</h2>
<p>В реальном Exp3 мы уже увидели несколько сигналов, но не закрыли всю матрицу:</p>
<table><tr><th>Что известно из Exp3</th><th>Чего не хватало</th><th>Что делает Exp4</th></tr>
<tr><td>GDN raw rank 1 на all-four дал преимущество над activation raw.</td><td>Не было raw rank 4.</td><td>Добавляет raw rank 4 и сравнивает его с RSS rank 4.</td></tr>
<tr><td>RSS rank 4 выглядел лучше RSS rank 1.</td><td>Неясно, это ранг или сама нормализация.</td><td>Ставит raw и RSS рядом при rank 1 и rank 4.</td></tr>
<tr><td>Одиночные признаки и некоторые пары работают.</td><td>Неясно, ломается ли full composition из-за интерференции.</td><td>Считает все 6 пар и all-four одинаковыми методами.</td></tr>
</table><p class="callout warn"><b>Это не проверка SVD-обрезки как таковой:</b> rank 1 и rank 4 здесь — количество оставленных компонент направления каждого признака. Отдельный вопрос про orthogonal rank slots («свой ранг каждому признаку») оставляем на следующий эксперимент.</p>`;

document.querySelector('#features').innerHTML=`<h2>Какие признаки и как читается маска</h2><p class="note">Маска имеет порядок <b>joy · concrete · optimism · candor</b>: 1111 — все четыре, 0011 — joy + concrete, 0100 — только optimism.</p><div class="cards">${Object.entries(featureNames).map(([k,v])=>`<div class="card"><b>${v.split(' — ')[0]}</b><strong>${v.split(' — ')[1]}</strong><small>Для каждого признака есть отдельное направление GDN.</small></div>`).join('')}</div><p class="note">Используем сильные, но quality-safe alpha, выбранные заранее на dev: joy=4, concrete=4, optimism=8, candor=8. Общий множитель λ выбирается только на dev из {0.5, 0.75, 1.0}; тестовые ответы не участвуют в выборе.</p>`;

document.querySelector('#methods').innerHTML=`<h2>Четыре метода в главном сравнении</h2><table><tr><th>Метод</th><th>Что делаем математически</th><th>Какой вопрос отвечает</th></tr>
<tr><th>GDN raw, rank 1</th><td>Для каждого активного признака берём его rank-1 матрицу и складываем направления напрямую.</td><td>Может ли компактное rank-1 направление каждого признака работать вместе?</td></tr>
<tr><th>GDN RSS, rank 1</th><td>Складываем те же rank-1 направления, затем нормализуем сумму по root-sum-square.</td><td>Помогает ли контроль нормы не дать комбинации разогнаться?</td></tr>
<tr><th>GDN raw, rank 4</th><td>Для каждого признака берём rank-4 направление и складываем напрямую.</td><td>Достаточно ли rank 1 или нужна дополнительная ёмкость?</td></tr>
<tr><th>GDN RSS, rank 4</th><td>Складываем rank-4 направления и затем применяем RSS-нормализацию.</td><td>Есть ли совместный выигрыш от ёмкости и контроля нормы?</td></tr></table>
<div class="formula">raw:       Δ = Σᵢ αᵢ · Dᵢ
RSS:       Δ = (Σᵢ αᵢ · Dᵢ) · (target_norm / (||Σᵢ αᵢ · Dᵢ||₂ + ε))

Важно: «rank 1 для каждого признака» не означает, что вся сумма имеет rank 1.
Сумма четырёх разных rank-1 матриц может иметь ранг до 4.</div>`;

document.querySelector('#matrix').innerHTML=`<h2>Что именно будет посчитано</h2><table><tr><th>Блок</th><th>Составы</th><th>Методы</th><th>Ответов</th><th>Зачем</th></tr>
<tr><th>All-four</th><td>1111</td><td>4</td><td>4 × 128 = 512</td><td>Главный endpoint: все четыре режима одновременно.</td></tr>
<tr><th>Singletons</th><td>1000, 0100, 0010, 0001</td><td>raw rank1 + raw rank4</td><td>8 × 128 = 1024</td><td>Проверка, что rank4 не портит одиночный признак и сколько retention теряется в полном составе.</td></tr>
<tr><th>All pairs</th><td>0011, 0101, 0110, 1001, 1010, 1100</td><td>4</td><td>24 × 128 = 3072</td><td>Самая информативная часть для взаимодействий: сравниваем пары одинаковыми методами.</td></tr>
<tr><th>Итого main</th><td>36 условий</td><td>4 режима в тех блоках, где это нужно</td><td><b>4608</b></td><td>Каждая строка использует те же 128 test prompts.</td></tr></table>
<p class="callout">Singleton-блок намеренно не дублирует RSS: RSS для одной матрицы почти не отличается по смыслу от исходного направления. Зато raw rank1/rank4 даёт понятную retention-проверку.</p>`;

document.querySelector('#metrics').innerHTML=`<h2>Какие числа будут в итоговом отчёте</h2><div class="cards">
<div class="card"><b>1. Feature score</b><strong>1–5</strong><small>Средний expected score каждого активного признака.</small></div>
<div class="card"><b>2. Mean minimum</b><strong>1–5</strong><small>Средняя сила самого слабого активного признака в ответе.</small></div>
<div class="card"><b>3. Joint ≥4</b><strong>0–100%</strong><small>Доля ответов, где все активные признаки ≥4.</small></div>
<div class="card"><b>4. Quality</b><strong>1–5</strong><small>Отдельная оценка качества ответа; не смешиваем её с trait score.</small></div></div>
<p>Для каждого сравнения считаем paired difference по одинаковым prompt_id и bootstrap 95% CI:</p><table><tr><th>Контраст</th><th>Интерпретация</th></tr>
<tr><th>rank4 − rank1</th><td>Нужна ли дополнительная ёмкость направления?</td></tr>
<tr><th>RSS − raw</th><td>Помогает ли нормализация именно при той же rank?</td></tr>
<tr><th>full − singleton</th><td>Сохраняется ли признак, когда рядом стоят другие направления?</td></tr>
<tr><th>all-four − expected from pairs</th><td>Есть ли дополнительная интерференция именно при четырёх признаках?</td></tr></table>
<p class="note">Judge: финальный компактный v3 без бинарного A/B, с независимым score каждого из четырёх признаков и отдельным quality endpoint. Результаты возобновляются по стабильным answer_id.</p>`;

document.querySelector('#queue').innerHTML=`<h2>Порядок автономной очереди</h2><p class="note">Очередь сначала считает короткие проверки, затем наиболее важные блоки. Если процесс остановится, уже готовые блоки не теряются.</p><div class="timeline">
<div class="step"><b>1. Self-test</b><small>Проверка условий и формул.</small></div><div class="step"><b>2. GPU smoke</b><small>Один prompt, два режима.</small></div><div class="step"><b>3. Dev</b><small>3 λ × all-four на 32 prompts.</small></div><div class="step"><b>4. All-four</b><small>4 метода × 128.</small></div><div class="step"><b>5. Singletons</b><small>rank1/rank4 × 4 признака.</small></div><div class="step"><b>6. Pairs</b><small>6 пар × 4 метода.</small></div><div class="step"><b>7. Judge + summary</b><small>CI, contrasts, report.</small></div></div>
<p class="callout"><b>Сейчас:</b> очередь Exp4 уже запущена на GPU 3. Dashboard — статический дизайн, он не делает новых вызовов и не заменяет серверный queue.log.</p>`;

document.querySelector('#interpret').innerHTML=`<h2>Как поймём результат</h2><div class="insights">
<article class="insight"><h3>Если rank4 лучше rank1</h3><p>У rank1 не хватало ёмкости для совместной композиции. Это поддерживает гипотезу «каждый признак требует нескольких компонент».</p></article>
<article class="insight"><h3>Если RSS лучше raw</h3><p>Проблема была в росте нормы при сложении, а не обязательно в конфликте смыслов. Это аргумент в пользу нормализованного GDN.</p></article>
<article class="insight"><h3>Если full-four хуже пар</h3><p>Появляется дополнительная интерференция при четырёх направлениях. Пары остаются рабочими, но простая линейность не масштабируется.</p></article>
<article class="insight"><h3>Если методы почти равны</h3><p>На текущей силе и 128 prompts различий недостаточно. Это не «методы одинаковы вообще», а повод расширить holdout или диапазон λ.</p></article>
<article class="insight warn"><h3>Если падает quality</h3><p>Trait score нельзя считать успехом, если ответы становятся хуже. Отсечка для dev — quality не ниже baseline −0.25.</p></article>
<article class="insight"><h3>Главный итог</h3><p>Нас интересует не одна красивая цифра, а согласованный паттерн: feature scores растут, joint rate растёт, quality сохраняется, а CI контрастов не пересекает ноль.</p></article>
</div>`;

document.querySelector('#limits').innerHTML=`<h2>Что Exp4 пока не проверяет</h2><table><tr><th>Вопрос</th><th>Статус</th><th>Почему это отдельный шаг</th></tr>
<tr><th>Свой rank slot каждому признаку в одной общей матрице</th><td class="warn">Не входит</td><td>Это уже не просто сумма четырёх SVD-направлений; нужно определить ортогональные/закреплённые компоненты и отдельный протокол.</td></tr>
<tr><th>Заморозка или clamp отдельных компонент</th><td class="warn">Не входит</td><td>Сначала измеряем базовую композицию; иначе не отделим эффект clamp от эффекта rank/normalization.</td></tr>
<tr><th>Классический activation steering</th><td class="warn">Не входит</td><td>Его лучше сравнить после полной GDN матрицы на тех же парах и с тем же Judge.</td></tr>
<tr><th>French и новые признаки</th><td class="warn">Не входит</td><td>Exp4 сосредоточен на четырёх уже подготовленных направлениях и качестве композиционного абляционного теста.</td></tr>
<tr><th>Тройки признаков</th><td class="warn">Не входит</td><td>В Exp4 приоритет — все 6 пар и all-four. Тройки уже были в Exp3; следующий расширенный run может повторить их на выбранных методах.</td></tr>
<tr><th>256 prompts вместо 128</th><td class="warn">Не входит</td><td>Текущая очередь фиксирована на 128 для сопоставимости с Exp3. Увеличение до 256 разумно после закрытия raw-rank4/RSS и выбора компактного набора условий.</td></tr>
<tr><th>Full-rank direction без SVD-обрезки</th><td class="warn">Не входит</td><td>RSS можно применять и к full-rank сумме; это отдельная контрольная клетка, которую стоит добавить после текущей rank-абляции.</td></tr>
</table><p class="callout">Поэтому Exp4 — не финальное доказательство «GDN лучше activation steering». Это аккуратный следующий блок: он сначала закрывает пропущенную rank×normalization матрицу и показывает, где именно ломается композиция.</p><p class="source">Источник дизайна: <span class="mono">experiments/strong-composition-exp4/run.yaml</span> и <span class="mono">run.py</span>. Большие генерации и Judge artifacts остаются на сервере, в Git попадает только код, manifest и этот объясняющий dashboard.</p>`;
</script></body></html>"""


if __name__ == "__main__":
    main()
