# Experiment 4: strong-composition GDN ablation

Статус: **итоговый дизайн после аудита фактического Exp3; запуск не выполнен**.

Цель Exp4 — за оставшиеся восемь часов проверить наиболее важный незакрытый
вопрос: как складываются несколько **сильных** GDN-направлений и зависят ли
результаты от rank и RSS-нормализации.

Этот дизайн заменяет предыдущую версию. Предыдущая версия ошибочно считала,
что Dashboard 3 показывает новый `composition-normalization-v3`. На самом деле
Dashboard 3 читает результаты старого `composition-generation-queue` с
`casualness`. Фактический Exp3 на сервере использовал другой набор:

- joy ↔ sadness;
- concrete ↔ abstract language;
- optimism ↔ pessimism;
- principled candor ↔ sycophancy.

Фактический Exp3 полностью завершён. Его результаты лежат на сервере в
`outputs/composition-normalization-v3/summary.json`.

---

## 1. Что реально посчитал Exp3

Модель: Qwen3.5-9B.

Test: 128 одинаковых prompt IDs.

Judge: probabilistic Judge v3, `expected_score` 1–5.
Фичи: joy, concrete, optimism, candor.

Для всех 15 непустых масок были посчитаны:

- GDN per-feature rank 1, raw addition;
- GDN per-feature rank 1, RSS;
- GDN per-feature rank 4, RSS;
- activation steering, raw addition;
- activation steering, RSS.

Baseline также сохранён. Всего в summary 64 условия и 55 paired contrasts.

### Полная комбинация четырёх признаков

| Метод | Mean minimum expected score | Все четыре integer scores ≥4 | Quality |
|---|---:|---:|---:|
| Без steering | 2.897 | 6.25% | 4.607 |
| Activation raw | 2.925 | 10.16% | 4.483 |
| Activation RSS | 2.900 | 9.38% | 4.438 |
| GDN rank 1 raw | 3.016 | 8.59% | 4.683 |
| GDN rank 1 RSS | 2.983 | 8.59% | 4.637 |
| GDN rank 4 RSS | **3.089** | **12.50%** | **4.695** |

Главные paired contrasts по `mean minimum expected score`:

| Сравнение | Разность | 95% CI | Вывод |
|---|---:|---:|---|
| GDN rank 1 RSS − GDN rank 1 raw | −0.034 | [−0.092, +0.015] | Нет выигрыша RSS |
| GDN rank 4 RSS − GDN rank 1 RSS | **+0.106** | **[+0.030, +0.191]** | Rank 4 помогает полной композиции |
| Activation RSS − Activation raw | −0.025 | [−0.057, +0.005] | RSS не помогает activation |
| GDN rank 1 raw − Activation raw | **+0.092** | **[+0.016, +0.172]** | В matched-режиме GDN немного лучше |
| GDN rank 1 RSS − Activation RSS | +0.083 | [−0.007, +0.174] | Направление эффекта в пользу GDN, CI включает ноль |

### Что происходит с отдельными признаками в полной четвёрке

| Признак | Baseline | GDN rank 1 raw | GDN rank 4 RSS | Delta rank 4 RSS к baseline |
|---|---:|---:|---:|---:|
| Joy | 3.042 | 3.107 | 3.148 | +0.106 |
| Concrete | 3.925 | 3.952 | 4.017 | +0.092 |
| Optimism | 3.443 | 3.714 | 3.765 | **+0.322** |
| Candor | 3.922 | 3.883 | 3.873 | −0.049 |

Композиция не равна нулю: три оценки из четырёх растут, quality не падает.
Но hard joint остаётся низким, потому что joy остаётся около 3.15, а candor
изначально находится около потолка и слегка снижается.

---

## 2. Главная проблема Exp3

Exp3 выбирал alpha не для максимального эффекта GDN, а для **сопоставления
одиночной силы GDN и activation**. Это хороший режим для честного сравнения
методов, но плохой режим для демонстрации максимальной композиции.

Выбранные GDN alpha:

| Признак | Alpha в Exp3 | Лучший quality-safe alpha на dev | Trait при выбранном | Trait при сильном |
|---|---:|---:|---:|---:|
| Joy | 0.5 | 4 | 3.130 | 3.311 |
| Concrete | 0.5 | 4 | 3.970 | 3.987 |
| Optimism | 2 | 8 | 3.846 | 3.877 |
| Candor | 2 | 8 | 4.047 | 4.141 |

То есть особенно joy и concrete искусственно ослаблены ради matched comparison.
Перед заменой фичей нужно проверить уже существующие направления в их сильном
quality-safe режиме.

---

## 3. Короткие ответы на вопросы пользователя

### 1. Краткий dashboard неинформативен

**Вердикт: прав. Новый эксперимент не нужен, нужен исправленный dashboard.**

Dashboard 3 показывает не фактический Exp3, а старые результаты с casualness.
Надпись «лучший GDN» скрывает baseline, отдельные признаки и paired contrast.
Joy + optimism в старом dashboard действительно является отдельным наблюдением,
а не итоговым доказательством.

Нужно перенести в Dashboard 4 фактический server summary и показывать:

- absolute score;
- delta к baseline;
- `P(score >= 4)` до и после;
- `all active >= 4`;
- quality;
- paired CI.

### 2. Нет raw addition рядом с rank и RSS

**Вердикт: частично прав.**

В фактическом Exp3 уже есть:

- rank 1 raw;
- rank 1 RSS;
- rank 4 RSS.

Не хватает только **rank 4 raw**. Поэтому невозможно полностью разделить
факторы rank и normalization в матрице 2×2:

| | Raw | RSS |
|---|---|---|
| Rank 1 | есть | есть |
| Rank 4 | **нет** | есть |

Exp4 должен добавить rank 4 raw.

### 3. «Каждому признаку свой rank»

**Вердикт: это уже почти сделано в `per-feature rank 1`.**

Каждое направление отдельно усечено до rank 1. Сумма четырёх rank-1 матриц
может иметь общий rank до 4. Это и есть наиболее близкий к предложенной идее
вариант: каждый признак приносит свой rank-1 компонент.

Это не «заменить четыре элемента из 128×128». Rank-1 компонент — целое внешнее
произведение двух векторов в каждом GDN-слое.

Чего пока нет: принудительно сделать компоненты разных признаков взаимно
ортогональными и закрепить за ними разные singular slots. Это отдельная
гипотеза `orthogonal rank slots`, которую не нужно смешивать с Exp4.

### 4. Casualness портит статистику

**Вердикт: прав для старого dashboard, но проблема уже устранена в Exp3.**

Фактический Exp3 вообще не содержит casualness. Он заменён на joy. Calm там
тоже нет. Поэтому сейчас не нужно ещё раз менять casualness: сначала нужно
перестроить dashboard по правильным данным.

Joy всё ещё слабоват, но причина может быть в alpha=0.5. Exp4 сначала тестирует
joy при alpha=4. Только если он не работает и тогда, заменяем его.

### 5. Порог `>=4` без baseline мало что говорит

**Вердикт: полностью прав. Новый Judge-run для Exp3 не нужен.**

В server summary уже есть baseline и probability-weighted expected scores.
Проблема только в dashboard.

Основной endpoint:

```text
DeltaE(feature, method) =
    expected_score(method) - expected_score(baseline)
```

`P(score >= 4)` и `all active >= 4` остаются вторичными понятными метриками.

### 6. Непонятная «retention delta»

**Вердикт: прав. Новый эксперимент не нужен.**

Старая таблица считала разность score между полной композицией и singleton.
Например, `casualness +0.398` не означало «признак сработал в 39.8% случаев».
Это означало рост средней оценки с низкого значения до всё ещё низкого.

В новом dashboard эта таблица заменяется на:

| Feature | Baseline | Singleton | Full composition | Singleton − baseline | Full − singleton |

### 7. Сравнение GDN с activation непонятно

**Вердикт: dashboard плохой, но сам matched-эксперимент уже есть.**

На полной четвёрке:

- GDN rank 1 raw − activation raw: `+0.092`,
  95% CI `[+0.016, +0.172]`;
- GDN quality: `4.683`;
- activation quality: `4.483`.

В matched singleton-strength режиме GDN не хуже и по этой метрике немного
лучше. Новый activation-run нужен только для другого вопроса:
«кто сильнее при independently optimized alpha». Он не входит в первые восемь
часов.

### 8. Нужен ли бинарный A/B

**Вердикт: нет, пользователь прав.**

Фактический Exp3 уже использует probabilistic Judge v3 и expected score.
Бинарный A/B остаётся только историческим результатом Dashboard 1 и не
используется в Exp4.

### 9. RSS уже доказан как лучший baseline

**Вердикт: нет. Это был ошибочный вывод из старого dashboard.**

При одинаковом rank 1:

```text
RSS - raw = -0.034, 95% CI [-0.092, +0.015]
```

То есть текущие данные не показывают выигрыша RSS. Наоборот, точечная оценка
слегка хуже raw. RSS нельзя назначать основным методом до strong-alpha
stress-test.

### 10. Почему нет French

**Вердикт: French действительно отсутствует в фактическом Exp3.**

French полезен как ось другого типа, но добавление пятого признака сейчас
увеличивает число масок с 15 до 31 и почти удваивает генерацию/Judge.
За оставшиеся восемь часов он не должен вытеснять незакрытый rank×RSS
эксперимент.

French — первый дополнительный эксперимент после встречи:

- French singleton;
- French + optimism;
- French + candor;
- French + concrete;
- French + три поведенческих признака.

---

## 4. Exp4-A: что запускаем в оставшиеся восемь часов

### Исследовательский вопрос

При сильных quality-safe alpha:

1. сохраняются ли четыре признака вместе;
2. помогает ли RSS при одинаковом rank;
3. помогает ли rank 4 сам по себе;
4. является ли прирост rank 4 эффектом композиции или он уже есть на singleton.

### Фиксированный набор признаков

- joy;
- concrete;
- optimism;
- candor.

Новые donor data не нужны. Directions и SVD-кэши уже есть.

### Alpha

Из уже завершённого dev sweep:

```text
joy      = 4
concrete = 4
optimism = 8
candor   = 8
```

Перед test делается только маленький **global composition scale** sweep на
32 dev prompts:

```text
lambda in {0.5, 0.75, 1.0}
effective_alpha[f] = lambda * alpha[f]
```

Выбирается максимальный mean-minimum expected score при условии:

```text
quality >= baseline_quality - 0.25
```

После выбора `lambda` test больше не используется для tuning.

### Методы

Полный 2×2 factorial:

| ID | Rank | Composition |
|---|---:|---|
| `gdn_raw_r1` | 1 per feature | raw sum |
| `gdn_rss_r1` | 1 per feature | RSS |
| `gdn_raw_r4` | 4 per feature | raw sum |
| `gdn_rss_r4` | 4 per feature | RSS |

### Маски

Обязательные:

- 4 singleton-маски;
- все 6 пар;
- полная четвёрка.

Тройки откладываются: они уже есть в matched Exp3 и не нужны для основного
rank×RSS вывода до встречи.

Для singleton RSS совпадает с raw, поэтому повторные RSS-генерации не нужны.

### Число условий

```text
Singletons: 4 features × 2 ranks = 8 conditions
Pairs:      6 pairs × 4 methods  = 24 conditions
All four:   1 mask × 4 methods   = 4 conditions
Total:                              36 conditions
```

При 128 prompts это 4,608 ответов. Baseline переиспользуется из Exp3.

### Порядок автономной очереди

1. Dev global-scale sweep.
2. Полная четвёрка — все четыре метода.
3. Singleton rank 1/rank 4.
4. Все пары — raw/RSS × rank 1/rank 4.
5. Judge v3 по мере появления завершённых блоков.
6. Summary и Dashboard 4.

Так даже при остановке через восемь часов первыми будут готовы самые важные
all-four и singleton результаты.

### Оценка времени

Фактический прошлый run:

- 37 GDN conditions: примерно 6 ч 53 мин;
- Judge и summarize: примерно 1 ч 16 мин;
- activation отдельно занял ещё 4 ч 25 мин.

Exp4-A содержит 36 GDN conditions и использует готовые directions/SVD-кэши.
Ожидание:

- GPU generation: 6–7 часов;
- Judge при concurrency 8: 45–75 минут;
- summary/dashboard: 10–20 минут.

Итого: примерно **7–8.5 часа**. Чтобы уложиться, Judge нужно запускать
параллельно с уже завершёнными generation-блоками, а не ждать конца всей GPU
очереди.

На момент аудита GPU 3 свободна.

### Оценка OpenRouter

Прошлый Exp3 потратил около `$3.50` на значительно большее число оценок.
Здесь Judge оценивает только активные признаки и переиспользует baseline.
Ожидаемый порядок расходов: **$0.8–1.4**.

---

## 5. Метрики Exp4-A

Для каждого признака:

- baseline expected score;
- method expected score;
- paired `DeltaE`;
- baseline и method `P(integer score >= 4)`;
- paired `DeltaP4`.

Для каждой композиции:

- mean minimum expected score;
- `all active integer scores >= 4`;
- paired delta к baseline;
- paired delta raw↔RSS;
- paired delta rank1↔rank4;
- quality;
- inactive-feature leakage.

95% CI: paired bootstrap по prompt ID. Judge не получает method, rank, RSS,
alpha или имя intervention.

Primary contrasts:

```text
rank1 RSS - rank1 raw
rank4 RSS - rank4 raw
rank4 raw - rank1 raw
rank4 RSS - rank1 RSS
```

---

## 6. Что откладываем после встречи

### Exp4-B: independently optimized GDN vs activation

Те же четыре фичи, одинаковые prompt IDs, но каждый метод выбирает собственный
лучший quality-safe alpha. Это отвечает на вопрос о максимальной эффективности,
в то время как завершённый Exp3 отвечает на вопрос о matched-strength
композиции.

### Exp4-C: French as an orthogonal language axis

Сначала French singleton с новым Judge, затем French в парах и одной
четырёхпризнаковой комбинации. Не нужно сразу запускать полный factorial из
пяти признаков.

### Exp4-D: explicit orthogonal rank slots

Текущий per-feature rank 1 уже даёт каждому признаку отдельную rank-1 матрицу,
но не гарантирует ортогональность. Отдельная гипотеза:

1. извлечь rank-1 компонент каждого признака;
2. ортогонализовать компоненты между признаками;
3. сравнить обычную сумму, RSS и orthogonalized sum;
4. только после этого обсуждать freezing/clamp.

### Persistence / freezing

Заморозка singular components и обучение модели обходиться без них — отдельный
проект. Его нельзя интерпретировать, пока не установлено, что выбранные
компоненты сохраняют несколько признаков одновременно.

---

## 7. Что должно быть готово к встрече

Минимальный достаточный результат:

1. Исправленный Dashboard 3/4 на фактическом Exp3 summary.
2. Таблица baseline → GDN/activation с expected deltas.
3. Доказанный текущий вывод: rank 4 RSS лучше rank 1 RSS в полной композиции.
4. Доказанный текущий вывод: RSS не лучше raw при rank 1.
5. Доказанный текущий вывод: GDN raw немного лучше activation raw в matched
   full-four режиме.
6. Exp4-A strong-alpha all-four и singleton результаты.
7. Если хватает времени — все шесть пар.
8. Готовые, но не обязательно запущенные планы French, optimized activation и
   orthogonal rank slots.

Если пары не успевают, их очередь должна безопасно продолжаться после встречи.
Нельзя ради количества условий откладывать all-four, singleton и точный
rank×RSS factorial.
