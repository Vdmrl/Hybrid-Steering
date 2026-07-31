# Final experiment: five concept-aligned GDN slots

Статус: **готовый дизайн следующего 18–20-часового запуска**.

## Исследовательский вопрос

Можно ли одновременно и устойчиво удерживать несколько различимых режимов в
recurrent state GDN, если каждому концепту оставить собственную rank-1
компоненту и после каждого recurrent update восстанавливать её коэффициент?

Это inference-time эксперимент. Мы не меняем веса модели и не дообучаем её.
Слово «заморозка» ниже означает только clamp выбранных компонент состояния во
время генерации.

## Почему этот эксперимент нужен после Exp4

Exp4 показал:

- raw rank 1 даёт лучший mean-minimum score полной комбинации;
- RSS помогает именно перегруженному rank 4, но не rank 1;
- candor сильно теряется в полной композиции;
- joy и optimism недостаточно независимы.

Следовательно, ещё один полный factorial на тех же четырёх признаках не нужен.
Нужно одновременно исправить набор концептов и проверить механизм, который
предназначен не только для начальной инъекции, но и для сохранения каждого
концепта во времени.

## Пять основных осей

1. **French language** — язык ответа.
2. **Concrete language** — конкретность и операциональность.
3. **Principled candor** — позиция под социальным давлением.
4. **Optimism** — оценка будущих возможностей.
5. **Casualness** — регистр речи.

Это не гарантированно «идеальные» признаки, а заранее выбранные кандидаты
разных типов. Joy исключён, потому что в Exp3/Exp4 он смешивался с optimism.
Calm исключён, потому что прошлый singleton steering не дал устойчивого
эффекта. Casualness возвращён: старый эксперимент показывал заметную дельту от
низкого baseline, а концептуально register отделим от языка, содержания,
позиции и прогноза.

Если casualness не проходит singleton gate, мы не подменяем его после просмотра
test. Основной результат честно становится четырёхпризнаковым; пятый признак
остаётся отрицательным singleton-результатом.

## Данные

- Модель: Qwen3.5-9B.
- Dev: 32 prompt IDs только для alpha и scale.
- Test: 128 фиксированных prompt IDs.
- `max_new_tokens`: 256.
- Judge: стандартный probabilistic Judge v3.
- Quality оценивается отдельно один раз на ответ.
- French дополнительно проверяется детерминированным language detector; Judge
  остаётся общей метрикой для единого формата.

Сохраняем старые test prompt IDs там, где это возможно, чтобы сравниваться с
Exp3/Exp4. До запуска проверяем, что промпты не требуют явно отвечать на
английском и допускают оценку candor, concrete language и optimism. Нельзя
подбирать test prompts по уже увиденным ответам.

## Этап 0: обязательный singleton gate

На 32 dev prompts для каждого признака выбирается один quality-safe alpha.
Фиксированные критерии:

1. для признака с baseline ниже 3.75:
   `delta expected_score >= +0.20` относительно baseline;
2. для ceiling-признака с baseline не ниже 3.75 допускается меньшая positive
   delta, но положительный и отрицательный знаки направления должны давать
   разность не меньше 0.30 балла;
3. 95% CI на test не используется для выбора;
4. `quality delta >= -0.25`;
5. целевая signed difference должна быть больше максимальной абсолютной
   off-target дельты;
6. для French дополнительно main-language rate не ниже 90% на dev.

Если признак не проходит gate, он не считается активным при расчёте primary
joint metric. Это предотвращает повторение Exp4, где слабый или спутанный
признак автоматически рушил всю композицию.

До генерации также сохраняем для направлений:

- pairwise Frobenius cosine;
- Gram matrix;
- effective rank суммы;
- condition number.

Это геометрическая диагностика, а не замена поведенческой оценки Judge.

## Concept-aligned rank-1 slots

Для каждого концепта и каждого GDN-слоя сохраняем собственную rank-1 матрицу:

\[
B_i = u_i v_i^\top.
\]

Имена слотов задаются исходным концептом. Мы не делаем SVD уже сложенной
матрицы, потому что её singular vectors могут смешать несколько концептов.

Обычная add-once интервенция:

\[
S_0' = S_0 + \sum_i c_i B_i.
\]

После этого модель свободно обновляет state, поэтому коэффициенты концептов
могут затухать или смешиваться.

Slot clamp после каждого recurrent update:

\[
S_t = S_t^{raw} + \sum_i (c_i^* - \hat c_i) B_i.
\]

Для неортогональных слотов коэффициенты оцениваются совместно:

\[
G\hat c=b,\qquad
G_{ij}=\langle B_i,B_j\rangle_F,\qquad
b_i=\langle B_i,S_t^{raw}\rangle_F.
\]

Используется устойчивое решение Gram system с маленькой ridge-регуляризацией.
Независимые скалярные проекции здесь некорректны: один слот может содержать
часть другого.

## Что означает RSS

Для активных направлений `D_i = alpha_i B_i` обычная сумма равна
`D_raw = sum D_i`. RSS сохраняет направление этой суммы, но приводит её норму
к норме, ожидаемой для ортогональных компонент:

\[
D_{RSS}=D_{raw}
\frac{\sqrt{\sum_i\|D_i\|_F^2}}
{\|D_{raw}\|_F+\epsilon}.
\]

RSS не создаёт отдельные слоты и не делает направления ортогональными. Поэтому
в финальном эксперименте `RSS add-once` и `RSS slot clamp` — разные условия.

## Primary methods

| ID | Интервенция | Зачем |
|---|---|---|
| `baseline` | без steering | исходное поведение |
| `activation_raw` | обычный activation steering | стандартный внешний baseline |
| `gdn_add_r1_raw` | add once, отдельный rank 1 на концепт | простой GDN baseline |
| `gdn_add_r1_rss` | add once, rank 1 + RSS | matched baseline для RSS clamp |
| `gdn_add_r4_rss` | add once, rank 4 + RSS | лучший joint-rate вариант Exp4 |
| `gdn_clamp_r1_raw` | clamp пяти rank-1 слотов | проверка persistence |
| `gdn_clamp_r1_rss` | clamp пяти rank-1 слотов с RSS target | нормализованный clamp |

`orthogonal clamp` не является primary: ортогонализация может изменить
семантику. Она запускается последней только при оставшемся времени.

## Минимальная матрица условий на 18–20 часов

### A. Singleton screen — обязательно

По пять singleton-условий для `gdn_add_r1_raw` и `gdn_clamp_r1_raw` на test.
Они подтверждают, что каждое направление работает отдельно, дают off-target
cross-talk matrix и показывают, помогает ли persistence до сложной композиции.

### B. Full five — обязательно и запускается первым после gate

Одна полная комбинация для каждого из шести методов без baseline:

- activation raw;
- GDN add rank 1 raw;
- GDN add rank 1 RSS;
- GDN add rank 4 RSS;
- GDN clamp rank 1 raw;
- GDN clamp rank 1 RSS.

### C. Leave-one-out — обязательно

Для двух matched GDN-методов:

- `gdn_add_r1_rss`;
- `gdn_clamp_r1_rss`.

По пять условий: full set без одного признака. Для clamp это одновременно
проверка независимого управления слотами.

### D. Flip-one slot — обязательно

Для `gdn_clamp_r1_rss` создаются пять условий: остальные четыре слота остаются
в положительном полюсе, а у одного целевой коэффициент меняет знак. Это главный
тест независимого управления и снимает проблему baseline ceiling у candor и
concrete language.

Успешный flip должен менять преимущественно score своего признака. Он не должен
одновременно рушить French, optimism, register и quality.

### E. Все десять пар — желательно

Все пары прогоняются для тех же двух главных GDN-методов. Это показывает, какие
именно пары интерферируют, и не требует полного набора 31 маски.

### F. Clamp persistence — обязательно, почти бесплатно

Во время generation сохраняем коэффициенты каждого слота после 0%, 25%, 50%,
75% и 100% сгенерированных токенов. Это не требует вызовов Judge и напрямую
показывает затухание add-once и сохранение clamp.

### G. Orthogonal clamp — только если основная очередь завершилась

Одна full-five комбинация и пять leave-one-out условий. Если Gram matrix хорошо
обусловлена и clamp уже работает, этот блок пропускается как ненужный.

Итого до optional-блока:

```text
singleton add/clamp                          10
full-five methods                             6
leave-one-out: 5 masks × 2 methods           10
flip-one clamp                                5
                                             --
mandatory generation conditions              31

all pairs: 10 masks × 2 methods              20
                                             --
full generation queue                        51
```

Baseline генерируется один раз или переиспользуется только при полном совпадении
prompt IDs, generation config и model revision.

## Порядок автономной очереди

1. Self-test на синтетической матрице: Gram projection и clamp возвращают
   заданные коэффициенты.
2. Smoke: один prompt, два слота, 16 tokens; убедиться, что KV и conversation
   state не подмешиваются между ответами.
3. Dev singleton alpha selection.
   На dev также выбирается один `beta` из `{0.2, 0.5, 1.0}` для soft clamp:
   `S <- S_raw + beta * correction`. Выбор делается по mean-minimum при
   ограничении quality delta >= −0.25; test для выбора beta не используется.
4. Test singleton screen.
5. Full-five методы.
6. Leave-one-out.
7. Flip-one slot.
8. Все пары.
9. Optional orthogonal clamp.
10. Judge запускается блоками после появления завершённых generations;
   генерация не ждёт окончания Judge.
11. Summary, paired bootstrap CI и dashboard.

Каждый блок имеет `.DONE`, stable IDs и до трёх retry. Повторный запуск обязан
продолжать очередь, а не пересчитывать готовые ответы.

## Primary endpoints

1. Для каждого признака: paired `delta expected_score` к baseline.
2. Для композиции: `mean minimum expected_score`.
3. Joint rate: все прошедшие singleton gate признаки имеют integer score >= 4.
4. Answer quality и paired quality delta.
5. Slot persistence: отклонение `c_i(t)` от target по позиции токена.

Secondary:

- `P(score >= 4)` для каждого признака;
- observed joint rate против произведения marginal rates как описательная
  проверка совместного проявления;
- off-target cross-talk;
- leave-one-out selectivity;
- language detector agreement для French.

## Главные paired contrasts

```text
gdn_clamp_r1_raw - gdn_add_r1_raw
gdn_clamp_r1_rss - gdn_add_r1_rss
gdn_clamp_r1_rss - gdn_clamp_r1_raw
gdn_add_r1_raw - activation_raw
```

Для каждого показываем score delta, joint-rate delta, quality delta и 95% CI.
Нельзя выбирать лучший метод на test и затем сообщать его CI как
подтверждающий.

## Критерии успеха

Clamp считается полезным, если одновременно:

1. улучшает mean-minimum относительно соответствующего add-once baseline и CI
   не включает ноль;
2. не снижает quality более чем на 0.25 балла;
3. уменьшает drift slot coefficients по длине генерации;
4. leave-one-out меняет преимущественно целевой признак, а не все признаки;
5. flip-one меняет преимущественно выбранный признак;
6. эффект воспроизводится хотя бы для четырёх прошедших gate осей.

Если растут только French и optimism, а остальные не меняются, это не считается
доказательством пятислотовой композиции.

## Оценка времени и стоимости

Exp4 содержал 36 generation conditions. Здесь 31 обязательное и ещё 20
pairwise условий, а clamp добавляет
операцию над recurrent state на каждом токене. Реалистичный бюджет:

- подготовка направлений, self-test и dev: 1–2 часа;
- обязательные singleton/full/leave-one-out/flip блоки: 12–17 часов;
- все пары доводят GPU generation примерно до 20–28 часов;
- Judge, запущенный параллельно блоками: 1–2 часа критического пути;
- summary/dashboard: 20–40 минут.

Итого полная очередь: **22–30 часов**. Первичный результат должен быть готов за
**14–20 часов**, потому что все пары стоят последними и могут продолжаться
после встречи.

Оценка Judge при 128 prompts, compact scalar output и оценке только активных
признаков: примерно **$2.5–4.5**. Это оценка порядка величины; runner обязан
сохранять фактические token usage и стоимость.

## Что остаётся на следующий день

Если primary run успешен:

1. повторить выбранные baseline/add/clamp условия на 256 новых test prompts;
2. проверить orthogonal clamp, если geometry показывает сильное пересечение;
3. отдельно измерить decay по более длинной генерации;
4. только после этого обсуждать SFT с замороженными concept slots.

Если primary run неуспешен, не расширять sample size автоматически. Сначала
определить, не провалился ли singleton gate, не плохо ли обусловлена Gram matrix
и не разрушает ли clamp answer quality.
