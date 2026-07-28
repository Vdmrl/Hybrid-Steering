# Как вносить изменения

## Ветки

Рекомендуемый формат:

```text
exp/<концепт>-<абляция>
feat/<короткое-имя>
fix/<короткое-имя>
docs/<короткое-имя>
refactor/<короткое-имя>
test/<короткое-имя>
chore/<короткое-имя>
```

Примеры:

```text
exp/optimism-layer-ablation
exp/optimism-svd-rank
feat/pairwise-judge
fix/resume-duplicate-rows
docs/optimism-rubric
```

`main` содержит только согласованную воспроизводимую базу. Эксперименты,
абляции, изменения judge-промптов и модификации кода не коммитятся напрямую в
`main`.

Правила работы с ветками:

- перед любым изменением агент обязан выполнить `git branch --show-current`;
- если активна `main` или `master`, сначала создаётся отдельная ветка;
- каждый smoke test, новая комбинация концептов, alpha или выбор слоёв считается
  отдельным экспериментом и получает собственную `exp/...` ветку;
- одна абляция или одно логическое изменение кода — одна ветка;
- не продолжать и не переписывать ветку другого участника без согласования;
- не использовать force-push в общую ветку;
- если эксперименту нужна новая общая возможность Judge, сначала сделать её
  отдельной веткой `feat/...`, а затем использовать в `exp/...`;
- перед merge в experiment-ветке должны лежать точный config/manifest и
  компактная сводка, но не большие генерации и веса;
- после проверки изменения попадают в `main` через pull request.

Пример разделения:

```text
feat/pairwise-judge           # переиспользуемая возможность
exp/optimism-layer-ablation   # конкретный эксперимент
fix/pairwise-order-bias       # исправление найденной ошибки
```

## Коммиты

Используем Conventional Commits:

```text
<type>(<scope>): <краткое описание>
```

Допустимые `type`:

- `feat` — новая возможность;
- `fix` — исправление ошибки;
- `docs` — только документация;
- `test` — тесты и fixtures;
- `refactor` — изменение структуры без нового поведения;
- `perf` — ускорение или снижение стоимости;
- `chore` — зависимости и служебные изменения;
- `ci` — автоматические проверки.

Основные `scope`:

- `judge` — общий pipeline;
- `steering` — извлечение и изменение recurrent state;
- `artifact` — форматы direction и run manifests;
- `experiment` — воспроизводимый config конкретного эксперимента;
- `rubric` — шкалы и определения;
- `prompt` — judge-промпты;
- `schema` — форматы входа и результата;
- `provider` — OpenRouter и другие API;
- `runner` — очередь, retries и resume;
- `docs`, `ci`.

Примеры:

```text
feat(judge): add blind pairwise evaluation
feat(rubric): add optimism versus pessimism scale
fix(runner): resume partially judged prompts
docs(prompt): explain anchored score meanings
test(schema): cover malformed answer ids
```

Заголовок пишется в повелительной форме, без точки в конце. Не используем
`update`, `changes`, `work` и `wip` как описание готового коммита.

Если изменение несовместимо со старым форматом:

```text
feat(schema)!: split scalar and pairwise results
```

Причину и важные решения следует добавить в тело коммита:

```text
feat(prompt): add answer-order reversal check

Run every comparison in both orientations to detect position bias.
Store both raw decisions before aggregation.
```

## Что должно быть в pull request

- краткая цель;
- ссылка или название experiment-ветки;
- какие файлы являются источниками истины;
- как проверялось изменение;
- меняется ли стоимость judge;
- меняются ли prompt/rubric/schema versions;
- небольшой пример входа и результата для нового поведения.

Не добавляем в pull request:

- `.env` и токены;
- полные модельные генерации;
- большие датасеты и веса;
- случайные локальные отчёты;
- одновременно несвязанные рефакторинги и новую метрику.

## Изменение промптов и шкал

Промпт, использованный в опубликованном или командном результате, не
перезаписывается. Создаётся новая версия (`scalar_v1.txt`, `scalar_v2.txt`) и
в итогах явно сохраняется её имя.

Изменение определения концепта также требует повышения `rubric_version`.
Старые результаты после этого нельзя молча объединять с новыми.
