# Hybrid Judge

Общий blind LLM-as-a-Judge для сравнения steering-абляций. Он не знает про
GDN, residual stream, SVD, слои или `alpha`: на вход подаются только scenario
и несколько ответов.

## Возможности

- scalar-оценка target/opposite и трёх quality-метрик;
- pairwise `A / B / tie`, при необходимости в обоих порядках;
- определения концептов в общем `concepts/features.yaml`;
- детерминированная анонимизация и перестановка ответов;
- 8 параллельных запросов без batch barrier;
- append-only JSONL и resume по стабильному `task_id`;
- Pydantic-валидация входа и ответов judge;
- OpenRouter/DeepSeek V4 Flash по умолчанию.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e "judge[dev]"
export OPENROUTER_API_KEY="..."
```

В PowerShell:

```powershell
$env:OPENROUTER_API_KEY="..."
```

## Вход

Одна JSON-строка на scenario:

```json
{
  "prompt_id": "scenario-001",
  "scenario": "What should the assistant do next?",
  "answers": [
    {"answer_id": "baseline", "text": "First answer"},
    {"answer_id": "steered", "text": "Second answer"}
  ],
  "metadata": {"dataset": "shared-core-v1", "split": "test"}
}
```

`answer_id` нужен только для обратного соединения результатов. Judge получает
анонимные `answer_0`, `answer_1` или `A`, `B`.

## Запуск

Scalar:

```bash
hybrid-judge judge/examples/input.example.jsonl runs/scalar.jsonl \
  --feature optimism
```

Pairwise с проверкой перестановки сторон:

```bash
hybrid-judge judge/examples/input.example.jsonl runs/pairwise.jsonl \
  --mode pairwise \
  --feature optimism \
  --both-orders
```

Полезные параметры:

```text
--model deepseek/deepseek-v4-flash
--workers 8
--seed 20260728
--config-root judge
```

Повторный запуск с тем же output пропускает уже записанные `task_id`.
Неудачные запросы остаются незаписанными и будут повторены.

## Структура

```text
judge/
├── config/
│   └── judge.yaml
├── examples/input.example.jsonl
├── prompts/
│   ├── scalar_v1.txt
│   └── pairwise_v1.txt
├── src/hybrid_judge/
│   ├── cli.py
│   ├── config.py
│   ├── models.py
│   └── runner.py
└── tests/test_judge.py
```

Промпт, использованный в результате, не редактируется задним числом:
создаётся `scalar_v2.txt` или `pairwise_v2.txt`, а filename сохраняется в
provenance.

## Командная работа

- общий Judge меняется в `feat/...` или `fix/...`;
- конкретная абляция живёт в `exp/<concept>-<ablation>`;
- `main` содержит проверенный общий код;
- большие outputs и API-ключи в Git не добавляются.

Подробнее: корневые `AGENTS.md` и `CONTRIBUTING.md`.
