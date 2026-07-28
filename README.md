# Hybrid Steering

Общий репозиторий экспериментов по steering гибридных LLM.

Сейчас реализован первый общий компонент:

- [`judge/`](judge/) — blind LLM-as-a-Judge для одинаковой оценки разных
  абляций.

## Быстрый старт Judge

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e "judge[dev]"
export OPENROUTER_API_KEY="..."

hybrid-judge judge/examples/input.example.jsonl runs/scalar.jsonl \
  --feature optimism
```

Подробности: [`judge/README.md`](judge/README.md).

## Работа в команде

- общий код: `feat/...`, `fix/...`;
- эксперименты: `exp/<concept>-<ablation>`;
- не коммитить эксперименты напрямую в `main`;
- формат коммитов: Conventional Commits.

Правила для людей: [`CONTRIBUTING.md`](CONTRIBUTING.md).
Правила для агентов: [`AGENTS.md`](AGENTS.md).
