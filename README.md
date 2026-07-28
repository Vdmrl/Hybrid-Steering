# Hybrid Steering

Общий репозиторий экспериментов по steering гибридных LLM.

Общие компоненты:

- [`judge/`](judge/) — blind LLM-as-a-Judge для одинаковой оценки разных
  абляций.
- [`steering/`](steering/) — минимальные операции над GDN recurrent state.
- [`concepts/`](concepts/) — единые определения оцениваемых признаков.
- [`experiments/`](experiments/) — формат воспроизводимых experiment manifests.

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

## Быстрый старт Steering

```bash
pip install -e "steering[dev]"
pytest steering/tests -q
```

Подробности: [`steering/README.md`](steering/README.md).

## Работа в команде

- общий код: `feat/...`, `fix/...`;
- каждый эксперимент, включая smoke test, запускается в собственной
  `exp/<concept>-<ablation>` ветке;
- не коммитить эксперименты напрямую в `main`;
- формат коммитов: Conventional Commits.

Правила для людей: [`CONTRIBUTING.md`](CONTRIBUTING.md).
Правила для агентов: [`AGENTS.md`](AGENTS.md).
