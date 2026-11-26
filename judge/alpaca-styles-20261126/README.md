# Alpaca style benchmark · 26 November 2026

The four training banks share 150 Stanford Alpaca tasks (75 questions and 75 instructions) and the same 150 neutral answers. The positive answer for each style was written independently by DeepSeek for the same task. A pair can therefore differ in factual details as well as style. Pair IDs are shared across banks. Keep the same ID in the same split across concepts.

| Training pairs | Trait evaluator |
| --- | --- |
| [Numbered](../../concepts/neutral-numbered/data/pairs.jsonl) | [Deterministic sentence-list parser](numbered.py) |
| [Technical](../../concepts/neutral-technical/data/pairs.jsonl) | [Technical register, v3](prompts/technical.txt) |
| [Theistic](../../concepts/neutral-theistic/data/pairs.jsonl) | [Personal faith in God](prompts/theistic.txt) |
| [Probabilistic](../../concepts/neutral-probabilistic/data/pairs.jsonl) | [Personal epistemic uncertainty](prompts/probabilistic.txt) |

## Scoring an answer

Use the complete text of the chosen prompt file as the system message. The user message is a JSON string. Pass only the answer to [Coherence](prompts/coherence.txt). Pass the answer and the original user question to each trait judge and to [Relevance](prompts/relevance.txt). The benchmark sends no method, scale, expected label, or generation instructions. It uses the model and decoding settings in [protocol.json](protocol.json).

This example uses an OpenAI-compatible chat client pointed at the deployed DeepSeek model. Run it from this directory with your client, task and answer. The prompt files contain the exact system text, including the shared single-digit instruction.

```python
import json
import math
from pathlib import Path

def judge_messages(dimension, question, answer):
    system = Path("prompts", dimension + ".txt").read_text(encoding="utf-8")
    data = {"answer": answer}
    if dimension != "coherence":
        data["question"] = question
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(data, ensure_ascii=False)},
    ]

def expected_score(logp):
    # logp maps each literal digit to its unmasked first-token log probability.
    values = [logp[d] for d in "012"]
    weights = [math.exp(x - max(values)) for x in values]
    return (weights[1] + 2 * weights[2]) / sum(weights)

def first_token_logprobs(client, dimension, question, answer):
    cfg = json.loads(Path("protocol.json").read_text())
    response = client.chat.completions.create(
        model=cfg["model"], messages=judge_messages(dimension, question, answer),
        temperature=1, top_p=1, max_tokens=1, logprobs=True, top_logprobs=20,
        extra_body={"chat_template_kwargs": cfg["chat_template_kwargs"]},
    )
    entries = response.choices[0].logprobs.content[0].top_logprobs
    return {entry.token: entry.logprob for entry in entries}
```

Compute the expectation from all three literal digit tokens. If a digit falls outside the top 20, the run requests its unmasked log probability by token ID from vLLM, then teacher-forces any missing digit on the exact same chat prefix. Verify the single-token IDs with the deployed tokenizer when using another endpoint. The example raises a key error for missing digits. Record coherence and relevance separately. The optional combined quality score is their harmonic mean, with zero when both scores are zero.

For Numbered, run `from numbered import metric` and call `metric(answer, capped=hit_generation_limit)["fmt"]`. A complete uncapped list starting at 1, increasing by one and containing one detected sentence per nonempty line earns 2. An answer that starts at 1 or contains a pair of consecutive numbers among its numbered lines earns 1 if it fails the strict check. All other answers earn 0. Thus a prose introduction followed by a list can earn at most 1. The parser also returns the strict-pass flag and the valid-prefix fraction.

Compare scores within this benchmark version. The repository's existing mixed-scale judge has its own rubrics and score definitions.
