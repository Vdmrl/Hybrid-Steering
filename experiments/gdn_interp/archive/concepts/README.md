# Concept steering

The checked-in benchmark has 20 neutral-condition prompts for each concept. The first side is the positive direction: support−oppose, certain−uncertain, past−future, literal−metaphorical. Five matched prompt pairs generate the source prose whose final recurrent states are averaged and subtracted.

```bash
RUN=artifacts/concepts/$(date -u +%Y%m%d-%H%M%S)
uv run python experiments/concepts/run.py pairs --run-dir "$RUN"
uv run python experiments/concepts/run.py direction --run-dir "$RUN"
uv run python experiments/concepts/run.py benchmark --run-dir "$RUN" --scales -10 0 10
uv run python experiments/concepts/run.py judge --run-dir "$RUN"
uv run python experiments/concepts/report.py --run-dir "$RUN"
```

For the two-GPU run, use one subdirectory per concept and pass `--concept`; `report.py` automatically combines child-run judgments.

`judge_prompt.txt` is the evaluator template. The judge script uses the supplied Qwen model by default and preserves its raw output for audit. The report is self-contained at `report.html`.
