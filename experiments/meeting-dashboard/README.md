# Meeting dashboard

Builds one self-contained HTML report from available experiment `summary.json`
files. It uses only the Python standard library and works while later
experiments are still pending.

```bash
python experiments/meeting-dashboard/build.py \
  --four-axis outputs/four-axis-night/summary.json \
  --calm-french outputs/calm-french-composition/summary.json \
  --candor-french outputs/candor-french-composition/summary.json \
  --optimism outputs/optimism-factorial-extension/summary.json \
  --output outputs/meeting-dashboard/index.html
```

The page shows main steering effects, answer-quality effects, 95% bootstrap
confidence intervals, feature interactions, and the two language-composition
experiments. Missing summaries are shown as pending. The page refreshes every
60 seconds. To rebuild it as results arrive, run:

```bash
bash experiments/meeting-dashboard/watch.sh
```

The watcher paths can be overridden with environment variables documented at
the top of `watch.sh`.

This is intentionally a static report rather than a Streamlit application:
it is easier to copy, archive, and show during a meeting. Add an interactive
raw-response browser only if inspecting individual generations becomes a
recurring need.
