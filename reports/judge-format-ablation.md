# Judge output-format ablation

GPT-4o mini was evaluated at temperature 0 on 35 newly authored balanced
cases that were not used to tune the prompts. The cases cover principled
candor, calm composure, concrete language, casualness, optimism, joy, and
French language.

| Format | Exact | MAE | First-pass format validity |
|---|---:|---:|---:|
| Audited JSON with evidence | 25/35 | 0.314 | 35/35 |
| One digit from 1 to 5 | 24/35 | 0.400 | 35/35 |

The formats assigned the same score in 28/35 cases. Among the seven
disagreements, JSON was closer to the label four times and the compact format
three times. This small calibration does not establish statistical
non-inferiority, but it found no practically large accuracy loss.

The experiment queue therefore keeps completed audited JSON judgments and uses
the compact format only to fill missing `(prompt_id, answer_id, feature)` keys.
This avoids paying to judge completed answers again while preserving a clear
provenance boundary between formats.
