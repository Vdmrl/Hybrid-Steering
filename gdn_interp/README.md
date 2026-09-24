# GDN Interp

Small Python utilities and reproducible experiments for Qwen Gated DeltaNet
(GDN) recurrent states.

## Setup

```bash
uv sync
uv run python -m unittest discover -s tests
```

Use `CUDA_VISIBLE_DEVICES=0` for GPU commands on the shared server.

## Layout

- `gdn_interp/`: reusable model loading, steering, and state-metric code.
- `experiments/context_length/`: one-shot `S₀` steering across context lengths.
- `experiments/state_dynamics/`: recurrent-state dynamics analysis.
- `experiments/toy_models/`: attention-only Transformer Circuits reproductions.
- `artifacts/<experiment>/<YYYYMMDD-HHMMSS>/`: every input, intermediate,
  response, log, and plot belonging to one run.

Each experiment directory contains its short reproduction guide.

## Steering

`GDNRunner` has one prompt path and one decode path. The prompt is a single
left-padded batch and makes one call to the model. NNsight intercepts only the
GDN state-update kernel. Projections, convolution, full attention and MLPs
still process the complete batch once.

The padded `input_ids` and `attention_mask` go directly from the tokenizer to
prefill; the runner does not unpack prompts into a Python list and pad them
again.

`prompt_steer_position` counts real prompt tokens already processed. `0` means
the initial GDN state, `-1` means immediately before the final prompt token,
and `None` disables prompt steering. It can be a scalar or a tensor with one
position per row. `scale` has the same scalar-or-per-row form; a zero scale
leaves that row unchanged.

| Operation | Arguments | Where the state changes | Model-forward boundary | Batch behavior |
| --- | --- | --- | --- | --- |
| No prompt intervention | `prompt_steer_position=None` | Nowhere | One prefill forward | All rows follow the ordinary model path. |
| Exact prompt intervention | `prompt_steer_position=p`, `prompt_steer_mode="exact"` | Inside GDN, after exactly `p` real tokens for each row | No extra model forward. The intercepted kernel runs its prefix, its recurrent state is changed, then its suffix runs. | Each row can have a distinct `p` and scale. Distinct positions add GDN-kernel calls, while the rest of the model remains one batch forward. |
| Chunk-aligned prompt intervention | `prompt_steer_mode="chunk"` | At the preceding 64-token GDN chunk boundary | No extra model forward | The boundary is aligned in the left-padded batch coordinate and is clamped to the first real token. Its effective real-token position can change if prompt lengths in the batch change. |
| Initial-state intervention | `prompt_steer_position=0` | Before the first real token | No extra model forward | Normalized deltas have zero magnitude here because the current state is zero; use `normalize=False` when this is intentional. |
| Decode intervention | `generation_steer_steps={...}` or `generation_steer_period=n` | Directly in `past_key_values` after selecting token `n`, immediately before the decode forward that consumes it | There is already one model forward per generated token. The delta is added between two decode forwards. | The selected step is shared by unfinished rows. Finished rows receive scale zero. |

`exact` and `chunk` are both prompt modes. They do not make a DynamicCache,
pause the model after a prompt prefix, add the delta, and resume a second prompt
forward. That older design was the source of the right-padding cache bug. The
only internal split is the GDN kernel call required to expose its recurrent
state at the requested boundary.

For example, this steers rows 0 and 2 before their final prompt token, leaves
rows 1 and 3 unchanged, and re-applies the delta after selecting the first
generated token, before the forward that consumes it:

```python
tokens = runner.generate(
    texts,
    scale=torch.tensor([1.0, 0.0, 0.8, 0.0]),
    prompt_steer_position=-1,
    prompt_steer_mode="exact",
    generation_steer_steps={1},
)
```

`normalize=True` rescales every head's delta to the current head-state norm
before multiplying by `scale`. Pass `GDNTrace()` to `generate(..., trace=trace)`
to collect final kernel outputs and states through that same path.
`trace.prompt_positions` contains the effective unpadded positions, which is
especially useful in chunk mode. Trace keys count real tokens and
`trace.indices[position]` identifies their batch rows. Tracing stores full
state snapshots, so keep inspected batches and generations short.

For a local comparison with native greedy generation and mixed steering scales:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m experiments.steering.smoke_generation \
  --deltas artifacts/steering/local_smoke/deltas.pt \
  --output artifacts/steering/local_unified/smoke \
  --questions 4 --scales 0.5 0.8 1.0 1.2 --max-new-tokens 128
```

The pipeline reports a fixed scale grid. Target-language detection measures
language, not answer correctness, and does not select a winning scale.
