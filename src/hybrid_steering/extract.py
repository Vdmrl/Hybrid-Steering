"""Collect a mean target-minus-source direction from paired texts."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from jaxtyping import Float, Float32
from torch import Tensor

from .runner import Runner, Trace
from .state import Accumulator, BatchIndex, Metric


@dataclass
class CollectedDirection:
    """Mean direction and the states it was averaged from."""

    delta: dict[int, Float32[Tensor, "heads key value"]]
    mean_target: dict[int, Float32[Tensor, "heads key value"]]
    mean_source: dict[int, Float32[Tensor, "heads key value"]]
    observations: dict[int, dict[str, Tensor]]
    pairs: int


def final_states(
    runner: Runner, texts: list[str]
) -> dict[int, Float[Tensor, "batch heads key value"]]:
    """Recurrent state after the last real token of each text. No steering is applied."""
    if hasattr(runner, "final_states"):
        return runner.final_states(texts)
    encoded = runner.tokenizer(
        texts, add_special_tokens=False, padding=True, return_tensors="pt"
    ).to(runner.device)
    lengths: BatchIndex = encoded.attention_mask.sum(-1)
    trace = runner.forward(texts, prompt_position=None)
    states: dict[int, Tensor] = {}
    filled = torch.zeros(len(texts), dtype=torch.bool, device=runner.device)
    for position, by_layer in trace.states.items():
        indices = trace.indices[position]
        final = lengths[indices].eq(position)
        filled[indices[final]] = True
        for layer, values in by_layer.items():
            states.setdefault(layer, values.new_empty((len(texts), *values.shape[1:])))
            states[layer][indices[final]] = values[final]
    if not filled.all() or len(states) != len(runner.layers):
        raise RuntimeError("failed to collect a final recurrent state for every input")
    return states


def collect_direction(
    runner: Runner,
    pairs: Sequence[tuple[str, str]],
    *,
    batch_size: int = 1,
    metrics: Sequence[Metric] = (),
) -> CollectedDirection:
    """Average ``target_text - source_text`` over matched pairs.

    Each pair is ``(target, source)``. The runner must be built with
    ``prompt_position`` left unused: this function forwards with steering off.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not pairs:
        raise ValueError("pairs must not be empty")
    accumulators: dict[int, Accumulator] | None = None
    for start in range(0, len(pairs), batch_size):
        batch = list(pairs[start : start + batch_size])
        target_states = final_states(runner, [target for target, _ in batch])
        source_states = final_states(runner, [source for _, source in batch])
        if accumulators is None:
            accumulators = {
                layer: Accumulator(tuple(state.shape[1:]), metrics)
                for layer, state in target_states.items()
            }
        for layer, accumulator in accumulators.items():
            accumulator.update(target_states[layer], source_states[layer])
    assert accumulators is not None
    return CollectedDirection(
        delta={
            layer: accumulator.mean_delta.clone() for layer, accumulator in accumulators.items()
        },
        mean_target={
            layer: accumulator.mean_target.clone() for layer, accumulator in accumulators.items()
        },
        mean_source={
            layer: accumulator.mean_source.clone() for layer, accumulator in accumulators.items()
        },
        observations={layer: accumulator.summary() for layer, accumulator in accumulators.items()},
        pairs=len(pairs),
    )


def trace_rows(trace: Trace, batch_size: int) -> dict[int, dict[int, Tensor]]:
    """Map each batch row to its prefill output vector at every traced layer."""
    vectors: dict[int, dict[int, Tensor]] = {index: {} for index in range(batch_size)}
    for length, indices in trace.prefill_indices.items():
        for layer, values in trace.prefill_outputs[length].items():
            for row, value in zip(indices.tolist(), values, strict=True):
                vectors[row][layer] = value
    if not trace.prefill_outputs:
        raise RuntimeError("prefill trace is empty")
    expected = len(next(iter(trace.prefill_outputs.values())))
    if any(len(layers) != expected for layers in vectors.values()):
        raise RuntimeError("prefill trace does not contain every configured GDN layer")
    return vectors
