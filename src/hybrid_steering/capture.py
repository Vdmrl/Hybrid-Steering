"""Capture GDN state transitions at selected token boundaries."""

from typing import Any, cast

import torch
import torch.nn.functional as F
from jaxtyping import Float
from nnsight import NNsight, save
from nnsight.intervention.envoy import Envoy
from torch import Tensor
from transformers import DynamicCache, PreTrainedModel

from .runner import unwrap_forward

MetricRow = tuple[int, int, int, int, str, float]
# Full-run positions used by the chunk-dynamics experiment. Smoke runs pass a short subset.
CAPTURE_POSITIONS = (*range(1, 32), 64, 128, 256, 512, 1024, 2048, 4096, 8192)
CHUNK = 64


def rank_metrics(
    state: Float[Tensor, "*leading key value"],
) -> dict[str, Float[Tensor, "*leading"]]:
    """Per-head rank summaries. Leading axes are kept; the last two are the head matrix."""
    singular: Float[Tensor, "*leading spectrum"] = torch.linalg.svdvals(state.float())
    energy = singular.square()
    normalized = energy / energy.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(energy.dtype).tiny)
    probability = singular / singular.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(singular.dtype).tiny
    )
    return {
        "frobenius_norm": torch.linalg.matrix_norm(state.float(), ord="fro"),
        "effective_rank": (-torch.special.xlogy(probability, probability).sum(dim=-1)).exp(),
        "stable_rank": energy.sum(dim=-1)
        / energy[..., 0].clamp_min(torch.finfo(energy.dtype).tiny),
        **{
            f"r_{percent}": (normalized.cumsum(dim=-1) < percent / 100).sum(dim=-1) + 1
            for percent in (50, 90, 99)
        },
    }


def transition_metrics(
    state: Float[Tensor, "batch heads key value"],
    key: Float[Tensor, "batch heads key"],
    value: Float[Tensor, "batch heads value"],
    beta: Float[Tensor, "batch heads"],
    gate: Float[Tensor, "batch heads"],
) -> dict[str, Float[Tensor, "batch heads"]]:
    """Per-head metrics for a state and the following delta-rule update."""
    memory: Float[Tensor, "batch heads value key"] = state.float().transpose(-2, -1)
    left: Float[Tensor, "batch heads value rank"]
    singular: Float[Tensor, "batch heads rank"]
    right: Float[Tensor, "batch heads key rank"]
    left, singular, right = torch.svd_lowrank(memory, q=min(2, memory.shape[-1]), niter=2)
    direction_v: Float[Tensor, "batch heads value"] = left[..., :, 0]
    direction_k: Float[Tensor, "batch heads key"] = right[..., :, 0]
    key, value = key.float(), value.float()
    beta = beta.float()
    value_norm = value.norm(dim=-1).clamp_min(torch.finfo(value.dtype).tiny)
    memory_key = memory @ key.unsqueeze(-1)
    old_without_decay = memory - beta[..., None, None] * memory_key * key.unsqueeze(-2)
    old_norm = (
        old_without_decay.square()
        .sum(dim=(-2, -1))
        .sqrt()
        .clamp_min(torch.finfo(old_without_decay.dtype).tiny)
    )
    log_rho = (
        beta.clamp_min(torch.finfo(beta.dtype).tiny).log()
        + value_norm.log()
        - gate.float()
        - old_norm.log()
    )
    suppressed = direction_k - beta[..., None] * key * (direction_k * key).sum(-1, keepdim=True)
    ratio_floor = torch.finfo(singular.dtype).tiny
    return {
        "sv1": singular[..., 0],
        "sv_ratio": singular[..., 1] / singular[..., 0].clamp_min(ratio_floor)
        if singular.shape[-1] > 1
        else singular[..., 0] * 0,
        "sim_v": F.cosine_similarity(
            direction_v, value, dim=-1, eps=torch.finfo(value.dtype).tiny
        ).abs(),
        "sim_k": F.cosine_similarity(
            direction_k, key, dim=-1, eps=torch.finfo(key.dtype).tiny
        ).abs(),
        "log_rho": log_rho,
        "key_retain": suppressed.norm(dim=-1),
    }


def _kernel(layer: Envoy, name: str) -> Any:
    return next(operation for operation in layer.source.operations if name in operation.path)


class ChunkCapture:
    """Read GDN kernel inputs and the resulting state at token boundaries."""

    def __init__(self, model: PreTrainedModel) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        paths = [
            f"model.layers.{index}.linear_attn"
            for index, layer in enumerate(model.model.layers)
            if hasattr(layer, "linear_attn")
        ]
        for path in paths:
            unwrap_forward(model.get_submodule(path))
        self.traced = NNsight(model)
        self.operations = {
            int(path.split(".")[2]): (
                _kernel(cast(Envoy, self.traced.get(path)), "chunk_gated_delta_rule"),
                _kernel(cast(Envoy, self.traced.get(path)), "recurrent_gated_delta_rule"),
            )
            for path in paths
        }

    def capture(
        self,
        segment: Tensor,
        end: int,
        cache: DynamicCache,
        recurrent: bool,
    ) -> dict[int, tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        saved = []
        with (
            torch.inference_mode(),
            self.traced.trace(
                input_ids=segment,
                attention_mask=torch.ones(len(segment), end, device=self.device, dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
            ),
        ):
            for layer, operations in self.operations.items():
                inputs, output = (
                    save(operations[recurrent].inputs),
                    save(operations[recurrent].output),
                )
                saved.append((layer, inputs, output))
        result = {}
        for layer, inputs, output in saved:
            positional, keyword = inputs
            _, state = output
            result[layer] = positional[1], positional[2], keyword["beta"], keyword["g"], state
        return result

    def measure(
        self, input_ids: Tensor, positions: tuple[int, ...] = CAPTURE_POSITIONS
    ) -> list[MetricRow]:
        """Return rows ``(layer, position, document, head, metric, value)``."""
        if not positions:
            raise ValueError("positions must not be empty")
        required = max(positions) + 1
        if input_ids.shape[1] < required:
            raise ValueError(f"need at least {required} tokens to measure every requested boundary")
        cache: DynamicCache = DynamicCache(config=self.model.config)
        rows: list[MetricRow] = []
        pending: dict[int, tuple[int, Tensor]] = {}
        start = 0
        while start < required:
            targets = [position for position in positions if start < position <= start + CHUNK]
            end = targets[0] if targets else min(start + CHUNK, required)
            if start < 32:
                end = start + 1
            captured = self.capture(
                input_ids[:, start:end],
                end,
                cache,
                recurrent=start > 0 and end - start == 1,
            )
            for layer, (key, value, beta, gate, state) in captured.items():
                if layer in pending:
                    position, old_state = pending.pop(layer)
                    measured = transition_metrics(
                        old_state,
                        F.normalize(key[:, 0].float(), dim=-1, eps=1e-6),
                        value[:, 0],
                        beta[:, 0],
                        gate[:, 0],
                    )
                    for metric, metric_values in measured.items():
                        for document, head in torch.cartesian_prod(
                            torch.arange(metric_values.shape[0]),
                            torch.arange(metric_values.shape[1]),
                        ).tolist():
                            rows.append(
                                (
                                    layer,
                                    position,
                                    document,
                                    head,
                                    metric,
                                    float(metric_values[document, head].cpu()),
                                )
                            )
                if end in positions:
                    pending[layer] = end, state
            start = end
        return rows
