from typing import Any
from jaxtyping import Float, Float32

import torch
from torch import Tensor
from .utils import truncate_svd


def difference_states(
    concept_b: dict[int, Tensor],
    concept_a: dict[int, Tensor],
    rank: int | None = 0,
) -> dict[int, Tensor]:
    """Return concept B minus concept A, optionally truncated to a given rank."""
    if concept_b.keys() != concept_a.keys():
        raise ValueError("concept state layers do not match")
    return {layer: truncate_svd(concept_b[layer] - concept_a[layer], rank) for layer in concept_b}


def mean_states(model: Any, inputs: Any, layers: list[int]) -> dict[int, Float32[Tensor, "heads key value"]]:
    """Collect mean recurrent states from a model forward pass."""
    cache = model(**inputs, use_cache=True, logits_to_keep=1).past_key_values
    return {layer: cache.layers[layer].recurrent_states[0].float().mean(0) for layer in layers}


class Stats:
    """Elementwise Welford statistics, persisted on CPU as float32."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = self.m2 = None

    def add_batch(self, values: torch.Tensor) -> None:
        """Merge a batch whose first axis is examples."""
        values = values.detach().float().cpu()
        batch_count = len(values)
        batch_mean = values.mean(0)
        batch_m2 = (values - batch_mean).square().sum(0)
        if self.mean is None:
            self.count, self.mean, self.m2 = batch_count, batch_mean, batch_m2
            return
        assert self.m2 is not None
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.m2 += batch_m2 + delta.square() * self.count * batch_count / total
        self.mean += delta * batch_count / total
        self.count = total

    def save(self) -> dict[str, Any]:
        assert self.mean is not None and self.m2 is not None
        return {"count": self.count, "mean": self.mean, "std": (self.m2 / max(self.count - 1, 1)).sqrt()}


def effective_rank(singular: torch.Tensor) -> torch.Tensor:
    p = singular / singular.sum(-1, keepdim=True).clamp_min(torch.finfo(singular.dtype).tiny)
    return (-torch.special.xlogy(p, p).sum(-1)).exp()


class LayerAccumulator:
    """Aggregate every head and every example in a layer in tensor batches."""

    def __init__(self, shape: tuple[int, ...], device: torch.device, fast: bool = True) -> None:
        self.count, self.fast = 0, fast
        self.mean_delta = torch.zeros(shape, dtype=torch.float32, device=device)
        self.mean_a = torch.zeros(shape, dtype=torch.float32, device=device)
        self.mean_b = torch.zeros(shape, dtype=torch.float32, device=device)
        self.stats: dict[str, Stats] = {}
        self.cos_delta_mean: list[torch.Tensor] = []

    def observe(self, name: str, values: torch.Tensor) -> None:
        self.stats.setdefault(name, Stats()).add_batch(values)

    def add(self, a: torch.Tensor, b: torch.Tensor) -> None:
        a, b = a.detach().float(), b.detach().float()
        if a.ndim != 4 or a.shape != b.shape or a.shape[1:] != self.mean_delta.shape:
            raise ValueError(f"Expected [batch, *{tuple(self.mean_delta.shape)}] states, got {tuple(a.shape)} and {tuple(b.shape)}")

        delta = a - b
        batch = len(delta)

        new_count = self.count + batch
        weight = batch / new_count

        self.mean_delta += weight * (delta.mean(0) - self.mean_delta)
        self.mean_a += weight * (a.mean(0) - self.mean_a)
        self.mean_b += weight * (b.mean(0) - self.mean_b)

        self.count = new_count

        if self.fast:
            return

        (u_a, s_a, vh_a), (u_b, s_b, vh_b) = torch.linalg.svd(a, full_matrices=False), torch.linalg.svd(b, full_matrices=False)
        self.observe("fro_state", (torch.linalg.matrix_norm(a) + torch.linalg.matrix_norm(b)) / 2)
        self.observe("fro_delta", torch.linalg.matrix_norm(delta))
        self.observe("effective_rank_a", effective_rank(s_a))
        self.observe("effective_rank_b", effective_rank(s_b))
        # SVD vector signs are arbitrary, so similarity is sign-invariant.
        self.observe("sim_k", torch.nn.functional.cosine_similarity(u_a[..., :, 0], u_b[..., :, 0], dim=-1).abs())
        self.observe("sim_v", torch.nn.functional.cosine_similarity(vh_a[..., 0, :], vh_b[..., 0, :], dim=-1).abs())

    def add_cosine(self, a: torch.Tensor, b: torch.Tensor) -> None:
        a, b = a.detach().float(), b.detach().float()
        if a.ndim != 4 or a.shape != b.shape or a.shape[1:] != self.mean_delta.shape:
            raise ValueError(f"Expected [batch, *{tuple(self.mean_delta.shape)}] states, got {tuple(a.shape)} and {tuple(b.shape)}")
        delta = a - b
        cosine = torch.nn.functional.cosine_similarity(delta.flatten(2), self.mean_delta.flatten(1), dim=-1)
        self.cos_delta_mean.append(cosine.cpu())
        self.observe("cos_delta_mean", cosine)

    def save(self) -> list[dict[str, Any]]:
        stats = {name: stat.save() for name, stat in self.stats.items()}
        cosine = torch.cat(self.cos_delta_mean) if self.cos_delta_mean else torch.empty((0, len(self.mean_delta)))
        rows = []
        for head in range(len(self.mean_delta)):
            row = {
                "count": self.count,
                "mean_delta": self.mean_delta[head].cpu(),
                "mean_a": self.mean_a[head].cpu(),
                "mean_b": self.mean_b[head].cpu(),
            }
            if not self.fast:
                row.update(
                    scalars={
                        name: {key: value[head] if isinstance(value, torch.Tensor) else value for key, value in stat.items()}
                        for name, stat in stats.items()
                    },
                    cos_delta_mean=cosine[:, head],
                )
            rows.append(row)
        return rows
