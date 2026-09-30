"""Per-token rank-1 clamp as a chunked gated delta rule.

The code-leak clamp runs after every token. The state that enters later tokens
therefore has the form ``S_⊥ + u targetᵀ`` with ``uᵀ S_⊥ = 0``. ``S_⊥`` obeys an
ordinary gated delta rule on the projected key and a shifted value, so the
chunk scan is the same matmul and triangular solve as
``torch_chunk_gated_delta_rule``. The attention output is read before the
clamp, and that read is a position-wise correction of the projected output.
Decoding keeps the one-token recurrence.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


def _l2norm(x: Tensor) -> Tensor:
    """Match the GDN kernel: ``x / sqrt(||x||^2 + 1e-6)``."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)


def _validate(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    u: Tensor,
    target: Tensor,
    initial_state: Tensor | None,
    chunk_size: int | None,
) -> None:
    if query.ndim != 4 or key.shape != query.shape or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [batch, sequence, heads, dim]")
    if value.shape[:3] != query.shape[:3]:
        raise ValueError("query, key, and value must share batch, sequence, and heads")
    if g.shape != query.shape[:3] or beta.shape != g.shape:
        raise ValueError("g and beta must have shape [batch, sequence, heads]")
    batch, sequence, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    if sequence < 1:
        raise ValueError("sequence length must be positive")
    if chunk_size is not None and chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if tuple(u.shape) != (heads, key_dim):
        raise ValueError("u must have shape [heads, key]")
    if target.shape != (heads, value_dim) and target.shape != (batch, heads, value_dim):
        raise ValueError("target must have shape [heads, value] or [batch, heads, value]")
    if initial_state is not None and tuple(initial_state.shape) != (
        batch,
        heads,
        key_dim,
        value_dim,
    ):
        raise ValueError("initial_state must have shape [batch, heads, key, value]")
    norms = u.detach().float().norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-4):
        raise ValueError("u must be a unit vector on each head")


def _heads_first(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    use_qk_l2norm_in_kernel: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Layout ``[batch, heads, sequence, dim]``, GDN L2 norm, and ``q / sqrt(d)``."""
    query, key, value, g, beta = (
        tensor.transpose(1, 2).float().contiguous() for tensor in (query, key, value, g, beta)
    )
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query)
        key = _l2norm(key)
    return query * (query.shape[-1] ** -0.5), key, value, g, beta


def rank1_factors(
    direction: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Top singular triple ``(u, w, sigma)`` of a ``[heads, key, value]`` direction."""
    left, singular, right = torch.linalg.svd(direction.detach().float(), full_matrices=False)
    return (
        left[..., :, 0].contiguous(),
        right[..., 0, :].contiguous(),
        singular[..., 0].contiguous(),
    )


def _clamp_rank1(state: Tensor, u: Tensor, target: Tensor) -> Tensor:
    """Replace each head's component along ``u`` with ``u targetᵀ``."""
    along = torch.einsum("hk,bhkv->bhv", u, state)
    removed = state - torch.einsum("hk,bhv->bhkv", u, along)
    if target.ndim == 2:
        return removed + torch.einsum("hk,hv->hkv", u, target)
    return removed + torch.einsum("hk,bhv->bhkv", u, target)


def _project(
    key: Tensor,
    value: Tensor,
    g: Tensor,
    u: Tensor,
    target: Tensor,
    initial_state: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Map a clamped recurrence onto an ordinary delta rule in ``u``'s complement.

    Incoming coefficient ``a_0`` is ``uᵀ S_0``. After the first clamp it is
    ``target``. The projected value removes the part of that coefficient the
    unprojected key would have written.
    """
    lam = (key * u[None, :, None, :]).sum(dim=-1)
    key_perp = key - lam[..., None] * u[None, :, None, :]
    gamma = g.exp()
    batch, heads, sequence, _ = key.shape
    value_dim = value.shape[-1]
    if initial_state is None:
        along0 = key.new_zeros(batch, heads, value_dim)
        state = key.new_zeros(batch, heads, key.shape[-1], value_dim)
    else:
        state0 = initial_state.to(device=key.device, dtype=torch.float32)
        along0 = torch.einsum("hk,bhkv->bhv", u, state0)
        state = state0 - torch.einsum("hk,bhv->bhkv", u, along0)
    if target.ndim == 2:
        incoming = (
            target.view(1, heads, 1, value_dim).expand(batch, heads, sequence, value_dim).clone()
        )
    else:
        incoming = target[:, :, None, :].expand(batch, heads, sequence, value_dim).clone()
    incoming[:, :, 0] = along0
    value_tilde = value - gamma[..., None] * lam[..., None] * incoming
    return key_perp, value_tilde, state, incoming, lam, gamma


def recurrent_gated_delta_rule_clamped(
    query: Float[Tensor, "batch sequence heads key"],
    key: Float[Tensor, "batch sequence heads key"],
    value: Float[Tensor, "batch sequence heads value"],
    g: Float[Tensor, "batch sequence heads"],
    beta: Float[Tensor, "batch sequence heads"],
    u: Float[Tensor, "heads key"],
    target: Float[Tensor, "heads value"],
    initial_state: Float[Tensor, "batch heads key value"] | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[
    Float[Tensor, "batch sequence heads value"],
    Float[Tensor, "batch heads key value"],
]:
    """One gated-delta step per token, then the code-leak clamp.

    ``u`` is a unit vector per head. ``target`` is the value-side coordinate
    written along ``u`` (``scale * c_ref * sigma * w`` in code-leak). The
    output of a token is read before that token's clamp. Returns float32.
    """
    _validate(query, key, value, g, beta, u, target, initial_state, None)
    query, key, value, g, beta = _heads_first(
        query, key, value, g, beta, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel
    )
    u = u.to(device=query.device, dtype=torch.float32)
    target = target.to(device=query.device, dtype=torch.float32)
    batch, heads, _, key_dim = query.shape
    if initial_state is None:
        state = query.new_zeros(batch, heads, key_dim, value.shape[-1])
    else:
        state = initial_state.to(device=query.device, dtype=torch.float32)
    outputs: list[Tensor] = []
    for index in range(query.shape[2]):
        query_t = query[:, :, index]
        key_t = key[:, :, index]
        gamma = g[:, :, index].exp()[:, :, None, None]
        state = state * gamma
        read = (state * key_t[:, :, :, None]).sum(dim=-2)
        delta = (value[:, :, index] - read) * beta[:, :, index, None]
        state = state + key_t[:, :, :, None] * delta[:, :, None, :]
        outputs.append((state * query_t[:, :, :, None]).sum(dim=-2))
        state = _clamp_rank1(state, u, target)
    output = torch.stack(outputs, dim=2).transpose(1, 2).contiguous()
    return output, state


def _chunk_scan(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    initial_state: Tensor,
    chunk_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Chunked delta rule. Returns output, state, and the per-token write ``δ``."""
    batch, heads, sequence, _ = key.shape
    value_dim = value.shape[-1]
    pad_size = (chunk_size - sequence % chunk_size) % chunk_size
    query, key, value = (F.pad(tensor, (0, 0, 0, pad_size)) for tensor in (query, key, value))
    beta, decay = (F.pad(tensor, (0, pad_size)) for tensor in (beta, g))
    total = sequence + pad_size
    num_chunks = total // chunk_size
    value_beta = value * beta.unsqueeze(-1)
    key_beta = key * beta.unsqueeze(-1)
    query, key, key_beta, value_beta = (
        tensor.reshape(batch, heads, num_chunks, chunk_size, tensor.shape[-1])
        for tensor in (query, key, key_beta, value_beta)
    )
    decay = decay.reshape(batch, heads, num_chunks, chunk_size)
    strictly_upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(
        1
    )
    cumulative = decay.cumsum(dim=-1)
    pairwise = cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)
    pairwise = pairwise.masked_fill(strictly_upper, float("-inf")).exp()
    system = (key_beta @ key.transpose(-1, -2)) * pairwise
    intra = (query @ key.transpose(-1, -2)) * pairwise
    decayed_key_beta = key_beta * cumulative.exp().unsqueeze(-1)
    new_values = torch.linalg.solve_triangular(system, value_beta, upper=False, unitriangular=True)
    key_cumulative = torch.linalg.solve_triangular(
        system, decayed_key_beta, upper=False, unitriangular=True
    )
    state = initial_state.to(dtype=new_values.dtype)
    output = torch.zeros_like(new_values)
    query = query * cumulative.exp().unsqueeze(-1)
    key = key * (cumulative[..., -1:] - cumulative).exp().unsqueeze(-1)
    chunk_decay = cumulative[..., -1].exp()[..., None, None]
    written = torch.empty_like(new_values)
    for index in range(num_chunks):
        delta = new_values[:, :, index] - key_cumulative[:, :, index] @ state
        written[:, :, index] = delta
        output[:, :, index] = query[:, :, index] @ state + intra[:, :, index] @ delta
        state = state * chunk_decay[:, :, index] + key[:, :, index].transpose(-1, -2) @ delta
    output = output.reshape(batch, heads, total, value_dim)[:, :, :sequence]
    written = written.reshape(batch, heads, total, value_dim)[:, :, :sequence]
    return output, state, written


def chunk_gated_delta_rule_clamped(
    query: Float[Tensor, "batch sequence heads key"],
    key: Float[Tensor, "batch sequence heads key"],
    value: Float[Tensor, "batch sequence heads value"],
    g: Float[Tensor, "batch sequence heads"],
    beta: Float[Tensor, "batch sequence heads"],
    u: Float[Tensor, "heads key"],
    target: Float[Tensor, "heads value"],
    chunk_size: int = 64,
    initial_state: Float[Tensor, "batch heads key value"] | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[
    Float[Tensor, "batch sequence heads value"],
    Float[Tensor, "batch heads key value"],
]:
    """Prefill clamp: one chunked delta-rule scan, equivalent to the token loop.

    Same arguments and float32 return as ``recurrent_gated_delta_rule_clamped``.
    ``chunk_size`` is the scan width. Padding inside the last chunk does not
    move a state that is already clamped.
    """
    _validate(query, key, value, g, beta, u, target, initial_state, chunk_size)
    query, key, value, g, beta = _heads_first(
        query, key, value, g, beta, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel
    )
    u = u.to(device=query.device, dtype=torch.float32)
    target = target.to(device=query.device, dtype=torch.float32)
    key_perp, value_tilde, state_perp, incoming, lam, gamma = _project(
        key, value, g, u, target, initial_state
    )
    projected, state_perp, delta = _chunk_scan(
        query, key_perp, value_tilde, g, beta, state_perp, chunk_size
    )
    query_u = (query * u[None, :, None, :]).sum(dim=-1)
    correction = gamma.unsqueeze(-1) * incoming + lam.unsqueeze(-1) * delta
    output = (projected + query_u.unsqueeze(-1) * correction).transpose(1, 2).contiguous()
    if target.ndim == 2:
        state = state_perp + torch.einsum("hk,hv->hkv", u, target)
    else:
        state = state_perp + torch.einsum("hk,bhv->bhkv", u, target)
    return output, state
