"""Chunked code-leak clamp matches a token loop, including a later decode step."""

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

from hybrid_steering.delta_rule import (
    chunk_gated_delta_rule_clamped,
    recurrent_gated_delta_rule_clamped,
)


def _pack(batch, sequence, heads, key_dim, value_dim, *, parallel_keys=False):
    generator = torch.Generator().manual_seed(0)
    direction = torch.randn(heads, key_dim, value_dim, generator=generator)
    left, singular, right = torch.linalg.svd(direction, full_matrices=False)
    u = left[:, :, 0].contiguous()
    target = (0.7 * singular[:, 0])[:, None] * right[:, 0, :]
    query = torch.randn(batch, sequence, heads, key_dim, generator=generator)
    if parallel_keys:
        key = u.view(1, 1, heads, key_dim).expand(batch, sequence, heads, key_dim).clone()
    else:
        key = torch.randn(batch, sequence, heads, key_dim, generator=generator)
    value = torch.randn(batch, sequence, heads, value_dim, generator=generator)
    gate = -torch.rand(batch, sequence, heads, generator=generator)
    beta = torch.rand(batch, sequence, heads, generator=generator)
    initial = torch.randn(batch, heads, key_dim, value_dim, generator=generator)
    return query, key, value, gate, beta, u, target, initial


def _token_loop(query, key, value, g, beta, u, target, initial, *, l2norm, clamp):
    """Independent spec: HF token recurrence, optional code-leak clamp after the read."""
    query, key, value, g, beta = (
        tensor.transpose(1, 2).float().contiguous() for tensor in (query, key, value, g, beta)
    )
    if l2norm:
        query = query * torch.rsqrt((query * query).sum(-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)
    query = query * (query.shape[-1] ** -0.5)
    state = torch.zeros(query.shape[0], query.shape[1], query.shape[-1], value.shape[-1])
    if initial is not None:
        state = initial.float().clone()
    outputs = []
    for index in range(query.shape[2]):
        gamma = g[:, :, index].exp()[:, :, None, None]
        state = state * gamma
        read = (state * key[:, :, index, :, None]).sum(-2)
        delta = (value[:, :, index] - read) * beta[:, :, index, None]
        state = state + key[:, :, index, :, None] * delta[:, :, None, :]
        outputs.append((state * query[:, :, index, :, None]).sum(-2))
        if clamp:
            along = torch.einsum("hk,bhkv->bhv", u.float(), state)
            state = state - torch.einsum("hk,bhv->bhkv", u.float(), along)
            state = state + torch.einsum("hk,hv->hkv", u.float(), target.float())
    return torch.stack(outputs, 2).transpose(1, 2).contiguous(), state


def _assert_match(actual, expected, u, target) -> None:
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-4, atol=1e-5)
    along = torch.einsum("hk,bhkv->bhv", u, actual[1])
    torch.testing.assert_close(along, target.unsqueeze(0).expand_as(along), rtol=1e-4, atol=1e-5)


def test_unclamped_token_loop_matches_the_torch_chunk_kernel() -> None:
    query, key, value, gate, beta, *_ = _pack(2, 7, 2, 4, 4)
    output, state = _token_loop(
        query, key, value, gate, beta, None, None, None, l2norm=True, clamp=False
    )
    kernel_output, kernel_state = torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        gate,
        beta,
        chunk_size=3,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    torch.testing.assert_close(output, kernel_output, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(state, kernel_state, rtol=1e-4, atol=1e-5)


def test_chunk_clamp_matches_the_token_loop() -> None:
    query, key, value, gate, beta, u, target, initial = _pack(2, 17, 3, 5, 4)
    expected = _token_loop(
        query, key, value, gate, beta, u, target, initial, l2norm=False, clamp=True
    )
    _assert_match(
        recurrent_gated_delta_rule_clamped(
            query, key, value, gate, beta, u, target, initial_state=initial
        ),
        expected,
        u,
        target,
    )
    for chunk_size in (1, 4, 64):
        _assert_match(
            chunk_gated_delta_rule_clamped(
                query,
                key,
                value,
                gate,
                beta,
                u,
                target,
                chunk_size=chunk_size,
                initial_state=initial,
            ),
            expected,
            u,
            target,
        )


def test_chunk_clamp_matches_with_kernel_l2norm() -> None:
    query, key, value, gate, beta, u, target, initial = _pack(2, 9, 2, 8, 3)
    expected = _token_loop(
        query, key, value, gate, beta, u, target, initial, l2norm=True, clamp=True
    )
    _assert_match(
        chunk_gated_delta_rule_clamped(
            query,
            key,
            value,
            gate,
            beta,
            u,
            target,
            chunk_size=4,
            initial_state=initial,
            use_qk_l2norm_in_kernel=True,
        ),
        expected,
        u,
        target,
    )


def test_prefill_chunk_then_one_generation_step() -> None:
    query, key, value, gate, beta, u, target, initial = _pack(2, 10, 2, 4, 4)
    prefix = tuple(tensor[:, :6] for tensor in (query, key, value, gate, beta))
    suffix = tuple(tensor[:, 6:] for tensor in (query, key, value, gate, beta))
    prefill_output, prefill_state = chunk_gated_delta_rule_clamped(
        *prefix, u, target, chunk_size=4, initial_state=initial
    )
    step_output, step_state = recurrent_gated_delta_rule_clamped(
        *suffix, u, target, initial_state=prefill_state
    )
    full_output, full_state = _token_loop(
        query, key, value, gate, beta, u, target, initial, l2norm=False, clamp=True
    )
    torch.testing.assert_close(prefill_output, full_output[:, :6], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(step_output, full_output[:, 6:], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(step_state, full_state, rtol=1e-4, atol=1e-5)


def test_per_row_target_matches_separate_calls() -> None:
    query, key, value, gate, beta, u, target, initial = _pack(2, 5, 2, 4, 3)
    per_row = torch.stack((target, target * 0.4))
    both, both_state = chunk_gated_delta_rule_clamped(
        query, key, value, gate, beta, u, per_row, chunk_size=3, initial_state=initial
    )
    for row in range(2):
        output, state = chunk_gated_delta_rule_clamped(
            query[row : row + 1],
            key[row : row + 1],
            value[row : row + 1],
            gate[row : row + 1],
            beta[row : row + 1],
            u,
            per_row[row],
            chunk_size=3,
            initial_state=initial[row : row + 1],
        )
        torch.testing.assert_close(both[row], output[0], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(both_state[row], state[0], rtol=1e-4, atol=1e-5)


def test_keys_parallel_to_u_stay_finite() -> None:
    query, key, value, gate, beta, u, target, initial = _pack(1, 6, 2, 4, 3, parallel_keys=True)
    beta = torch.ones_like(beta)
    expected = _token_loop(
        query, key, value, gate, beta, u, target, initial, l2norm=False, clamp=True
    )
    actual = chunk_gated_delta_rule_clamped(
        query, key, value, gate, beta, u, target, chunk_size=3, initial_state=initial
    )
    _assert_match(actual, expected, u, target)
    plain, _ = torch_chunk_gated_delta_rule(
        query, key, value, gate, beta, chunk_size=3, output_final_state=True
    )
    assert (actual[0] - plain).abs().max() > 1e-2
