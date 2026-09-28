import torch

from hybrid_steering import (
    Accumulator,
    CosineToMean,
    EffectiveRank,
    FrobeniusDelta,
    add_delta,
    clamp_delta,
    combine,
    difference,
    mean,
    truncate_rank,
)


def test_truncate_rank_keeps_the_leading_component() -> None:
    matrix = torch.diag(torch.tensor([4.0, 3.0, 2.0]))
    truncated = truncate_rank(matrix, 1)
    assert torch.linalg.matrix_rank(truncated).item() == 1
    torch.testing.assert_close(truncated, torch.diag(torch.tensor([4.0, 0.0, 0.0])))
    torch.testing.assert_close(truncate_rank(matrix, None), matrix)
    torch.testing.assert_close(truncate_rank(matrix, 0), matrix)


def test_difference_is_target_minus_source() -> None:
    target = {0: torch.ones(1, 2, 2)}
    source = {0: torch.zeros(1, 2, 2)}
    torch.testing.assert_close(difference(target, source)[0], target[0])


def test_normalized_delta_matches_state_norm() -> None:
    state = torch.tensor([[[[3.0, 4.0], [0.0, 0.0]]]])
    correction = add_delta(state, torch.ones(1, 2, 2), 1.0)
    torch.testing.assert_close(torch.linalg.matrix_norm(correction), torch.tensor([[5.0]]))


def test_normalized_delta_is_raw_at_zero_state() -> None:
    state = torch.zeros(1, 1, 2, 2)
    delta = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    correction = add_delta(state, delta, 0.5)
    torch.testing.assert_close(correction, delta[None] * 0.5)


def test_combine_is_a_weighted_sum_and_raw_add_is_linear() -> None:
    first = {0: torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])}
    second = {0: torch.tensor([[[0.0, 2.0], [0.0, 0.0]]])}
    combined = combine([first, second], [0.5, 2.0])
    torch.testing.assert_close(combined[0], 0.5 * first[0] + 2.0 * second[0])

    state = torch.zeros(1, 1, 2, 2)
    left = state.clone()
    right = state.clone()
    both = state.clone()
    add_delta(left, first[0], 1.0, normalize=False)
    add_delta(right, second[0], 1.0, normalize=False)
    add_delta(both, combine([first, second])[0], 1.0, normalize=False)
    torch.testing.assert_close(left + right - state, both)


def test_clamp_closes_the_projection_gap() -> None:
    state = torch.zeros(1, 1, 2, 2)
    direction = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    target = torch.tensor([[[4.0, 0.0], [0.0, 0.0]]])
    clamp_delta(state, direction, target, 1.0)
    projection = (state.float() * direction.float()).sum()
    assert projection.item() == 4.0


def test_accumulator_mean_is_target_minus_source() -> None:
    accumulator = Accumulator((1, 2, 2), (FrobeniusDelta(), EffectiveRank(), CosineToMean()))
    target = torch.ones(2, 1, 2, 2)
    source = torch.zeros(2, 1, 2, 2)
    accumulator.update(target, source)
    torch.testing.assert_close(accumulator.mean_delta, torch.ones(1, 2, 2))
    torch.testing.assert_close(accumulator.mean_target, torch.ones(1, 2, 2))
    torch.testing.assert_close(accumulator.mean_source, torch.zeros(1, 2, 2))
    summary = accumulator.summary()
    assert summary["frobenius_delta"].shape[0] == 2
    assert torch.isfinite(summary["effective_rank"]).all()


def test_mean_rejects_an_empty_collection() -> None:
    try:
        mean([])
    except ValueError as error:
        assert "zero" in str(error)
    else:
        raise AssertionError("empty mean must fail")
