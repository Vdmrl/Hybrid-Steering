"""Paired binary-metric differences with reproducible bootstrap intervals."""

import random


def paired_interval(a: list[bool], b: list[bool], *,
                    draws: int = 10000, seed: int = 42) -> dict[str, float]:
    """Return a-b and one-sided 97.5% lower / two-sided 95% bounds."""
    if len(a) != len(b) or not a or draws < 1000:
        raise ValueError("Expected equal nonempty paired outcomes and >=1000 draws")
    if any(type(value) is not bool for value in [*a, *b]):
        raise ValueError("Outcomes must be boolean")
    differences = [int(x) - int(y) for x, y in zip(a, b, strict=True)]
    n = len(differences)
    rng = random.Random(seed)
    samples = sorted(
        sum(differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(draws)
    )
    return {
        "n": n,
        "a_rate": sum(a) / n,
        "b_rate": sum(b) / n,
        "difference": sum(differences) / n,
        "lower_97_5_one_sided": samples[int(0.025 * draws)],
        "upper_95_two_sided": samples[min(draws - 1, int(0.975 * draws))],
        "draws": draws,
        "seed": seed,
    }
