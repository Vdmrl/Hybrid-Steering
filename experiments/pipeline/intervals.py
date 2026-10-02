"""Deterministic 95% intervals over independent prompt units."""

import random


def bootstrap_ratio(pairs: list[tuple[float, int]], *, draws: int = 2000, seed: int = 42) -> dict:
    """Resample prompts, preserving all instructions within each prompt.

    Each pair is (numerator, denominator). For a mean Judge score use
    denominator 1; for IFEval instructions use successes and instruction count.
    """
    if not pairs or any(denominator < 1 for _, denominator in pairs) or draws < 100:
        raise ValueError("nonempty prompt contributions and >=100 draws required")
    numerator = sum(value for value, _ in pairs)
    denominator = sum(count for _, count in pairs)
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        selected = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        samples.append(sum(value for value, _ in selected) / sum(count for _, count in selected))
    samples.sort()
    return {
        "estimate": numerator / denominator,
        "low": samples[int(0.025 * draws)],
        "high": samples[min(draws - 1, int(0.975 * draws))],
        "n_prompts": len(pairs),
        "n_items": denominator,
        "draws": draws,
        "seed": seed,
    }
