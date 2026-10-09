"""GDN recurrent-state steering math. No model loading or cache configuration."""
import torch
import math


def sparsify_heads(direction: dict, fraction: float) -> tuple[dict, dict]:
    """Keep globally strongest training-difference heads; do not renormalize."""
    if not 0 < fraction <= 1:
        raise ValueError('Head fraction must be in (0, 1]')
    scores = [(float(head.square().sum()), layer, index)
              for layer, value in sorted(direction.items()) for index, head in enumerate(value)]
    keep = math.ceil(len(scores) * fraction)
    selected = sorted(scores, key=lambda item: (-item[0], item[1], item[2]))[:keep]
    result = {}
    for _, layer, head in selected:
        if layer not in result:
            result[layer] = torch.zeros_like(direction[layer])
        result[layer][head] = direction[layer][head]
    total_energy = sum(score for score, _, _ in scores)
    return result, {'active_heads': keep, 'total_heads': len(scores),
                    'layers': len(result), 'selected': [(l, h) for _, l, h in selected],
                    'retained_energy': sum(s for s, _, _ in selected) / max(total_energy, 1e-30),
                    'renormalized': False}


def analyze(direction: dict, device: str, rank: int = 1) -> tuple[dict, dict]:
    """SVD statistics and top-k factors per layer/head; keep legacy rank-1 shapes."""
    stable, rank90 = [], []
    factors = {}
    for i, state in direction.items():
        u, singular, vh = torch.linalg.svd(state, full_matrices=False)
        energy = singular.square()
        stable.extend((energy.sum(-1) / energy[:, 0].clamp_min(1e-12)).tolist())
        cumulative = energy.cumsum(-1) / energy.sum(-1, keepdim=True).clamp_min(1e-12)
        rank90.extend(((cumulative < 0.9).sum(-1) + 1).tolist())
        if not 1 <= rank <= singular.shape[-1]:
            raise ValueError(f"Invalid rank: {rank}")
        if rank == 1:
            selected = (u[:, :, 0], singular[:, 0], vh[:, 0])
        else:
            selected = (u[:, :, :rank], singular[:, :rank], vh[:, :rank])
        factors[i] = tuple(value.to(device) for value in selected)
    stats = {
        "heads": len(stable),
        "stable_rank_mean": sum(stable) / len(stable),
        "rank90_mean": sum(rank90) / len(rank90),
    }
    return stats, factors


def add(cache, direction: dict, strength: float) -> None:
    """Add strength * direction to each selected recurrent state in place."""
    if not strength:
        return
    for i, delta in direction.items():
        state = cache.layers[i].recurrent_states[0]
        state.add_(delta.to(state.device, state.dtype), alpha=strength)


def clamp(cache, rank_one: dict, strength: float) -> None:
    """Set U_k.T @ state to strength * diag(s) @ Vh (per head).

    State: [batch, heads, key_dim, value_dim]. Higher-rank factors:
    U [heads, key_dim, rank], s [heads, rank], Vh [heads, rank, value_dim].
    Rank-1 uses squeezed factors for compatibility and its original arithmetic.
    """
    if not strength:
        return
    for i, (u, singular, w) in rank_one.items():
        state = cache.layers[i].recurrent_states[0]
        u, singular, w = u.to(state), singular.to(state), w.to(state)
        if u.ndim == 3:
            current = torch.einsum("hdr,bhdk->bhrk", u, state)
            target = strength * singular[None, :, :, None] * w[None]
            state.add_(torch.einsum("hdr,bhrk->bhdk", u, target - current))
            continue
        current = torch.einsum("hd,bhdk->bhk", u, state)
        target = strength * singular[None, :, None] * w[None]
        state.add_(u[None, :, :, None] * (target - current)[:, :, None, :])


def clamp_centered(cache, rank_one: dict, anchors: dict, strength: float) -> None:
    """Legacy rank-1 clamp relative to an anchor; preserve its 0.5 scaling."""
    if not strength:
        return
    for i, (u, singular, w) in rank_one.items():
        state = cache.layers[i].recurrent_states[0]
        u, singular, w, center = u.to(state), singular.to(state), w.to(state), anchors[i].to(state)
        current = torch.einsum("hd,bhdk->bhk", u, state)
        target = center[None] + 0.5 * strength * singular[None, :, None] * w[None]
        state.add_(u[None, :, :, None] * (target - current)[:, :, None, :])


def reconstruct(factors: dict) -> dict:
    """Rebuild direction matrices from either legacy rank-1 or rank-k factors."""
    result = {}
    for layer, (u, singular, vh) in factors.items():
        if u.ndim == 2:
            result[layer] = u[:, :, None] * singular[:, None, None] * vh[:, None, :]
        else:
            result[layer] = (u * singular[:, None, :]) @ vh
    return result


def match_head_norm(direction: dict, reference: dict, max_gain: float = 10.) -> tuple[dict, dict]:
    """Match each layer/head Frobenius norm; cap gains for near-zero directions."""
    assert direction.keys() == reference.keys() and max_gain >= 1
    result, report = {}, {}
    for layer, value in direction.items():
        norm = torch.linalg.vector_norm(value, dim=(-2, -1))
        target = torch.linalg.vector_norm(reference[layer], dim=(-2, -1))
        gain = target / norm.clamp_min(1e-12)
        result[layer] = value * gain.clamp(max=max_gain)[..., None, None]
        assert torch.isfinite(result[layer]).all()
        report[layer] = {'gain': gain.clamp(max=max_gain).tolist(),
                         'capped_heads': int((gain > max_gain).sum())}
    return result, report


def clamp_interval(schedule: str) -> int:
    """Validate a schedule; return 0 for nonperiodic modes, N for periodic clamp."""
    if schedule == "clamp":
        return 1
    if schedule.startswith("clamp_every_"):
        interval = int(schedule.removeprefix("clamp_every_"))
        if interval < 1:
            raise ValueError("Clamp interval must be positive")
        return interval
    if schedule in ("once", "clamp_once", "centered"):
        return 0
    raise ValueError(f"Unknown steering schedule: {schedule}")
