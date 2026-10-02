import torch
import torch.nn.functional as F


def rank_metrics(state: torch.Tensor) -> dict[str, torch.Tensor]:
    singular = torch.linalg.svdvals(state.float())
    energy = singular.square()
    normalized_energy = energy / energy.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(energy.dtype).tiny)
    probability = singular / singular.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(singular.dtype).tiny)
    return {
        "frobenius_norm": torch.linalg.matrix_norm(state.float(), ord="fro"),
        "effective_rank": (-torch.special.xlogy(probability, probability).sum(dim=-1)).exp(),
        "stable_rank": energy.sum(dim=-1) / energy[..., 0].clamp_min(torch.finfo(energy.dtype).tiny),
        **{f"r_{percent}": (normalized_energy.cumsum(dim=-1) < percent / 100).sum(dim=-1) + 1 for percent in (50, 90, 99)},
    }


def state_transition_metrics(
    state: torch.Tensor, key: torch.Tensor, value: torch.Tensor, beta: torch.Tensor, gate: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Per-head metrics for S_t and the t+1 delta-rule update."""
    memory = state.float().transpose(-2, -1)  # [batch, head, value, key]
    left, singular, right = torch.svd_lowrank(memory, q=2, niter=2)
    u, r = left[..., :, 0], right[..., :, 0]
    key, value = key.float(), value.float()
    beta = beta.float()
    value_norm = value.norm(dim=-1).clamp_min(torch.finfo(value.dtype).tiny)
    memory_key = memory @ key.unsqueeze(-1)
    old_without_decay = memory - beta[..., None, None] * memory_key * key.unsqueeze(-2)
    old_norm = old_without_decay.square().sum(dim=(-2, -1)).sqrt().clamp_min(torch.finfo(old_without_decay.dtype).tiny)
    log_rho = beta.clamp_min(torch.finfo(beta.dtype).tiny).log() + value_norm.log() - gate.float() - old_norm.log()
    suppressed_r = r - beta[..., None] * key * (r * key).sum(-1, keepdim=True)
    return {
        "sv1": singular[..., 0],
        "sv_ratio": singular[..., 1] / singular[..., 0].clamp_min(torch.finfo(singular.dtype).tiny),
        "sim_v": F.cosine_similarity(u, value, dim=-1, eps=torch.finfo(value.dtype).tiny).abs(),
        "sim_k": F.cosine_similarity(r, key, dim=-1, eps=torch.finfo(key.dtype).tiny).abs(),
        "log_rho": log_rho,
        "key_retain": suppressed_r.norm(dim=-1),
    }
