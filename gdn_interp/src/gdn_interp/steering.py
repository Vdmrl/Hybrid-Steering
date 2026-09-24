import torch
from torch import Tensor


def add_delta(state: Tensor, delta: Tensor, scale: float | Tensor, normalize: bool = True) -> Tensor:
    """Add a delta to recurrent states, optionally scaled to each head's state norm."""
    correction = delta.to(state)
    if normalize:
        delta_norm = torch.linalg.matrix_norm(correction.float(), dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        state_norm = torch.linalg.matrix_norm(state.float(), dim=(-2, -1), keepdim=True)
        correction = correction * torch.where(state_norm > 0, state_norm / delta_norm, torch.ones_like(state_norm)).to(correction)
    scales = torch.as_tensor(scale, device=state.device, dtype=state.dtype)
    correction = scales.reshape(-1, 1, 1, 1) * correction if scales.ndim else scales * correction
    state.add_(correction)
    return correction
