"""Single-device ports of the historical HyS runtime, cache and CAA hooks.

Source paths and behavioral invariants are recorded in HANDOFF.md.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import median

import torch

FEATURES = "NFTRP"
LAYERS = tuple(i for i in range(32) if i % 4 != 3)
HYS_ALPHA = dict(zip(FEATURES, (2.5, 8.0, 2.0, 4.0, 1.5), strict=True))
HYS_BETA = dict(zip(FEATURES, (0.25, 0.50, 0.50, 0.50, 0.25), strict=True))
WINDOWS = {"N": 28, "P": 32}
LAYER_SETS = {
    "gdn24": LAYERS,
    "all32": tuple(range(32)),
    "mid12_gdn": tuple(i for i in LAYERS if 8 <= i < 24),
    "mid16_all": tuple(range(8, 24)),
    "early_mid12_gdn": tuple(i for i in LAYERS if 4 <= i < 20),
}


def rss_sum(pieces, mode="rss"):
    """Eq. 27, on FP32 weighted directions before model-dtype conversion."""
    if mode not in ("rss", "raw"):
        raise ValueError(f"Unknown composition mode: {mode}")
    raw = sum(pieces[1:], pieces[0].clone())
    if mode == "raw" or len(pieces) == 1:
        return raw, 1.0
    target = math.sqrt(sum(float(piece.square().sum()) for piece in pieces))
    gamma = target / max(float(raw.norm()), 1e-12)
    if not math.isfinite(gamma):
        raise FloatingPointError("nonfinite RSS coefficient")
    return raw * gamma, gamma


def gamma_stats(gammas):
    values = list(gammas.values())
    return {
        "min": min(values),
        "median": median(values),
        "max": max(values),
        "per_layer": gammas,
    }


def caa_phases(
    directions,
    alphas,
    strength,
    norm_mode,
    composition_mode,
    use_windows,
    device,
    layers=LAYERS,
    windows=WINDOWS,
    *,
    per_concept=None,
):
    if per_concept is None:
        per_concept = {
            m: {
                "layers": layers,
                "norm_mode": norm_mode,
                "value": alpha,
                "window": windows.get(m) if use_windows else None,
            }
            for m, alpha in alphas.items()
        }
    else:
        layers = sorted({i for spec in per_concept.values() for i in spec["layers"]})
    pieces = {}
    for m, spec in per_concept.items():
        per_layer = {i: directions[m][i] for i in spec["layers"]}
        mode = spec["norm_mode"]
        alpha = spec["value"]
        if mode == "unit":
            pieces[m] = {
                i: (v / v.norm()) * (strength * alpha / math.sqrt(len(per_layer)))
                for i, v in per_layer.items()
            }
        elif mode == "natural":
            pieces[m] = {i: v * (strength * alpha) for i, v in per_layer.items()}
        else:
            raise ValueError(f"Unknown CAA norm mode: {mode}")
    ends = {m: s["window"] for m, s in per_concept.items() if s["window"] is not None}
    # Bridge call 1 produces token t=0, so a window of n ends at call n+1.
    starts = [1] + sorted({n + 1 for n in ends.values()})
    phases, stats = {}, []
    for start in starts:
        members = [m for m in per_concept if start <= ends.get(m, math.inf)]
        offsets, gammas = {}, {}
        for i in layers:
            layer_pieces = [pieces[m][i] for m in members if i in pieces[m]]
            if layer_pieces:
                offset, gamma = rss_sum(layer_pieces, composition_mode)
                offsets[i] = offset.to(device=device)
                gammas[i] = gamma
            else:
                gammas[i] = 1.0
        phases[start] = offsets
        stats.append({"start_call": start, "active": "".join(members), **gamma_stats(gammas)})
    return phases, stats


def hys_coefficients(directions, alphas, strength, mode):
    # Raw ablation path from qwen9b-gdn-rss-ablation-dev8-20261007/search.py.
    gammas = {
        i: rss_sum(
            [directions[m][i] * (strength * a) for m, a in alphas.items()],
            "raw" if mode == "no_rss" else "rss",
        )[1]
        for i in LAYERS
    }
    return (
        {i: {m: a * strength * g for m, a in alphas.items()} for i, g in gammas.items()},
        gamma_stats(gammas),
    )


@dataclass
class Step:
    phase: str = ""
    token_call: int = 0  # bridge=1, then one increment per decode forward


def caa_hook(step, layer, phases, audit):
    def hook(_module, _inputs, output):
        if step.phase not in ("bridge", "decode"):
            return output
        active = phases[max(i for i in phases if i <= step.token_call)]
        delta = active.get(layer)
        if delta is None:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if (
            hidden.ndim != 3
            or hidden.shape != (1, 1, delta.numel())
            or hasattr(hidden, "placements")
        ):
            raise ValueError(f"unexpected CAA residual at layer {layer}: {hidden.shape}")
        changed = hidden.clone()
        changed[:, -1, :] += delta.to(hidden)
        if not bool(torch.isfinite(changed).all()):
            raise FloatingPointError(f"nonfinite CAA residual at layer {layer}")
        audit[layer] = audit.get(layer, 0) + 1
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    return hook


def _tensors(value, path="cache", seen=None):
    """Tensor traversal ported from hybrid_steering/cache.py."""
    if seen is None:
        seen = set()
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            yield from _tensors(value[key], f"{path}[{key!r}]", seen)
    elif isinstance(value, (tuple, list)):
        for i, v in enumerate(value):
            yield from _tensors(v, f"{path}[{i}]", seen)
    elif hasattr(value, "__dict__"):
        for name, v in sorted(vars(value).items()):
            yield from _tensors(v, f"{path}.{name}", seen)


def recurrent_tensor(cache, layer):
    tensors = list(_tensors(cache.layers[layer].recurrent_states))
    if len(tensors) != 1:
        raise ValueError(f"layer {layer} must have one recurrent tensor")
    return tensors[0][1]


def snapshot_nonrecurrent(cache):
    from hashlib import sha256

    return {
        p: sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
        for p, t in _tensors(cache)
        if "recurrent_states" not in p
    }


def gram_inverse(basis, ridge=1e-6):
    gram = basis @ basis.T
    scale = torch.diag(gram).mean().clamp_min(1e-12)
    return torch.linalg.inv(gram + torch.eye(len(basis), device=gram.device) * ridge * scale)


def make_clamp_runtime(cache, directions, deltas, *, no_gram=False, device=None):
    """Port of GDN make_clamp_runtime, plus independent no_gram coordinates."""
    result = {}
    for layer, weights in deltas.items():
        state = recurrent_tensor(cache, layer)
        names = tuple(weights)
        if (
            hasattr(state, "placements")
            or not state.is_floating_point()
            or (device is not None and state.device != device)
            or any(directions[m][layer].shape != state.shape for m in names)
        ):
            raise ValueError(
                f"unexpected recurrent state at layer {layer}: {state.shape}, {state.device}"
            )
        # Match the historical direction cast through the cache dtype.
        basis = torch.stack([directions[m][layer].to(state).float().flatten() for m in names])
        norms_sq = basis.square().sum(dim=1)
        if not bool(torch.isfinite(state).all() & (norms_sq > 0).all()):
            raise FloatingPointError(f"invalid initial state/direction at layer {layer}")
        inverse = None if no_gram else gram_inverse(basis)
        dots = basis @ state.float().flatten()
        requested = torch.tensor(list(weights.values()), device=state.device, dtype=torch.float32)
        if no_gram:
            # Both solvers aim for the projections after the same first write.
            # Store independent coordinates, including the cross terms in that write.
            target = (dots + basis @ (requested @ basis)) / norms_sq
        else:
            target = inverse @ dots + requested
        result[layer] = (
            names,
            basis,
            inverse,
            norms_sq,
            target,
            requested,
            state.shape,
            state.dtype,
        )
    return result


def checked_state(cache, layer, shape, dtype, device):
    state = recurrent_tensor(cache, layer)
    if state.shape != shape or state.dtype != dtype or state.device != device:
        raise ValueError(f"recurrent layout changed at layer {layer}")
    return state


@torch.no_grad()
def apply_clamp(cache, runtime, scales, *, first=False):
    """Use one shared pre-write snapshot per layer and one simultaneous correction."""
    finite = []
    for layer, (
        names,
        basis,
        inverse,
        norms_sq,
        target,
        requested,
        shape,
        dtype,
    ) in runtime.items():
        state = checked_state(cache, layer, shape, dtype, basis.device)
        if first and inverse is None:
            correction = requested
        else:
            dots = basis @ state.float().flatten()
            current = dots / norms_sq if inverse is None else inverse @ dots
            correction = target - current
        weights = torch.tensor([scales[m] for m in names], device=state.device, dtype=torch.float32)
        delta = (correction * weights) @ basis
        state.add_(delta.reshape(shape).to(state))
        finite.append(torch.isfinite(state).all())
    if finite and not bool(torch.stack(finite).all()):
        raise FloatingPointError("nonfinite recurrent state after clamp")


class RecurrentWrite:
    """Single-GPU port of gdn_hook.recurrent_hook's after-forward schedule."""

    def __init__(self, directions, alphas, deltas, mode, device):
        self.directions = directions
        self.alphas = alphas
        self.deltas = deltas
        self.mode = mode
        self.device = device
        self.runtime = None
        self.writes = 0
        self.audit_first = []

    def after_forward(self, cache, phase):
        first = self.runtime is None
        if first != (phase == "bulk"):
            raise RuntimeError("recurrent context must start at the bulk output")
        if not first and self.mode == "no_clamp":
            finite = [
                torch.isfinite(checked_state(cache, layer, shape, dtype, basis.device)).all()
                for layer, (_, basis, _, _, _, _, shape, dtype) in self.runtime.items()
            ]
            if not bool(torch.stack(finite).all()):
                raise FloatingPointError("nonfinite recurrent state in single-write decode")
            return
        before = snapshot_nonrecurrent(cache) if self.writes < 2 else None
        if first:
            self.runtime = make_clamp_runtime(
                cache,
                self.directions,
                self.deltas,
                no_gram=self.mode == "no_gram",
                device=self.device,
            )
        scales = {m: 1.0 if first else HYS_BETA[m] for m in self.alphas}
        apply_clamp(cache, self.runtime, scales, first=first)
        self.writes += 1
        if before is not None:
            if before != snapshot_nonrecurrent(cache):
                raise RuntimeError("non-recurrent cache changed during clamp")
            self.audit_first.append(
                {"phase": phase, "layers": list(self.runtime), "scales": scales}
            )
