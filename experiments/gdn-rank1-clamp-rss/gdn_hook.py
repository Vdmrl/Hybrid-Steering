"""Bounded replicated recurrent clamp audit; fresh runtime for each generation."""

import hashlib


def recurrent_hook(base, directions, active, alphas, betas, audit, *, zero=False):
    import torch
    import torch.distributed as dist

    runtime = None

    def digest(cache):
        result = {}
        for index in sorted({i for feature in active for i in directions[feature]}):
            state = base.recurrent_tensor(cache, index)
            if hasattr(state, "placements"):
                raise TypeError("unexpected DTensor recurrent state")
            if state.device.type != "cuda" or not torch.isfinite(state).all():
                raise FloatingPointError("invalid recurrent state")
            raw = state.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
            result[index] = [list(state.shape), hashlib.sha256(raw).hexdigest()]
        return result

    def hook(module, inputs, output):
        nonlocal runtime
        cache = output.past_key_values
        first = runtime is None
        if first:
            before = base.snapshot_nonrecurrent(cache)
            state = digest(cache)
            states = [None, None]
            dist.all_gather_object(states, state)
            if states[0] != states[1]:
                raise RuntimeError("recurrent ranks disagree before intervention")
            runtime = base.make_runtime(cache, directions, active, alphas)
            base.assert_nonrecurrent_unchanged(before, cache)
        before = base.snapshot_nonrecurrent(cache) if len(audit) < 2 else None
        scales = {name: 0.0 if zero else (1.0 if first else betas[name]) for name in active}
        base.apply_clamp(cache, runtime, scales)
        base.finite_active(cache, runtime)
        if before is not None:
            base.assert_nonrecurrent_unchanged(before, cache)
        if len(audit) < 2:
            states = [None, None]
            dist.all_gather_object(states, digest(cache))
            if states[0] != states[1]:
                raise RuntimeError("recurrent ranks disagree after intervention")
        audit.append({"initial": first, "zero_beta": zero, "layers": sorted(runtime)})
        return output

    return hook
