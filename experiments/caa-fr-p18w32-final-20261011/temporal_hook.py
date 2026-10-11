"""Sum active ordinary CAA components with independent feature onset windows."""
from caa_hook import hook_for as additive_hook


def hook_for(components, audit):
    import torch
    components=[(v,limit,1) if len(row)==2 else row for row in components for v,limit in [row[:2]]]
    assert components and all(start>0 and (limit is None or limit>=start) for _,limit,start in components)
    calls = 0

    def hook(module, inputs, output):
        nonlocal calls
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] != 1:
            return output
        calls += 1
        active = [vector for vector,limit,start in components if calls>=start and (limit is None or calls<=limit)]
        if not active:
            return output
        vector = sum(active, torch.zeros_like(active[0]))
        result = additive_hook(vector, audit)(module, inputs, output)
        audit[-1].update(single_token_forward=calls, active_components=len(active))
        return result

    return hook
