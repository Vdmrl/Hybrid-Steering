"""Ordinary additive CAA on replicated residual outputs, never on bulk prefill."""
def hook_for(vector, audit, max_calls=None):
    import torch
    calls = 0
    if max_calls is not None and max_calls <= 0: raise ValueError("max_calls must be positive")
    def hook(_module, _inputs, output):
        nonlocal calls
        hidden = output[0] if isinstance(output,tuple) else output
        if hidden.shape[1] != 1:
            return output
        calls += 1
        if max_calls is not None and calls > max_calls:
            return output
        local = hidden.to_local() if hasattr(hidden,'to_local') else hidden
        correction = vector.to(local.device,local.dtype)
        if hasattr(hidden,'placements'):
            from torch.distributed.tensor import DTensor, Replicate
            if not all(isinstance(p,Replicate) for p in hidden.placements):
                raise RuntimeError('CAA gate requires replicated full residual')
            correction = DTensor.from_local(correction,hidden.device_mesh,hidden.placements,run_check=False)
        changed = hidden.clone()
        changed[:, -1, :] += correction
        actual = changed.to_local() if hasattr(changed,'to_local') else changed
        if not torch.isfinite(actual).all(): raise FloatingPointError('nonfinite CAA hidden')
        # Audit the actual BF16 operation, not an unrounded FP32 ideal delta.
        rounded_expected = local.clone()
        rounded_expected[:, -1, :] += vector.to(local.device,local.dtype)
        if not torch.equal(actual,rounded_expected):
            raise RuntimeError('CAA output differs from dtype-rounded additive intervention')
        delta = (actual-local)[:, -1, :].float().flatten()
        expected = vector.to(local.device).float().flatten()
        if expected.norm() == 0:
            assert torch.equal(actual,local), 'zero hook changed residual'
        else:
            ratio = float(delta.norm()/expected.norm())
            cosine = float(torch.nn.functional.cosine_similarity(delta,expected,dim=0))
            # Small gains may lose coordinates when added to large BF16 values.
            # Retain distortion measurements without confusing rounding with a bug.
        audit.append({'layout':'DTensor-Replicate' if hasattr(hidden,'placements') else 'Tensor',
                      'delta_norm':float(delta.norm()), 'rounded_addition_exact':True,
                      'ideal_delta_ratio':float(delta.norm()/expected.norm()) if expected.norm()>0 else 0,
                      'ideal_delta_cosine':float(torch.nn.functional.cosine_similarity(delta,expected,dim=0)) if expected.norm()>0 else 1})
        return (changed,*output[1:]) if isinstance(output,tuple) else changed
    return hook

def self_check():
    import torch
    x=torch.randn(1,1,16); v=torch.ones(16)
    assert torch.equal(hook_for(v,[])(None,None,x),x+1)
    assert torch.equal(hook_for(v*0,[])(None,None,x),x)
    bulk=torch.randn(1,5,16)
    assert hook_for(v,[])(None,None,bulk) is bulk
    torch.manual_seed(123)
    x=(torch.randn(1,1,4096)*8).to(torch.bfloat16)
    v=torch.randn(4096); v=v/v.norm()*.5
    audit=[]
    y=hook_for(v,audit)(None,None,x)
    assert torch.equal(y,x+v.to(x.dtype))
    assert audit[0]['ideal_delta_cosine'] < .95
    assert audit[0]['rounded_addition_exact']
    assert torch.equal(hook_for(v*0,[])(None,None,x),x)

if __name__=='__main__': self_check(); print('HOOK_CPU_CHECK_PASS')
