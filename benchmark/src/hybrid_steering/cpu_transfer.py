"""Opt-in host-staged Accelerate transfers for machines with broken CUDA P2P.

Weights stay on their assigned GPUs. Only cross-GPU tensors pass through CPU.
Blocking copies are intentional: correctness before asynchronous optimization.
"""
import torch

transfer_count = 0


def enable() -> None:
    import accelerate.hooks as hooks
    import accelerate.utils.operations as operations

    if getattr(operations.send_to_device, '_host_staged', False):
        return
    original = operations.send_to_device

    def send(tensor, device, non_blocking=False, skip_keys=None):
        global transfer_count
        if isinstance(tensor, torch.Tensor) and tensor.device.type == 'cuda':
            target = torch.device(f'cuda:{device}' if isinstance(device, int) else device)
            if target.type == 'cuda':
                index = target.index if target.index is not None else torch.cuda.current_device()
                if tensor.device.index != index:
                    transfer_count += 1
                    return tensor.to('cpu', non_blocking=False).to(target, non_blocking=False)
        # Original recursion calls operations.send_to_device, preserving nested
        # container types and skip_keys (especially the recurrent/KV cache).
        return original(tensor, device, non_blocking=non_blocking, skip_keys=skip_keys)

    send._host_staged = True
    operations.send_to_device = send
    hooks.send_to_device = send
