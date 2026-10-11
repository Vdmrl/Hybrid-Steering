"""Existing native TP2 loading/generation path for frozen follow-up tasks."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from datetime import timedelta


def load_colleague(path):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != '54788c5c77ecbd99baa46c0de6932441e4b74e723cae6d9b308099f5c853730e':
        raise ValueError('colleague script changed')
    spec = importlib.util.spec_from_file_location('colleague_tp',path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def generate(model, tokenizer, prompt, limit, device, original):
    # Preserve the colleague's TP-safe split-prefill loop; match our prompt/EOS.
    import torch
    code = original.generate.__code__
    import inspect
    source = inspect.getsource(original.generate)
    source = source.replace('[{"role": "user", "content": prompt}]',
        '[{"role": "system", "content": "Answer the user\'s request directly and accurately. Preserve supplied facts and do not invent details. Use enough words to complete the task, without padding."}, {"role": "user", "content": prompt}]')
    source = source.replace('batch = tokenizer(text, add_special_tokens=False, return_tensors="pt")',
        'batch = tokenizer(text, add_special_tokens=False, return_tensors="pt")\n    bridge = tokenizer.encode("\\n", add_special_tokens=False)\n    batch["input_ids"] = torch.cat((batch.input_ids, torch.tensor([bridge], dtype=batch.input_ids.dtype)), dim=1)\n    batch["attention_mask"] = torch.ones_like(batch.input_ids)')
    source = source.replace('generated = []', 'eos.add(int(tokenizer.eos_token_id))\n    generated = []')
    source = source.replace('token = out.logits[:, -1].argmax(-1)',
        'if not torch.isfinite(out.logits).all():\n                raise FloatingPointError("nonfinite TP logits")\n            token = out.logits[:, -1].argmax(-1)')
    namespace = {'position_ids': original.position_ids}
    exec(compile(source,'adapted_colleague_generate','exec'),namespace)
    return namespace['generate'](model,tokenizer,prompt,limit,device)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--colleague',type=Path,required=True)
    parser.add_argument('--prompts',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--worker',type=int,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runtime-mode', choices=['single-offload','native-tp2'], required=True)
    args=parser.parse_args()
    original=load_colleague(args.colleague)
    import torch
    import torch.distributed as dist
    from transformers import AutoConfig,AutoTokenizer,AutoModelForImageTextToText
    from transformers.distributed.configuration_utils import DistributedConfig
    offload=args.runtime_mode=='single-offload'
    if not offload and (int(os.environ.get('WORLD_SIZE','0'))!=2 or torch.cuda.device_count()!=2):
        raise ValueError('exactly two torchrun ranks and GPUs required')
    rank=int(os.environ.get('LOCAL_RANK','0')); torch.cuda.set_device(rank)
    torch.manual_seed(20260911)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    device=torch.device('cuda',rank)
    if offload:
        if torch.cuda.device_count()!=1: raise ValueError('offload requires exactly one GPU')
    else:
        dist.init_process_group('nccl',timeout=timedelta(minutes=10),device_id=device)
    model_path='/home/afendulevki/models/Qwen3.5-9B'
    try:
        if offload:
            from cpu_resource import available_cpu_bytes
            available=available_cpu_bytes()
            if available < 24*1024**3: raise RuntimeError('insufficient free CPU RAM for BF16 offload')
            model=AutoModelForImageTextToText.from_pretrained(model_path,local_files_only=True,
                dtype=torch.bfloat16,attn_implementation='sdpa',device_map='auto',
                max_memory={0:'10GiB','cpu':min(64*1024**3,available-8*1024**3)}).eval()
            if any(str(v)=='disk' for v in model.hf_device_map.values()):
                raise RuntimeError('disk offload is not authorized')
        else:
            config=AutoConfig.from_pretrained(model_path,local_files_only=True)
            plan=original.tp_plan(config)
            model=AutoModelForImageTextToText.from_pretrained(model_path,local_files_only=True,
                dtype=torch.bfloat16,attn_implementation='sdpa',
                distributed_config=DistributedConfig(tp_size=2,tp_plan=plan)).eval()
            for name,tensor in list(model.named_parameters())+list(model.named_buffers()):
                local=tensor.to_local() if hasattr(tensor,'to_local') else tensor
                if local.device.type!='cuda': raise ValueError(f'non-CUDA tensor {name}')
        tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        from search import execute
        execute(model,tokenizer,device,original,args,generate)
    finally:
        if dist.is_initialized(): dist.destroy_process_group()


if __name__=='__main__': main()
