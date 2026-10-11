"""Standalone two-GPU Qwen3.5-9B TP smoke test.

Run with: CUDA_VISIBLE_DEVICES=0,1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0 \
    torchrun --standalone --nproc-per-node=2 tp_qwen35_9b.py

Tested TP layout: attention/MLP sharded, recurrent linear-attention replicated.
This demonstrates baseline generation, not activation steering or benchmarking.
"""

import argparse
import hashlib
import json
import os
import time
from datetime import timedelta


MODEL = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


def tp_plan(model_config: object) -> dict[str, str]:
    """Reuse Qwen's TP mapping except GDN, whose state must stay whole."""
    base = model_config.text_config.base_model_tp_plan
    if not isinstance(base, dict) or not base:
        raise ValueError("Model has no supported text TP plan")
    plan = {"model.language_model." + key: value for key, value in base.items()
            if ".linear_attn." not in key}
    if not plan:
        raise ValueError("TP plan is empty after excluding recurrent modules")
    return plan


def position_ids(mask):
    return (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 0)


def generate(model, tokenizer, prompt: str, max_new_tokens: int, device):
    """Greedy no-thinking decode matching the project's split-prefill baseline."""
    import torch

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    batch = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    ids = batch.input_ids.to(device)
    mask = batch.attention_mask.to(device)
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    generated = []

    with torch.inference_mode():
        first = model(input_ids=ids[:, :-1], attention_mask=mask[:, :-1],
                      position_ids=position_ids(mask[:, :-1]), use_cache=True,
                      logits_to_keep=1)
        out = model(input_ids=ids[:, -1:], attention_mask=mask,
                    position_ids=position_ids(mask)[:, -1:],
                    past_key_values=first.past_key_values, use_cache=True,
                    logits_to_keep=1)
        for step in range(max_new_tokens):
            token = out.logits[:, -1].argmax(-1)
            token_id = token.item()
            if token_id in eos:
                break
            generated.append(token_id)
            if step + 1 == max_new_tokens:
                break
            mask = torch.cat((mask, torch.ones((1, 1), device=device, dtype=mask.dtype)), dim=1)
            out = model(input_ids=token[:, None], attention_mask=mask,
                        position_ids=position_ids(mask)[:, -1:],
                        past_key_values=out.past_key_values, use_cache=True,
                        logits_to_keep=1)
    return generated, tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="Explain why public parks matter in a city.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--offline", action="store_true", help="Use only locally cached model files")
    args = parser.parse_args()
    if not args.prompt or args.max_new_tokens < 1:
        parser.error("Prompt must be nonempty and max-new-tokens positive")
    if "RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", "0")) != 2:
        parser.error("Launch with torchrun --standalone --nproc-per-node=2")

    import torch
    import torch.distributed as dist
    import transformers
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer, DistributedConfig

    if torch.cuda.device_count() != 2:
        parser.error("Expose exactly two CUDA GPUs with CUDA_VISIBLE_DEVICES")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10), device_id=device)
    try:
        config = AutoConfig.from_pretrained(MODEL, revision=REVISION,
                                            local_files_only=args.offline)
        plan = tp_plan(config)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL, revision=REVISION, dtype=torch.bfloat16,
            local_files_only=args.offline,
            distributed_config=DistributedConfig(tp_size=2, tp_plan=plan)).eval()
        if model.config._commit_hash != REVISION:
            raise ValueError("Loaded model revision differs from pinned revision")
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION,
                                                   local_files_only=args.offline)
        dist.barrier()
        torch.cuda.synchronize()
        started = time.perf_counter()
        tokens, answer = generate(model, tokenizer, args.prompt, args.max_new_tokens, device)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        local = {"tokens": tokens, "seconds": elapsed,
                 "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20)}
        gathered = [None, None]
        dist.all_gather_object(gathered, local)
        if gathered[0]["tokens"] != gathered[1]["tokens"]:
            raise RuntimeError("TP ranks generated different token sequences")
        if dist.get_rank() == 0:
            seconds = max(item["seconds"] for item in gathered)
            print(json.dumps({"model": MODEL, "revision": REVISION,
                              "tp_size": 2, "gdn": "replicated",
                              "tp_plan_sha256": hashlib.sha256(
                                  json.dumps(plan, sort_keys=True).encode()).hexdigest(),
                              "torch": torch.__version__, "transformers": transformers.__version__,
                              "prompt": args.prompt, "answer": answer,
                              "generated_tokens": len(tokens), "seconds": round(seconds, 3),
                              "tokens_per_second": round(len(tokens) / seconds, 2),
                              "peak_reserved_mib": [item["peak_reserved_mib"] for item in gathered]},
                             ensure_ascii=False, indent=2), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
