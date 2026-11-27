"""Native Qwen BF16 loading and split-prefill loop.

Ports the frozen CAA native_run.py and its colleague.py/run.py decoding pair.
"""

import time
from contextlib import ExitStack

import torch
from build_job_configs import MODEL, REVISION
from interventions import LAYERS, RecurrentWrite, Step, caa_hook


def decoder_blocks(model):
    return model.model.language_model.layers


def load_model(path):
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Native CUDA BF16 support is required")
    fused = (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
    )
    if any(getattr(qwen, name) is not None for name in fused):
        raise RuntimeError(
            "Use the frozen pure-Torch recurrence environment, without FLA or causal-conv1d"
        )
    torch.manual_seed(20260911)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    model = AutoModelForImageTextToText.from_pretrained(
        path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    ).eval()
    tensors = [*model.parameters(), *model.buffers()]
    if any(t.device != device for t in tensors) or any(
        p.dtype != torch.bfloat16 for p in model.parameters()
    ):
        raise RuntimeError("Expected one CUDA device, BF16 model parameters and no offload")
    text = model.config.get_text_config()
    recurrent = tuple(i for i, kind in enumerate(text.layer_types) if kind == "linear_attention")
    if (
        text.hidden_size != 4096
        or recurrent != LAYERS
        or len(decoder_blocks(model)) != 32
        or (
            text.linear_num_value_heads,
            text.linear_key_head_dim,
            text.linear_value_head_dim,
        )
        != (32, 128, 128)
        or model.config.model_type != "qwen3_5"
    ):
        raise RuntimeError("Expected the Qwen3.5-9B 32-block layout")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return model, tokenizer, device


def model_provenance(path, *, inspect_device=True):
    import transformers
    from build_job_configs import sha

    path = path.resolve()
    cfg = path / "config.json"
    if not cfg.is_file():
        raise ValueError("--model must point to a local, complete pinned model snapshot")
    weights = sorted(path.glob("model*.safetensors"))
    if not weights:
        raise ValueError("No safetensors checkpoint in --model")
    snapshot_revision = path.name if path.parent.name == "snapshots" else None
    if snapshot_revision and snapshot_revision != REVISION:
        raise ValueError(f"Local HF snapshot is not revision {REVISION}")
    metadata = {
        p.name: sha(p)
        for p in [
            cfg,
            *path.glob("*.safetensors.index.json"),
            *path.glob("tokenizer.json"),
            *path.glob("tokenizer_config.json"),
            *path.glob("chat_template.jinja"),
            *path.glob("generation_config.json"),
        ]
    }
    return {
        "model": MODEL,
        "declared_revision": REVISION,
        "observed_snapshot_revision": snapshot_revision,
        "metadata_sha256": metadata,
        "weight_sizes": {p.name: p.stat().st_size for p in weights},
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name()
        if inspect_device and torch.cuda.is_available()
        else None,
        "attention": "sdpa",
        "recurrent_kernels": "pure-torch",
    }


def position_ids(mask):
    return (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 0)


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    device,
    prompt,
    system,
    limit,
    *,
    caa=None,
    recurrent: RecurrentWrite | None = None,
):
    step = Step()
    caa_calls = {}
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tokenizer(rendered, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    bridge = tokenizer.encode("\n", add_special_tokens=False)
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 2 or len(bridge) != 1:
        raise ValueError("Expected one bulk-prefill sequence and a single-token newline bridge")
    ids = torch.cat((ids, torch.tensor([bridge], device=device, dtype=ids.dtype)), dim=1)
    mask = torch.ones_like(ids)
    ends = model.generation_config.eos_token_id
    eos = {ends} if isinstance(ends, int) else set(ends or [])
    if tokenizer.eos_token_id is not None:
        eos.add(tokenizer.eos_token_id)
    if not eos:
        raise ValueError("EOS token IDs were not configured")

    def forward(pieces, attn_mask, phase, cache=None):
        step.phase = phase
        if phase != "bulk":
            step.token_call += 1
        kwargs = {
            "input_ids": pieces,
            "attention_mask": attn_mask,
            "position_ids": position_ids(attn_mask)[:, -pieces.shape[1] :],
            "use_cache": True,
            "logits_to_keep": 1,
        }
        if cache is not None:
            kwargs["past_key_values"] = cache
        out = model(**kwargs)
        if not bool(torch.isfinite(out.logits).all()):
            raise FloatingPointError(f"nonfinite logits after {phase}")
        current = out.past_key_values
        if current is None or (cache is not None and current is not cache):
            raise RuntimeError(f"Expected one fresh native cache per answer, then reuse on {phase}")
        if recurrent:
            # Preserve the original model-forward hook timing. The first write
            # changes the completed bulk cache before the bridge reads it.
            recurrent.after_forward(current, phase)
        return out

    started = time.monotonic()
    generated = []
    finish = "length"
    with ExitStack() as stack:
        if caa:
            blocks = decoder_blocks(model)
            caa_layers = caa[1]
            for layer in caa_layers:
                handle = blocks[layer].register_forward_hook(caa_hook(step, layer, caa, caa_calls))
                stack.callback(handle.remove)
        bulk = forward(ids[:, :-1], mask[:, :-1], "bulk")
        out = forward(ids[:, -1:], mask, "bridge", bulk.past_key_values)
        for position in range(limit):
            token = out.logits[:, -1].argmax(-1)
            token_id = token.item()
            if token_id in eos:
                finish = "eos"
                break
            generated.append(token_id)
            if position + 1 == limit:
                break
            mask = torch.cat((mask, torch.ones_like(mask[:, :1])), dim=1)
            out = forward(token[:, None], mask, "decode", out.past_key_values)
    if caa:
        expected_calls = {}
        for start, end in zip(caa, [*list(caa)[1:], step.token_call + 1], strict=True):
            count = max(0, min(end, step.token_call + 1) - start)
            if count:
                for layer in caa[start]:
                    expected_calls[layer] = expected_calls.get(layer, 0) + count
        if caa_calls != expected_calls:
            raise RuntimeError("CAA schedule did not run at every expected layer/step")
    if recurrent:
        expected = 1 if recurrent.mode == "no_clamp" else 1 + step.token_call
        if recurrent.writes != expected:
            raise RuntimeError("Unexpected recurrent write count")
    return {
        "token_ids": generated,
        "finish_reason": finish,
        "response": tokenizer.decode(generated, skip_special_tokens=True).strip(),
        "seconds": time.monotonic() - started,
        "audit": {
            "finite_logits": True,
            "bulk_forward_count": 1,
            "bridge_forward_count": 1,
            "decode_forward_count": step.token_call - 1,
            "caa_layer_calls": caa_calls,
            "recurrent_writes": recurrent.writes if recurrent else 0,
        },
        "audit_first": recurrent.audit_first if recurrent else [],
    }
