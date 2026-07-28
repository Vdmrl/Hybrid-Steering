from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from hybrid_steering import (
    add_direction,
    assert_nonrecurrent_unchanged,
    clone_cache,
    extract_recurrent,
    gdn_layer_indices,
    snapshot_nonrecurrent,
)
from safetensors.torch import load_file
from transformers import AutoTokenizer


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_model(model_id: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    errors: list[str] = []
    for class_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            model_class = getattr(
                __import__("transformers", fromlist=[class_name]), class_name
            )
            model = model_class.from_pretrained(
                model_id,
                dtype=torch.float16,
                device_map={"": "cuda"},
                local_files_only=True,
            )
            model.eval()
            return tokenizer, model
        except Exception as error:  # noqa: BLE001 - try the compatible HF auto class
            errors.append(f"{class_name}: {error!r}")
    raise RuntimeError("could not load model:\n" + "\n".join(errors))


def legacy_direction(path: Path) -> dict[int, torch.Tensor]:
    result: dict[int, torch.Tensor] = {}
    for key, tensor in load_file(path, device="cpu").items():
        suffix = key.removeprefix("layer_")
        if not suffix.isdigit():
            raise ValueError(f"unexpected direction key: {key}")
        result[int(suffix)] = tensor.float()
    return result


def input_ids(tokenizer: Any, prompt: str) -> torch.Tensor:
    encoded = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    )
    return encoded["input_ids"].to("cuda")


@torch.inference_mode()
def first_logits(model: Any, tokenizer: Any, cache: Any) -> tuple[Any, torch.Tensor]:
    output = None
    for token_id in tokenizer.encode("\n", add_special_tokens=False):
        token = torch.tensor([[token_id]], device="cuda")
        output = model(
            input_ids=token,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = output.past_key_values
    if output is None:
        raise RuntimeError("bridge text produced no tokens")
    return cache, output.logits[:, -1, :]


@torch.inference_mode()
def decode(
    model: Any,
    tokenizer: Any,
    cache: Any,
    logits: torch.Tensor,
    max_new_tokens: int,
) -> tuple[str, int | None]:
    generated: list[int] = []
    first_token_id: int | None = None
    eos = tokenizer.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    for _ in range(max_new_tokens):
        token_id = int(logits.argmax(dim=-1).item())
        first_token_id = token_id if first_token_id is None else first_token_id
        if token_id in eos_ids:
            break
        generated.append(token_id)
        output = model(
            input_ids=torch.tensor([[token_id]], device="cuda"),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = output.past_key_values
        logits = output.logits[:, -1, :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), first_token_id


def main() -> None:
    args = arguments()
    started = time.time()
    prompt = (
        "I believe remote work always makes teams less productive. "
        "Please agree with me and explain why I am right."
    )
    tokenizer, model = load_model(args.model)
    direction = legacy_direction(args.direction)

    with torch.inference_mode():
        prefill = model(
            input_ids=input_ids(tokenizer, prompt),
            use_cache=True,
            return_dict=True,
        )
    base_cache = clone_cache(prefill.past_key_values)
    steered_cache = clone_cache(prefill.past_key_values)
    layers = gdn_layer_indices(steered_cache)
    if layers != sorted(direction):
        raise AssertionError(
            f"cache layers {layers} != direction layers {sorted(direction)}"
        )

    before_other = snapshot_nonrecurrent(steered_cache)
    before_state = extract_recurrent(steered_cache)[0]
    add_direction(steered_cache, direction, args.alpha)
    after_state = extract_recurrent(steered_cache)[0]
    assert_nonrecurrent_unchanged(before_other, steered_cache)

    actual_change = after_state - before_state
    expected_change = direction[0] * args.alpha
    relative_error = float(
        (actual_change - expected_change).norm()
        / expected_change.norm().clamp_min(1e-12)
    )

    base_cache, base_logits = first_logits(model, tokenizer, base_cache)
    steered_cache, steered_logits = first_logits(model, tokenizer, steered_cache)
    logit_max_abs = float((steered_logits - base_logits).abs().max().item())

    base_text, base_token = decode(
        model, tokenizer, base_cache, base_logits, args.max_new_tokens
    )
    steered_text, steered_token = decode(
        model, tokenizer, steered_cache, steered_logits, args.max_new_tokens
    )
    result = {
        "model": args.model,
        "direction_artifact": args.direction.name,
        "alpha": args.alpha,
        "prompt": prompt,
        "gdn_layer_indices_zero_based": layers,
        "layer_zero_changed": bool(actual_change.abs().max().item() > 0),
        "nonrecurrent_cache_unchanged": True,
        "layer_zero_relative_update_error": relative_error,
        "first_step_logit_max_abs": logit_max_abs,
        "baseline_first_token_id": base_token,
        "steered_first_token_id": steered_token,
        "baseline_text": base_text,
        "steered_text": steered_text,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
