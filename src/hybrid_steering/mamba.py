"""Falcon-H1 Mamba cache adapter using the shared recurrent-state operations.

Qwen's chunked GDN Runner stays authoritative for Qwen. This adapter keeps the
same direction artifact and shared additive state operation for Falcon's cache.
"""

from __future__ import annotations

from typing import Any

import torch

from .cache import apply_direction, extract_recurrent, recurrent_tensor
from .state import clamp_delta, truncate_direction


def _restore_sequence_axis(_module: Any, inputs: tuple, output: torch.Tensor) -> torch.Tensor:
    """Correct older Falcon RMSNormGated versions that squeeze one decode token."""
    if inputs[0].ndim == 3 and inputs[0].shape[1] == 1 and output.ndim == 2:
        return output.unsqueeze(1)
    return output


class MambaRunner:
    """Greedy Falcon-H1 generation; steer recurrent states before final prompt token."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        direction: dict[int, torch.Tensor] | None = None,
        *,
        rank: int | None = None,
        normalize: bool = False,
        intervention: str = "add",
        mean_target: dict[int, torch.Tensor] | None = None,
    ) -> None:
        if getattr(model.config, "model_type", None) != "falcon_h1":
            raise ValueError("MambaRunner requires Falcon-H1")
        if intervention not in {"add", "clamp"}:
            raise ValueError("intervention must be add or clamp")
        self.model, self.tokenizer = model, tokenizer
        self.direction = truncate_direction(direction or {}, rank)
        self.target = mean_target or {}
        self.normalize, self.intervention = normalize, intervention
        if intervention == "clamp" and set(self.target) != set(self.direction):
            raise ValueError("Mamba clamp needs mean target states for every direction layer")
        layers = model.model.layers
        if any(
            layer < 0 or layer >= len(layers) or not hasattr(layers[layer], "mamba")
            for layer in self.direction
        ):
            raise ValueError("direction layers must be Falcon Mamba layers")
        for layer in layers:
            mixer = getattr(layer, "mamba", None)
            if (
                mixer is not None
                and getattr(mixer, "mamba_rms_norm", False)
                and not getattr(mixer, "_hybrid_norm_hook", False)
            ):
                mixer.norm.register_forward_hook(_restore_sequence_axis)
                mixer._hybrid_norm_hook = True
        tokenizer.padding_side = "left"

    @torch.inference_mode()
    def final_states(self, texts: list[str]) -> dict[int, torch.Tensor]:
        """Collect Falcon's final Mamba states using the shared direction accumulator."""
        states: dict[int, list[torch.Tensor]] = {}
        device = next(self.model.parameters()).device
        for text in texts:
            ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(
                device
            )
            if ids.shape[1] == 0:
                raise ValueError("direction text must contain a token")
            cache = self.model(input_ids=ids, use_cache=True, logits_to_keep=1).past_key_values
            for layer, state in extract_recurrent(cache).items():
                states.setdefault(layer, []).append(state)
        if not states or any(len(rows) != len(texts) for rows in states.values()):
            raise RuntimeError("failed to collect Falcon Mamba states")
        return {layer: torch.cat(rows) for layer, rows in states.items()}

    def _steer(self, cache: Any, scale: float) -> None:
        if scale == 0 or not self.direction:
            return
        if self.intervention == "add":
            apply_direction(cache, self.direction, scale, normalize=self.normalize)
        else:
            for layer, delta in self.direction.items():
                clamp_delta(recurrent_tensor(cache, layer), delta, self.target[layer], scale)

    @torch.inference_mode()
    def generate(
        self,
        texts: list[str],
        scale: float = 1.0,
        prompt_position: int | None = -1,
        max_new_tokens: int = 64,
    ) -> torch.Tensor:
        if prompt_position not in {-1, None}:
            raise ValueError("Falcon Mamba steering supports prompt_position=-1 or None")
        if not texts or max_new_tokens < 1:
            raise ValueError("nonempty texts and positive max_new_tokens required")
        device = next(self.model.parameters()).device
        batch = self.tokenizer(
            texts, add_special_tokens=False, padding=True, return_tensors="pt"
        ).to(device)
        ids, mask = batch.input_ids, batch.attention_mask
        if (mask.sum(-1) < 2).any():
            raise ValueError("Falcon prompts need at least two tokens")
        prefix_mask = mask[:, :-1]
        positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
        first = self.model(
            input_ids=ids[:, :-1],
            attention_mask=prefix_mask,
            position_ids=positions[:, :-1],
            use_cache=True,
            logits_to_keep=1,
        )
        cache = first.past_key_values
        if prompt_position == -1:
            self._steer(cache, scale)
        out = self.model(
            input_ids=ids[:, -1:],
            attention_mask=mask,
            position_ids=positions[:, -1:],
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = out.past_key_values
        logits = out.logits[:, -1].to(device)
        pad = self.tokenizer.pad_token_id
        eos = self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id
        eos_ids = torch.as_tensor(eos if isinstance(eos, list) else [eos], device=device)
        generated = torch.full((len(texts), max_new_tokens), pad, device=device, dtype=torch.long)
        finished = torch.zeros(len(texts), dtype=torch.bool, device=device)
        for step in range(max_new_tokens):
            token = torch.where(finished, pad, logits.argmax(-1))
            generated[:, step] = token
            finished |= torch.isin(token, eos_ids)
            if finished.all() or step + 1 == max_new_tokens:
                break
            if self.intervention == "clamp" or (prompt_position is None and step == 0):
                self._steer(cache, scale)
            mask = torch.cat((mask, torch.ones_like(mask[:, :1])), dim=1)
            out = self.model(
                input_ids=token[:, None],
                attention_mask=mask,
                position_ids=(mask.long().sum(-1) - 1)[:, None],
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache, logits = out.past_key_values, out.logits[:, -1].to(device)
        return generated
