"""Load the tiny local model or a Hugging Face checkpoint."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig


class TinyTokenizer:
    """Character tokenizer for the random tiny Qwen used by tests and smoke runs."""

    def __init__(self, vocab_size: int = 64) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.padding_side = "left"

    def encode(self, text: str) -> list[int]:
        return [2 + (ord(character) % (self.vocab_size - 2)) for character in text] or [2]

    def decode(
        self,
        token_ids: Any,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del clean_up_tokenization_spaces
        if isinstance(token_ids, torch.Tensor):
            values = token_ids.detach().cpu().tolist()
        else:
            values = list(token_ids)
        characters = []
        for token in values:
            if skip_special_tokens and token in {self.pad_token_id, self.eos_token_id}:
                continue
            characters.append(chr((int(token) - 2) % 26 + ord("a")))
        return "".join(characters)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
    ) -> str:
        del tokenize, enable_thinking
        parts = [f"{message['role']}: {message['content']}" for message in messages]
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)

    def __call__(
        self,
        texts: str | list[str],
        add_special_tokens: bool = False,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
    ) -> Any:
        del add_special_tokens
        single = isinstance(texts, str)
        rows = [texts] if single else list(texts)
        encoded = []
        for text in rows:
            ids = self.encode(text)
            if truncation and max_length is not None:
                ids = ids[:max_length]
            encoded.append(ids)
        if return_tensors != "pt":
            input_ids = encoded[0] if single else encoded
            return SimpleNamespace(input_ids=input_ids)
        width = max(len(ids) for ids in encoded)
        input_ids = torch.zeros(len(encoded), width, dtype=torch.long)
        attention_mask = torch.zeros(len(encoded), width, dtype=torch.long)
        for row, ids in enumerate(encoded):
            if self.padding_side == "left":
                input_ids[row, width - len(ids) :] = torch.tensor(ids)
                attention_mask[row, width - len(ids) :] = 1
            else:
                input_ids[row, : len(ids)] = torch.tensor(ids)
                attention_mask[row, : len(ids)] = 1
        return _Batch({"input_ids": input_ids, "attention_mask": attention_mask})


class _Batch(dict):
    """Tokenizer output that supports both attribute access and ``model(**batch)``."""

    def __getattr__(self, name: str) -> Tensor:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def to(self, device: torch.device | str) -> _Batch:
        return _Batch({key: value.to(device) for key, value in self.items()})


def build_tiny() -> tuple[Qwen3_5ForCausalLM, TinyTokenizer]:
    """Random two-layer Qwen3.5 with one GDN block. No checkpoint download."""
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [2, 3, 3],
        },
    )
    model = Qwen3_5ForCausalLM(config).eval()
    tokenizer = TinyTokenizer(config.vocab_size)
    model.generation_config.eos_token_id = None
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def load_runtime(model_name: str, *, dtype: torch.dtype = torch.bfloat16) -> tuple[Any, Any]:
    """Load ``tiny`` locally, or a Hugging Face id onto CUDA."""
    if model_name == "tiny":
        return build_tiny()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map="cuda").eval()
    return model, tokenizer


def token_prefixes(
    tokenizer: Any, text: str, lengths: list[int] | tuple[int, ...]
) -> dict[int, str]:
    """Cut one filler into prefixes of the requested token lengths."""
    tokens = tokenizer(text, add_special_tokens=False).input_ids
    if lengths and len(tokens) < max(lengths):
        raise ValueError(f"filler has {len(tokens)} tokens, need {max(lengths)}")
    return {
        length: tokenizer.decode(tokens[:length], clean_up_tokenization_spaces=False)
        for length in lengths
    }


def import_path(path: str | Path) -> Any:
    """Load a module from a file path. Used by experiment scripts that are not a package."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: str | Path) -> list[dict]:
    """Read one JSON object per line."""
    path = Path(path)
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    """Write one JSON object per line, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def batched(items: list, size: int):
    """Yield successive slices of ``size``."""
    if size < 1:
        raise ValueError("size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def collect_steered_rows(
    runner: Any,
    tokenizer: Any,
    examples: list[dict[str, str]],
    prefixes: dict[int, str],
    scales: Iterable[float],
    *,
    batch_size: int,
    token_budget: int,
    max_new_tokens: int,
    build_row: Callable[[int, dict[str, str], str, float, str], dict],
) -> list[dict]:
    """Baseline, then each scale, for every filler prefix.

    ``build_row`` receives the prefix length, example, baseline text, scale,
    and steered response.
    """
    rows: list[dict] = []
    scale_list = list(scales)
    for length, prefix in prefixes.items():
        width = min(batch_size, max(1, token_budget // max(length, 1)))
        for batch in batched(examples, width):
            questions = [item["question"] for item in batch]
            texts = chat_prompts(tokenizer, questions, prefix)
            baseline_tokens = runner.generate(
                texts, prompt_position=None, max_new_tokens=max_new_tokens
            )
            baseline_text = [
                tokenizer.decode(tokens, skip_special_tokens=True) for tokens in baseline_tokens
            ]
            for scale in scale_list:
                steered_tokens = runner.generate(
                    texts, scale=scale, prompt_position=0, max_new_tokens=max_new_tokens
                )
                for example, base, steered in zip(
                    batch, baseline_text, steered_tokens, strict=True
                ):
                    response = tokenizer.decode(steered, skip_special_tokens=True)
                    rows.append(build_row(length, example, base, scale, response))
        print(f"prefix {length}: {len(rows)} rows", flush=True)
    return rows


def chat_prompts(tokenizer: Any, questions: list[str], prefix: str = "") -> list[str]:
    """Render user prompts, optionally after a filler prefix."""
    leading = [{"role": "user", "content": prefix}] if prefix else []
    return [
        tokenizer.apply_chat_template(
            [*leading, {"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for question in questions
    ]
