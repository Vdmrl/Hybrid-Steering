"""One-layer attention-only model matching Elhage et al. (2021)."""

from dataclasses import dataclass
import math
import warnings

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from gdn import GatedDeltaNet


@dataclass(frozen=True)
class ModelSpec:
    d_model: int = 768
    n_heads: int = 12
    d_head: int = 64
    n_ctx: int = 2048
    seed: int = 42
    layer_types: tuple[str, ...] = ("attn",)
    gdn_backend: str = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_types", tuple(self.layer_types))
        if self.d_model != self.n_heads * self.d_head:
            raise ValueError("d_model must equal n_heads * d_head")
        if self.d_model % 2:
            raise ValueError("d_model must be even for sinusoidal positions")
        if not self.layer_types or any(layer not in {"attn", "gdn"} for layer in self.layer_types):
            raise ValueError("layer_types must contain one or more 'attn' or 'gdn' layers")
        if self.gdn_backend not in {"auto", "fla", "reference"}:
            raise ValueError("gdn_backend must be 'auto', 'fla', or 'reference'")


def sinusoidal_positions(length: int, width: int) -> torch.Tensor:
    """Return the fixed sinusoidal position matrix used by the paper."""
    position = torch.arange(length, dtype=torch.float32)[:, None]
    frequency = torch.exp(
        torch.arange(0, width, 2, dtype=torch.float32)
        * (-math.log(10_000) / width)
    )
    result = torch.empty(length, width)
    result[:, 0::2] = torch.sin(position * frequency)
    result[:, 1::2] = torch.cos(position * frequency)
    return result


def load_tokenizer(name: str, revision: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    tokenizer.model_max_length = 10**30
    if tokenizer.eos_token_id is None:
        raise ValueError("the tokenizer must define an EOS token")
    return tokenizer


class GDNBlock(nn.Module):
    """TransformerLens-compatible residual block around a hookable GDN mixer."""

    def __init__(self, config, spec: ModelSpec, block_index: int):
        super().__init__()
        from transformer_lens.components import LayerNorm
        from transformer_lens.hook_points import HookPoint

        self.ln1 = LayerNorm(config)
        self.gdn = GatedDeltaNet(spec.d_model, spec.n_heads, spec.d_head, backend=spec.gdn_backend)
        self.hook_resid_pre = HookPoint()
        self.hook_gdn_out = HookPoint()
        self.hook_resid_post = HookPoint()
        self._reset_parameters(config.initializer_range)

    def _reset_parameters(self, std: float) -> None:
        for module in self.gdn.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, resid_pre, shortformer_pos_embed=None, past_kv_cache_entry=None, attention_mask=None
    ):
        resid_pre = self.hook_resid_pre(resid_pre)
        gdn_out = self.hook_gdn_out(self.gdn(self.ln1(resid_pre), attention_mask))
        return self.hook_resid_post(resid_pre + gdn_out)


def build_model(spec: ModelSpec, tokenizer):
    """Build a causal Shortformer-style TransformerLens model.

    TransformerLens' ``shortformer`` mode adds positions only to Q and K. We
    replace its learned position matrix with the fixed sinusoidal matrix from
    the paper and freeze it. Attention and unembedding biases are zeroed and
    frozen so the learned paths match the paper's bias-free equations.
    """
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    vocabulary = tokenizer if isinstance(tokenizer, int) else len(tokenizer)
    config = HookedTransformerConfig(
        n_layers=len(spec.layer_types),
        n_heads=spec.n_heads,
        d_model=spec.d_model,
        d_head=spec.d_head,
        d_vocab=vocabulary,
        n_ctx=spec.n_ctx,
        attn_only=True,
        attention_dir="causal",
        normalization_type="LN",
        positional_embedding_type="shortformer",
        tie_word_embeddings=False,
        default_prepend_bos=False,
        tokenizer_prepends_bos=False,
        seed=spec.seed,
        device="cpu",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model = HookedTransformer(
            config,
            tokenizer=None if isinstance(tokenizer, int) else tokenizer,
            move_to_device=False,
        )

    with torch.no_grad():
        model.pos_embed.W_pos.copy_(
            sinusoidal_positions(spec.n_ctx, spec.d_model)
        )
        for name, parameter in model.named_parameters():
            if name.endswith(("b_Q", "b_K", "b_V", "b_O", "b_U")):
                parameter.zero_()
                parameter.requires_grad_(False)
    model.pos_embed.W_pos.requires_grad_(False)
    for index, layer_type in enumerate(spec.layer_types):
        if layer_type == "gdn":
            model.blocks[index] = GDNBlock(config, spec, index)
    model.layer_types = spec.layer_types
    model.setup()
    return model


def load_checkpoint(run_dir, checkpoint=None, device="cpu"):
    """Restore a model and tokenizer from a trainer-tools run directory."""
    from omegaconf import OmegaConf

    run_dir = run_dir.resolve()
    config = OmegaConf.to_container(
        OmegaConf.load(run_dir / "checkpoints" / "config.yaml"), resolve=True
    )
    tokenizer = load_tokenizer(config["data"]["tokenizer"])
    model = build_model(ModelSpec(**config["model"]), tokenizer)
    path = checkpoint or run_dir / "checkpoints" / "model_final.pt"
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, tokenizer, config, path
