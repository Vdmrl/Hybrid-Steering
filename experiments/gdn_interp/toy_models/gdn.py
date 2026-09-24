"""Hookable reference implementation of the Gated DeltaNet (GDN-v1) mixer."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_lens.hook_points import HookPoint


class GatedDeltaNet(nn.Module):
    """The recurrent GDN-v1 delta rule used by Qwen3-Next.

    CUDA uses FLA's chunked Triton kernel for training; CPU and the explicit
    ``reference`` backend use the readable recurrence needed for debugging.
    Its state update is
    ``S <- exp(g) S + k outer (beta * (v - k S))``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        conv_size: int = 4,
        backend: str = "auto",
    ):
        super().__init__()
        if d_model != n_heads * d_head:
            raise ValueError("GDN requires d_model == n_heads * d_head")
        if backend not in {"auto", "fla", "reference"}:
            raise ValueError("GDN backend must be 'auto', 'fla', or 'reference'")
        self.n_heads, self.d_head, self.conv_size, self.backend = n_heads, d_head, conv_size, backend
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.beta_proj = nn.Linear(d_model, n_heads, bias=True)
        self.decay_proj = nn.Linear(d_model, n_heads, bias=False)
        self.A_log = nn.Parameter(torch.empty(n_heads).uniform_(0, 16).log())
        dt = torch.exp(torch.empty(n_heads).uniform_(math.log(0.001), math.log(0.1)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.q_conv = nn.Conv1d(d_model, d_model, conv_size, groups=d_model, bias=False)
        self.k_conv = nn.Conv1d(d_model, d_model, conv_size, groups=d_model, bias=False)
        self.v_conv = nn.Conv1d(d_model, d_model, conv_size, groups=d_model, bias=False)
        self.rms_weight = nn.Parameter(torch.ones(n_heads, d_head))
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_v = HookPoint()
        self.hook_beta = HookPoint()
        self.hook_decay = HookPoint()
        self.hook_state = HookPoint()
        self.hook_out = HookPoint()

    def _conv(self, x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        # The official GDN uses causal depthwise short convolutions followed by SiLU.
        return F.silu(conv(F.pad(x.transpose(1, 2), (self.conv_size - 1, 0))).transpose(1, 2))

    def _fla(self, q, k, v, beta, decay):
        # TransformerLens' LayerNorm path can retain fp32 under autocast, while
        # FLA requires q/k/v in a low-precision dtype. The cast is differentiable.
        dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else torch.bfloat16
        q, k, v = (tensor.to(dtype) for tensor in (q, k, v))
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        except ImportError as error:
            raise RuntimeError(
                "CUDA GDN training requires flash-linear-attention; run with --extra kernels"
            ) from error
        output, state = chunk_gated_delta_rule(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            decay.transpose(1, 2).contiguous(),
            beta.transpose(1, 2).contiguous(),
            output_final_state=True,
        )
        return output.transpose(1, 2), state

    def _reference(self, q, k, v, beta, decay):
        batch, _, positions, _ = q.shape
        state = q.new_zeros(batch, self.n_heads, self.d_head, self.d_head)
        outputs = []
        for position in range(positions):
            state = state * decay[:, :, position].exp()[..., None, None]
            value = v[:, :, position] - torch.einsum("bhij,bhi->bhj", state, k[:, :, position])
            state = state + torch.einsum("bhi,bhj->bhij", k[:, :, position], beta[:, :, position, None] * value)
            outputs.append(torch.einsum("bhi,bhij->bhj", q[:, :, position], state) / math.sqrt(self.d_head))
        return torch.stack(outputs, dim=2), state

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        batch, positions, _ = x.shape
        q = self.hook_q(self._conv(self.q_proj(x), self.q_conv).view(batch, positions, self.n_heads, self.d_head).transpose(1, 2))
        k = self.hook_k(self._conv(self.k_proj(x), self.k_conv).view(batch, positions, self.n_heads, self.d_head).transpose(1, 2))
        v = self.hook_v(self._conv(self.v_proj(x), self.v_conv).view(batch, positions, self.n_heads, self.d_head).transpose(1, 2))
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        beta = self.hook_beta(self.beta_proj(x).sigmoid().transpose(1, 2))
        decay = self.hook_decay(
            (-self.A_log.exp() * F.softplus(self.decay_proj(x) + self.dt_bias)).transpose(1, 2)
        )
        if self.backend == "reference":
            output, state = self._reference(q, k, v, beta, decay)
        elif not x.is_cuda:
            if self.backend == "fla":
                raise RuntimeError("the FLA GDN backend requires CUDA")
            output, state = self._reference(q, k, v, beta, decay)
        else:
            output, state = self._fla(q, k, v, beta, decay)
        self.hook_state(state)
        output = output * torch.rsqrt(output.square().mean(dim=-1, keepdim=True) + 1e-5)
        output = output * self.rms_weight[None, :, None, :]
        output = output * F.silu(self.g_proj(x).view(batch, positions, self.n_heads, self.d_head).transpose(1, 2))
        return self.hook_out(self.o_proj(output.transpose(1, 2).reshape(batch, positions, -1)))
