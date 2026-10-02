"""GDN recurrent-state steering during prefill and greedy decoding.

The prompt is one left-padded batch and one model forward. NNsight intercepts
only the GDN state-update kernel. ``prompt_position`` counts real prompt tokens
already processed: ``0`` is the initial state, ``-1`` is the last prompt token,
and ``None`` disables prompt steering. ``scale`` is a scalar or one value per row.

``intervention="clamp"`` is the code-leak rank-1 clamp on every prompt token and
every generated token. ``scale`` multiplies ``sigma * w``. Zero scale skips the
clamp, so it stays the unsteered baseline. ``prompt_position`` and the decode
schedules are not used. ``release_position`` counts real prompt tokens like
``prompt_position``: the clamp acts after each of those tokens, then the state
evolves freely for the rest of the prompt and during decoding.
"""

import inspect
import types
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from jaxtyping import Float, Int
from nnsight import NNsight, save
from nnsight.intervention.source import OperationEnvoy
from torch import Tensor
from transformers import PreTrainedTokenizerBase
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

from .cache import gdn_layers
from .delta_rule import (
    chunk_gated_delta_rule_clamped,
    rank1_factors,
    recurrent_gated_delta_rule_clamped,
)
from .state import BatchIndex, BatchState, HeadState, Scale, add_delta, truncate_direction

PromptMode = Literal["exact", "chunk"]
PromptPosition = int | Tensor | None
Intervention = Literal["add", "clamp"]
CHUNK = 64


@dataclass
class Trace:
    """GDN outputs and states, indexed by each row's unpadded token count."""

    outputs: dict[int, dict[int, Tensor]] = field(default_factory=dict)
    states: dict[int, dict[int, Tensor]] = field(default_factory=dict)
    indices: dict[int, Tensor] = field(default_factory=dict)
    prefill_outputs: dict[int, dict[int, Tensor]] = field(default_factory=dict)
    prefill_indices: dict[int, Tensor] = field(default_factory=dict)
    logits: Tensor | None = None
    prompt_positions: Tensor | None = None


def open_gdn_trace(
    model: Any, layers: list[int]
) -> tuple[NNsight, dict[int, tuple[OperationEnvoy, OperationEnvoy]]]:
    """Unwrap GDN blocks and return the chunk and recurrent delta-rule kernels."""
    for layer in layers:
        unwrap_forward(model.model.layers[layer].linear_attn)
    traced = NNsight(model)
    operations: dict[int, tuple[OperationEnvoy, OperationEnvoy]] = {}
    for layer in layers:
        source = traced.get(f"model.layers.{layer}.linear_attn").source
        chunk, recurrent = (
            next(operation for operation in source.operations if name in operation.path)
            for name in ("chunk_gated_delta_rule", "recurrent_gated_delta_rule")
        )
        operations[layer] = chunk, recurrent
    return traced, operations


def unwrap_forward(module: torch.nn.Module) -> None:
    """Replace an Accelerate-wrapped ``forward`` with the original function."""
    wrapped = module.forward
    if isinstance(wrapped, types.MethodType) and wrapped.__name__ == "wrapped":
        function = wrapped.__func__
        forwards = [
            cell.cell_contents
            for cell in function.__closure__ or ()
            if inspect.isfunction(cell.cell_contents)
        ]
        module.forward = types.MethodType(forwards[0], module)


class Runner:
    """Apply GDN-state interventions and greedily decode a batch."""

    def __init__(
        self,
        model: Any,
        tokenizer: PreTrainedTokenizerBase | Any,
        layers: list[int],
        deltas: dict[int, HeadState] | None = None,
        normalize: bool = True,
        intervention: Intervention = "add",
    ) -> None:
        available = gdn_layers(model)
        deltas = dict(deltas or {})
        if intervention not in ("add", "clamp"):
            raise ValueError("intervention must be add or clamp")
        if any(layer not in available for layer in layers) or not set(deltas) <= set(layers):
            raise ValueError("deltas must target configured GDN layers")
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        self.deltas = deltas
        self.normalize = normalize
        self.intervention: Intervention = intervention
        self.device = next(model.parameters()).device
        self.clamp_factors = (
            {
                layer: tuple(factor.to(self.device) for factor in rank1_factors(delta))
                for layer, delta in deltas.items()
            }
            if intervention == "clamp"
            else {}
        )
        self.tokenizer.padding_side = "left"
        self.traced: NNsight | None = None
        self.operations: dict[int, tuple[OperationEnvoy, OperationEnvoy]] = {}

    @classmethod
    def from_direction(
        cls,
        model: Any,
        tokenizer: PreTrainedTokenizerBase | Any,
        direction: dict[int, HeadState],
        *,
        rank: int | None = None,
        normalize: bool = True,
        intervention: Intervention = "add",
    ) -> "Runner":
        """Build a runner whose deltas are the rank-truncated direction on the model device."""
        device = next(model.parameters()).device
        deltas = {
            layer: tensor.to(device)
            for layer, tensor in truncate_direction(direction, rank).items()
        }
        return cls(
            model,
            tokenizer,
            sorted(deltas),
            deltas,
            normalize=normalize,
            intervention=intervention,
        )

    def _inputs(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        if not texts:
            raise ValueError("texts must not be empty")
        encoded = self.tokenizer(
            texts, add_special_tokens=False, padding=True, return_tensors="pt"
        ).to(self.device)
        if not encoded.attention_mask.any(dim=1).all():
            raise ValueError("prompts must contain tokens")
        return encoded.input_ids, encoded.attention_mask

    def _enable_tracing(self) -> None:
        if self.traced is not None:
            return
        self.traced, self.operations = open_gdn_trace(self.model, self.layers)

    def _run_prefixes(
        self,
        layer: int,
        args: tuple[Tensor, ...],
        kwargs: dict[str, Any],
        columns: BatchIndex,
        scales: Scale,
    ) -> tuple[list[Tensor], tuple[tuple[Tensor, ...], dict[str, Any]]]:
        query: Float[Tensor, "batch seq heads key"]
        key: Float[Tensor, "batch seq heads key"]
        value: Float[Tensor, "batch seq heads value"]
        query, key, value = args
        state: BatchState | None = kwargs.get("initial_state")
        outputs: list[Tensor] = []
        start = 0
        kernel = torch_chunk_gated_delta_rule
        for end in sorted(
            column for column in columns.unique().tolist() if 0 <= column < query.shape[1]
        ):
            if end > start:
                output, state = kernel(
                    query[:, start:end],
                    key[:, start:end],
                    value[:, start:end],
                    **{
                        **kwargs,
                        "g": kwargs["g"][:, start:end],
                        "beta": kwargs["beta"][:, start:end],
                        "initial_state": state,
                    },
                )
                outputs.append(output)
            if state is None:
                state = torch.zeros(
                    query.shape[0],
                    query.shape[2],
                    query.shape[3],
                    value.shape[3],
                    device=query.device,
                    dtype=torch.float32,
                )
            add_delta(state, self.deltas[layer], scales * columns.eq(end), self.normalize)
            start = end
        tail = (query[:, start:], key[:, start:], value[:, start:])
        tail_kwargs = {
            **kwargs,
            "g": kwargs["g"][:, start:],
            "beta": kwargs["beta"][:, start:],
            "initial_state": state,
        }
        return outputs, (tail, tail_kwargs)

    def _replace_kernel(
        self,
        layer: int,
        operation: OperationEnvoy,
        columns: Tensor,
        scales: Tensor,
        capture: bool,
    ) -> tuple[int, Any] | None:
        prefix, (tail_args, tail_kwargs) = self._run_prefixes(
            layer, *operation.inputs, columns, scales
        )
        operation.inputs = tail_args, tail_kwargs
        tail, state = operation.output
        output = torch.cat([*prefix, tail], dim=1) if prefix else tail
        if columns.eq(output.shape[1]).any():
            add_delta(
                state, self.deltas[layer], scales * columns.eq(output.shape[1]), self.normalize
            )
        operation.output = output, state
        return (layer, save((output, state))) if capture else None

    def _released_kernel(
        self,
        layer: int,
        operation: OperationEnvoy,
        releases: Tensor,
        scales: Tensor,
        capture: bool,
    ) -> tuple[int, Any] | None:
        """Clamp each row before its release column, then run the original kernel.

        Between consecutive release columns every row is either clamped or free
        for the whole segment, so both kernels run and each row keeps its own.
        """
        (query, key, value), kwargs = operation.inputs
        u, target = self._clamp_target(layer, scales, query.shape[0])
        l2norm = bool(kwargs.get("use_qk_l2norm_in_kernel", False))
        state = kwargs.get("initial_state")
        outputs: list[Tensor] = []
        start = 0
        for end in sorted(releases.unique().tolist()):
            if end <= start:
                continue
            piece = (query[:, start:end], key[:, start:end], value[:, start:end])
            g, beta = kwargs["g"][:, start:end], kwargs["beta"][:, start:end]
            free_output, free_state = torch_chunk_gated_delta_rule(
                *piece,
                g=g,
                beta=beta,
                initial_state=state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=l2norm,
            )
            held_output, held_state = chunk_gated_delta_rule_clamped(
                *piece, g, beta, u, target, initial_state=state, use_qk_l2norm_in_kernel=l2norm
            )
            held = releases.ge(end)[:, None, None, None]
            outputs.append(torch.where(held, held_output.to(free_output.dtype), free_output))
            state = torch.where(held, held_state, free_state.float())
            start = end
        operation.inputs = (
            (query[:, start:], key[:, start:], value[:, start:]),
            {
                **kwargs,
                "g": kwargs["g"][:, start:],
                "beta": kwargs["beta"][:, start:],
                "initial_state": state,
            },
        )
        tail, state = operation.output
        output = torch.cat([*outputs, tail], dim=1)
        operation.output = output, state
        return (layer, save((output, state))) if capture else None

    @staticmethod
    def _record(
        trace: Trace, captured: dict[int, tuple[Tensor, Tensor]], mask: Tensor, logits: Tensor
    ) -> None:
        counts = mask.long().sum(-1)
        for position in counts.unique().tolist():
            indices = counts.eq(position).nonzero().flatten()
            trace.indices[position] = indices
            trace.outputs[position] = {
                layer: values[0][indices] for layer, values in captured.items()
            }
            trace.states[position] = {
                layer: values[1][indices] for layer, values in captured.items()
            }
        trace.logits = logits

    def _model_forward(
        self,
        inputs: Tensor,
        mask: Tensor,
        positions: Tensor,
        cache: Any = None,
        trace: Trace | None = None,
        columns: Tensor | None = None,
        scales: Tensor | None = None,
        releases: Tensor | None = None,
    ) -> tuple[Any, Tensor]:
        arguments = {
            "input_ids": inputs,
            "attention_mask": mask,
            "position_ids": positions,
            "past_key_values": cache,
            "use_cache": True,
            "logits_to_keep": 1,
        }
        if trace is None and columns is None and not self._clamp_active(scales):
            output = self.model(**arguments)
            return output.past_key_values, output.logits[:, -1]
        self._enable_tracing()
        saved: list[tuple[int, Any]] = []
        assert self.traced is not None
        with self.traced.trace(**arguments):
            for layer, operations in self.operations.items():
                recurrent = (
                    inputs.shape[1] == 1 and cache is not None and cache.has_previous_state(layer)
                )
                operation = operations[recurrent]
                if (
                    releases is not None
                    and self._clamp_active(scales)
                    and layer in self.clamp_factors
                ):
                    captured = self._released_kernel(
                        layer, operation, releases, scales, trace is not None
                    )
                    if captured is not None:
                        saved.append(captured)
                elif self._clamp_active(scales) and layer in self.clamp_factors:
                    args, kwargs = operation.inputs
                    clamped = self._clamped_kernel(layer, args, kwargs, scales, recurrent)
                    operation.output = clamped
                    if trace is not None:
                        saved.append((layer, save(clamped)))
                elif columns is not None and layer in self.deltas:
                    if scales is None:
                        raise ValueError("scale is required when prompt_position is set")
                    captured = self._replace_kernel(
                        layer, operation, columns, scales, trace is not None
                    )
                    if captured is not None:
                        saved.append(captured)
                elif trace is not None:
                    saved.append((layer, save(operation.output)))
            output = save(self.traced.output)
        logits = output.logits[:, -1]
        if trace is not None:
            captured = {layer: (values[0][:, -1], values[1]) for layer, values in saved}
            self._record(trace, captured, mask, logits)
        return output.past_key_values, logits

    def _scales(self, scale: float | Tensor, batch: int) -> Tensor:
        scales = torch.as_tensor(scale, device=self.device)
        if scales.ndim > 1 or (scales.ndim == 1 and scales.shape[0] not in (1, batch)):
            raise ValueError("scale must be a scalar or one value per prompt")
        if self.intervention == "clamp" and scales.numel() > 1:
            nonzero = scales != 0
            if bool(nonzero.any()) and bool((~nonzero).any()):
                raise ValueError("clamp scale must be all zero or all nonzero")
        return scales

    def _clamp_active(self, scales: Tensor | None) -> bool:
        return (
            self.intervention == "clamp"
            and scales is not None
            and bool(torch.as_tensor(scales).ne(0).any())
        )

    def _clamp_target(self, layer: int, scales: Tensor, batch: int) -> tuple[Tensor, Tensor]:
        """Value coordinate ``scale * sigma * w``. A length-1 scale is one value for the batch."""
        del batch
        u, w, sigma = self.clamp_factors[layer]
        factor = torch.as_tensor(scales, device=u.device, dtype=torch.float32)
        if factor.ndim == 0 or factor.shape[0] == 1:
            return u, (factor.reshape(()) * sigma)[:, None] * w
        return u, (factor[:, None] * sigma)[:, :, None] * w[None, :, :]

    def _clamped_kernel(
        self,
        layer: int,
        args: tuple[Tensor, ...],
        kwargs: dict[str, Any],
        scales: Tensor,
        recurrent: bool,
    ) -> tuple[Tensor, Tensor]:
        query, key, value = args[:3]
        g = kwargs["g"] if "g" in kwargs else args[3]
        beta = kwargs["beta"] if "beta" in kwargs else args[4]
        u, target = self._clamp_target(layer, scales, query.shape[0])
        function = (
            recurrent_gated_delta_rule_clamped if recurrent else chunk_gated_delta_rule_clamped
        )
        output, state = function(
            query,
            key,
            value,
            g,
            beta,
            u,
            target,
            initial_state=kwargs.get("initial_state"),
            use_qk_l2norm_in_kernel=bool(kwargs.get("use_qk_l2norm_in_kernel", False)),
        )
        return output.to(dtype=query.dtype), state

    def _add_to_cache(self, cache: Any, scales: Tensor) -> None:
        for layer, delta in self.deltas.items():
            add_delta(cache.layers[layer].recurrent_states[0], delta, scales, self.normalize)

    def _columns(
        self, position: PromptPosition, lengths: Tensor, padding: Tensor, name: str
    ) -> Tensor:
        """Padded column of a per-row token count; negative counts from the prompt end."""
        requested = torch.as_tensor(position, device=self.device)
        if requested.ndim > 1 or (requested.ndim == 1 and len(requested) not in (1, len(lengths))):
            raise ValueError(f"{name} must be scalar or have one value per prompt")
        if requested.is_floating_point():
            raise ValueError(f"{name} must contain integers")
        requested = requested.expand_as(lengths)
        columns = padding + torch.where(requested < 0, requested + lengths, requested)
        if ((columns < padding) | (columns > padding + lengths)).any():
            raise ValueError(f"{name} must resolve within every prompt")
        return columns

    @torch.inference_mode()
    def prefill(
        self,
        inputs: Tensor,
        mask: Tensor,
        prompt_position: PromptPosition,
        scale: float | Tensor,
        prompt_mode: PromptMode = "exact",
        trace: Trace | None = None,
        release_position: PromptPosition = None,
    ) -> tuple[Any, Tensor, Tensor]:
        """Process a prompt batch and optionally alter each row's GDN state once."""
        if inputs.ndim != 2 or mask.shape != inputs.shape:
            raise ValueError("inputs and mask must be two-dimensional tensors of the same shape")
        if not mask.any(dim=1).all():
            raise ValueError("every prompt must contain a token")
        if prompt_mode not in ("exact", "chunk"):
            raise ValueError("prompt_mode must be exact or chunk")
        scales = self._scales(scale, len(inputs))
        mask = mask.bool()
        lengths = mask.long().sum(-1)
        padding = inputs.shape[1] - lengths
        positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
        columns = releases = None
        if self.intervention == "add" and prompt_position is not None:
            columns = self._columns(prompt_position, lengths, padding, "prompt_position")
            if prompt_mode == "chunk":
                columns = torch.maximum(columns // CHUNK * CHUNK, padding)
            if trace is not None:
                trace.prompt_positions = columns - padding
        if self.intervention == "clamp" and release_position is not None:
            releases = self._columns(release_position, lengths, padding, "release_position")
            if (releases >= inputs.shape[1]).any():
                raise ValueError("release_position must leave at least one prompt token")
        cache, logits = self._model_forward(
            inputs,
            mask,
            positions,
            trace=trace,
            columns=columns,
            scales=scales,
            releases=releases,
        )
        if trace is not None:
            trace.prefill_outputs = {
                length: dict(values) for length, values in trace.outputs.items()
            }
            trace.prefill_indices = {
                length: indices.clone() for length, indices in trace.indices.items()
            }
        return cache, logits, mask

    @torch.inference_mode()
    def forward(
        self,
        texts: list[str],
        scale: float | Tensor = 1.0,
        prompt_position: PromptPosition = -1,
        prompt_mode: PromptMode = "exact",
        release_position: PromptPosition = None,
    ) -> Trace:
        """Capture the final prefill state through the same path as generation."""
        trace = Trace()
        inputs, mask = self._inputs(texts)
        self.prefill(inputs, mask, prompt_position, scale, prompt_mode, trace, release_position)
        return trace

    @torch.inference_mode()
    def generate(
        self,
        texts: list[str],
        scale: float | Tensor = 1.0,
        prompt_position: PromptPosition = -1,
        prompt_mode: PromptMode = "exact",
        generation_period: int | None = None,
        generation_steps: set[int] | None = None,
        max_new_tokens: int = 64,
        trace: Trace | None = None,
        release_position: PromptPosition = None,
    ) -> Int[Tensor, "batch token"]:
        """Greedily generate one batch, with optional prompt and decode interventions."""
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.intervention == "clamp" and (
            generation_period is not None or generation_steps is not None
        ):
            raise ValueError("clamp runs on every new token; generation schedules are unused")
        if generation_period is not None and generation_steps is not None:
            raise ValueError("generation_period and generation_steps are mutually exclusive")
        if generation_period is not None and generation_period < 1:
            raise ValueError("generation_period must be positive")
        if generation_steps is not None and any(step < 1 for step in generation_steps):
            raise ValueError("generation_steps must contain positive values")
        inputs, mask = self._inputs(texts)
        scales = self._scales(scale, len(inputs))
        cache, logits, mask = self.prefill(
            inputs, mask, prompt_position, scales, prompt_mode, trace, release_position
        )
        held = self._clamp_active(scales) and release_position is None
        generated = torch.full(
            (len(inputs), max_new_tokens),
            self.tokenizer.pad_token_id,
            device=self.device,
            dtype=torch.long,
        )
        finished = torch.zeros(len(inputs), dtype=torch.bool, device=self.device)
        eos = self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id
        eos_ids = torch.as_tensor(eos if isinstance(eos, list) else [eos], device=self.device)
        for step in range(max_new_tokens):
            token = torch.where(finished, self.tokenizer.pad_token_id, logits.argmax(-1))
            generated[:, step] = token
            finished |= torch.isin(token, eos_ids)
            if finished.all() or step + 1 == max_new_tokens:
                break
            inject = (generation_period is not None and (step + 1) % generation_period == 0) or (
                generation_steps is not None and step + 1 in generation_steps
            )
            if inject and self.intervention == "add":
                self._add_to_cache(cache, scales * ~finished)
            mask = torch.cat((mask, torch.ones_like(mask[:, :1])), dim=1)
            position = (mask.long().sum(-1) - 1)[:, None]
            cache, logits = self._model_forward(
                token[:, None],
                mask,
                position,
                cache,
                trace,
                scales=scales if held else None,
            )
        return generated
