import inspect
import types
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import torch
from jaxtyping import Float
from nnsight import NNsight, save
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.source import OperationEnvoy
from torch import Tensor
from transformers import PreTrainedTokenizer

from .steering import add_delta

PromptSteerMode = Literal["exact", "chunk"]
PromptPosition = int | Tensor | None


@dataclass
class GDNTrace:
    """GDN outputs and states, indexed by each row's unpadded token count."""

    outputs: dict[int, dict[int, Tensor]] = field(default_factory=dict)
    states: dict[int, dict[int, Tensor]] = field(default_factory=dict)
    indices: dict[int, Tensor] = field(default_factory=dict)
    prefill_outputs: dict[int, dict[int, Tensor]] = field(default_factory=dict)
    prefill_indices: dict[int, Tensor] = field(default_factory=dict)
    logits: Tensor | None = None
    prompt_positions: Tensor | None = None


def _unwrap_accelerate_forward(module: torch.nn.Module) -> None:
    wrapped = module.forward
    if isinstance(wrapped, types.MethodType) and wrapped.__name__ == "wrapped":
        function = cast(types.FunctionType, wrapped.__func__)
        functions = [cell.cell_contents for cell in function.__closure__ or () if inspect.isfunction(cell.cell_contents)]
        module.forward = types.MethodType(functions[0], module)


class GDNRunner:
    """Apply GDN-state interventions and greedily decode a batch."""

    def __init__(
        self,
        model: Any,
        tokenizer: PreTrainedTokenizer,
        layers: list[int],
        deltas: dict[int, Float[Tensor, "heads key value"]],
        normalize: bool = True,
    ) -> None:
        available = [index for index, layer in enumerate(model.model.layers) if hasattr(layer, "linear_attn")]
        if not deltas or not deltas.keys() <= set(layers) or any(layer not in available for layer in layers):
            raise ValueError("deltas must target configured GDN layers")

        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        self.deltas = deltas
        self.normalize = normalize
        self.device = next(model.parameters()).device
        self.tokenizer.padding_side = "left"
        self.traced: NNsight | None = None
        self.operations: dict[int, tuple[OperationEnvoy, OperationEnvoy]] = {}

    def _inputs(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        if not texts:
            raise ValueError("texts must not be empty")

        encoded = self.tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt").to(self.device)
        if not encoded.attention_mask.any(dim=1).all():
            raise ValueError("prompts must contain tokens")
        return encoded.input_ids, encoded.attention_mask

    def _enable_tracing(self) -> None:
        if self.traced is not None:
            return

        for layer in self.layers:
            _unwrap_accelerate_forward(self.model.model.layers[layer].linear_attn)
        self.traced = NNsight(self.model)
        for layer in self.layers:
            source = cast(Envoy, self.traced.get(f"model.layers.{layer}.linear_attn")).source
            self.operations[layer] = tuple(
                next(operation for operation in source.operations if name in operation.path)
                for name in ("chunk_gated_delta_rule", "recurrent_gated_delta_rule")
            )

    def _run_prefixes(
        self,
        layer: int,
        args: tuple[Tensor, ...],
        kwargs: dict[str, Any],
        columns: Tensor,
        scales: Tensor,
    ) -> tuple[list[Tensor], tuple[tuple[Tensor, ...], dict[str, Any]]]:
        query, key, value = args
        state = kwargs.get("initial_state")
        outputs: list[Tensor] = []
        start = 0
        kernel = self.model.model.layers[layer].linear_attn.chunk_gated_delta_rule

        for end in sorted(column for column in columns.unique().tolist() if 0 <= column < query.shape[1]):
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
                    query.shape[0], query.shape[2], query.shape[3], value.shape[3], device=query.device, dtype=torch.float32
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
        prefix, (tail_args, tail_kwargs) = self._run_prefixes(layer, *operation.inputs, columns, scales)
        operation.inputs = tail_args, tail_kwargs
        tail, state = operation.output
        output = torch.cat([*prefix, tail], dim=1) if prefix else tail

        if columns.eq(output.shape[1]).any():
            add_delta(state, self.deltas[layer], scales * columns.eq(output.shape[1]), self.normalize)
        operation.output = output, state
        return (layer, save((output, state))) if capture else None

    @staticmethod
    def _record(trace: GDNTrace, captured: dict[int, tuple[Tensor, Tensor]], mask: Tensor, logits: Tensor) -> None:
        counts = mask.long().sum(-1)
        for position in counts.unique().tolist():
            indices = counts.eq(position).nonzero().flatten()
            trace.indices[position] = indices
            trace.outputs[position] = {layer: values[0][indices] for layer, values in captured.items()}
            trace.states[position] = {layer: values[1][indices] for layer, values in captured.items()}
        trace.logits = logits

    def _model_forward(
        self,
        inputs: Tensor,
        mask: Tensor,
        positions: Tensor,
        cache: Any = None,
        trace: GDNTrace | None = None,
        columns: Tensor | None = None,
        scales: Tensor | None = None,
    ) -> tuple[Any, Tensor]:
        arguments = {
            "input_ids": inputs,
            "attention_mask": mask,
            "position_ids": positions,
            "past_key_values": cache,
            "use_cache": True,
            "logits_to_keep": 1,
        }
        if trace is None and columns is None:
            output = self.model(**arguments)
            return output.past_key_values, output.logits[:, -1]

        self._enable_tracing()
        saved: list[tuple[int, Any]] = []
        with self.traced.trace(**arguments):
            for layer, operations in self.operations.items():
                recurrent = inputs.shape[1] == 1 and cache is not None and cache.has_previous_state(layer)
                operation = operations[recurrent]
                if columns is not None and layer in self.deltas:
                    captured = self._replace_kernel(layer, operation, columns, scales, trace is not None)
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

    def _add_to_cache(self, cache: Any, scales: Tensor) -> None:
        for layer, delta in self.deltas.items():
            add_delta(cache.layers[layer].recurrent_states[0], delta, scales, self.normalize)

    @torch.inference_mode()
    def prefill(
        self,
        inputs: Tensor,
        mask: Tensor,
        prompt_steer_position: PromptPosition,
        scale: float | Tensor,
        prompt_steer_mode: PromptSteerMode = "exact",
        trace: GDNTrace | None = None,
    ) -> tuple[Any, Tensor, Tensor]:
        """Process a prompt batch and optionally alter each row's GDN state once."""
        if inputs.ndim != 2 or mask.shape != inputs.shape:
            raise ValueError("inputs and mask must be two-dimensional tensors of the same shape")
        if not mask.any(dim=1).all():
            raise ValueError("every prompt must contain a token")
        if prompt_steer_mode not in ("exact", "chunk"):
            raise ValueError("prompt_steer_mode must be exact or chunk")

        scales = torch.as_tensor(scale, device=self.device)
        if scales.ndim > 1 or (scales.ndim == 1 and len(scales) != len(inputs)):
            raise ValueError("scale must be scalar or have one value per prompt")

        mask = mask.bool()
        lengths = mask.long().sum(-1)
        padding = inputs.shape[1] - lengths
        positions = (mask.long().cumsum(-1) - 1).clamp_min(0)

        columns = None
        if prompt_steer_position is not None:
            requested = torch.as_tensor(prompt_steer_position, device=self.device)
            if requested.ndim > 1 or (requested.ndim == 1 and len(requested) not in (1, len(inputs))):
                raise ValueError("prompt_steer_position must be scalar or have one value per prompt")
            if requested.is_floating_point():
                raise ValueError("prompt_steer_position must contain integers")
            requested = requested.expand_as(lengths)
            columns = padding + torch.where(requested < 0, requested + lengths, requested)
            if ((columns < padding) | (columns > inputs.shape[1])).any():
                raise ValueError("prompt_steer_position must resolve within every prompt")
            if prompt_steer_mode == "chunk":
                columns = torch.maximum(columns // 64 * 64, padding)
            if trace is not None:
                trace.prompt_positions = columns - padding

        cache, logits = self._model_forward(inputs, mask, positions, trace=trace, columns=columns, scales=scales)
        if trace is not None:
            trace.prefill_outputs = {length: dict(values) for length, values in trace.outputs.items()}
            trace.prefill_indices = {length: indices.clone() for length, indices in trace.indices.items()}
        return cache, logits, mask

    @torch.inference_mode()
    def forward(
        self,
        texts: list[str],
        scale: float | Tensor = 1.0,
        prompt_steer_position: PromptPosition = -1,
        prompt_steer_mode: PromptSteerMode = "exact",
    ) -> GDNTrace:
        """Capture the final prefill state through the same path as generation."""
        trace = GDNTrace()
        inputs, mask = self._inputs(texts)
        self.prefill(inputs, mask, prompt_steer_position, scale, prompt_steer_mode, trace)
        return trace

    @torch.inference_mode()
    def generate(
        self,
        texts: list[str],
        scale: float | Tensor = 1.0,
        prompt_steer_position: PromptPosition = -1,
        prompt_steer_mode: PromptSteerMode = "exact",
        generation_steer_period: int | None = None,
        generation_steer_steps: set[int] | None = None,
        max_new_tokens: int = 64,
        trace: GDNTrace | None = None,
    ) -> Tensor:
        """Greedily generate one batch, with optional prompt and decode interventions."""
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if generation_steer_period is not None and generation_steer_steps is not None:
            raise ValueError("generation_steer_period and generation_steer_steps are mutually exclusive")
        if generation_steer_period is not None and generation_steer_period < 1:
            raise ValueError("generation_steer_period must be positive")
        if generation_steer_steps is not None and any(step < 1 for step in generation_steer_steps):
            raise ValueError("generation_steer_steps must contain positive values")

        inputs, mask = self._inputs(texts)
        scales = torch.as_tensor(scale, device=self.device)
        cache, logits, mask = self.prefill(inputs, mask, prompt_steer_position, scales, prompt_steer_mode, trace)
        generated = torch.full(
            (len(inputs), max_new_tokens), self.tokenizer.pad_token_id, device=self.device, dtype=torch.long
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

            inject = (
                generation_steer_period is not None and (step + 1) % generation_steer_period == 0
            ) or (generation_steer_steps is not None and step + 1 in generation_steer_steps)
            if inject:
                self._add_to_cache(cache, scales * ~finished)
            mask = torch.cat((mask, torch.ones_like(mask[:, :1])), dim=1)
            position = (mask.long().sum(-1) - 1)[:, None]
            cache, logits = self._model_forward(token[:, None], mask, position, cache, trace)

        return generated
