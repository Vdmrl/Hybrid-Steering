import inspect
import types
from typing import TypeAlias, cast

import torch
import torch.nn.functional as F
from nnsight import NNsight, save
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.source import OperationEnvoy
from transformers import DynamicCache, PreTrainedModel

from .metrics import state_transition_metrics

MetricRow: TypeAlias = tuple[int, int, int, int, str, float]
CHUNK_SIZE = 64
CAPTURE_POSITIONS = (*range(1, 32), 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def unwrap_accelerate_forward(module: torch.nn.Module) -> None:
    wrapped = module.forward
    if isinstance(wrapped, types.MethodType) and wrapped.__name__ == "wrapped":
        function = cast(types.FunctionType, wrapped.__func__)
        forwards = [cell.cell_contents for cell in function.__closure__ or () if inspect.isfunction(cell.cell_contents)]
        module.forward = types.MethodType(forwards[0], module)


def kernel_operation(layer: Envoy, name: str) -> OperationEnvoy:
    return next(operation for operation in layer.source.operations if name in operation.path)


class ChunkCapture:
    def __init__(self, model: PreTrainedModel) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        paths = [f"model.layers.{index}.linear_attn" for index, layer in enumerate(model.model.layers) if hasattr(layer, "linear_attn")]
        for path in paths:
            unwrap_accelerate_forward(model.get_submodule(path))
        self.traced = NNsight(model)
        self.operations = {
            int(path.split(".")[2]): (
                kernel_operation(cast(Envoy, self.traced.get(path)), "chunk_gated_delta_rule"),
                kernel_operation(cast(Envoy, self.traced.get(path)), "recurrent_gated_delta_rule"),
            )
            for path in paths
        }

    def capture(
        self, segment: torch.Tensor, end: int, cache: DynamicCache, recurrent: bool
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        saved = []
        with (
            torch.inference_mode(),
            self.traced.trace(
                input_ids=segment,
                attention_mask=torch.ones(len(segment), end, device=self.device, dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
            ),
        ):
            for layer, operations in self.operations.items():
                inputs, output = save(operations[recurrent].inputs), save(operations[recurrent].output)
                saved.append((layer, inputs, output))
        result = {}
        for layer, inputs, output in saved:
            positional, keyword = inputs
            key, value, beta, gate = positional[1], positional[2], keyword["beta"], keyword["g"]
            _, state = output
            result[layer] = key, value, beta, gate, state
        return result

    def capture_data(
        self, state: torch.Tensor, key: torch.Tensor, value: torch.Tensor, beta: torch.Tensor, gate: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return per-head measurements; subclasses can override this."""
        key = F.normalize(key[:, 0].float(), dim=-1, eps=1e-6).to(key)
        return state_transition_metrics(state, key, value[:, 0], beta[:, 0], gate[:, 0])

    def measure(self, input_ids: torch.Tensor) -> list[MetricRow]:
        required = max(CAPTURE_POSITIONS) + 1
        if input_ids.shape[1] < required:
            raise ValueError(f"need at least {required} tokens to measure every requested boundary")
        cache, rows, pending = DynamicCache(config=self.model.config), [], {}
        start = 0
        while start < required:
            targets = [position for position in CAPTURE_POSITIONS if start < position <= start + CHUNK_SIZE]
            end = targets[0] if targets else min(start + CHUNK_SIZE, required)
            if start < 32:
                end = start + 1
            captured = self.capture(input_ids[:, start:end], end, cache, recurrent=start > 0 and end - start == 1)
            for layer, (key, value, beta, gate, state) in captured.items():
                if layer in pending:
                    position, old_state = pending.pop(layer)
                    for metric, values in self.capture_data(old_state, key, value, beta, gate).items():
                        for document, head in torch.cartesian_prod(torch.arange(values.shape[0]), torch.arange(values.shape[1])).tolist():
                            rows.append((layer, position, document, head, metric, float(values[document, head].cpu())))
                if end in CAPTURE_POSITIONS:
                    pending[layer] = end, state
            start = end
        return rows
