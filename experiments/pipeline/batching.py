"""Single-GPU batch growth with OOM retry; never drop an input."""

import time


def adaptive_batches(
    items, initial, generate, *, maximum, target_fraction, memory_fraction, reset_memory
):
    if initial < 1 or maximum < initial or not 0 < target_fraction < 1:
        raise ValueError("invalid batch policy")
    offset, width, retries = 0, initial, 0
    previous = None
    while offset < len(items):
        batch = items[offset : offset + width]
        reset_memory()
        base_fraction = memory_fraction()
        started = time.monotonic()
        try:
            tokens = generate(batch)
        except RuntimeError as error:
            if "out of memory" not in str(error).lower() or len(batch) == 1:
                raise
            width = max(1, len(batch) // 2)
            # Retain a margin after OOM; do not immediately grow back into it.
            maximum = min(maximum, width)
            retries += 1
            reset_memory()
            continue
        if len(tokens) != len(batch):
            raise ValueError("generator returned a different number of answers")
        fraction = memory_fraction()
        elapsed = time.monotonic() - started
        throughput = len(batch) / max(elapsed, 1e-9)
        yield (
            batch,
            tokens,
            {
                "batch_size": len(batch),
                "peak_memory_fraction": fraction,
                "seconds": elapsed,
                "responses_per_second": throughput,
                "oom_retries": retries,
            },
        )
        offset += len(batch)
        if previous and len(batch) > previous[0] and throughput < previous[1] * 0.9:
            width = previous[0]
            maximum = min(maximum, width)
        elif 2 * fraction - base_fraction <= target_fraction and width < maximum:
            previous = (len(batch), throughput)
            width = min(maximum, width * 2)
