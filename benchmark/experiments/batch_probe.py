"""Quickly choose a safe, fast runtime batch for model-generated direction pairs."""

import gc
import hashlib
import json
import random
import time
from pathlib import Path


def choose_batch(records: list[dict]) -> int:
    safe = [row for row in records if row["status"] == "ok" and row["safe"]]
    if not safe:
        raise RuntimeError("No safe extraction batch found")
    return max(safe, key=lambda row: (row["examples_per_second"], -row["batch_size"]))["batch_size"]


def tune(model, tokenizer, prompts: list[str], instruction: str, settings: dict,
         output_dir: Path, source_sha256: str) -> int:
    """Probe target-conditioned answers with the real decoder, then cache the choice.

    Probe responses are discarded; `comparison.prepare` generates and saves the
    actual neutral/target pairs. Batch size is runtime-only, not direction data.
    """
    import torch
    from hybrid_steering.comparison import generate
    from hybrid_steering.paths import ROOT

    candidates = settings["candidates"]
    if (not candidates or any(type(n) is not int or n < 1 for n in candidates)
            or candidates != sorted(set(candidates)) or max(candidates) > len(prompts)):
        raise ValueError("Batch candidates must be sorted, unique and within the dataset")
    tokens = settings["probe_tokens"]
    limit = settings["max_memory_fraction"]
    if type(tokens) is not int or tokens < 16 or not 0 < limit < 1:
        raise ValueError("Invalid batch probe token limit or memory fraction")
    identity = {
        "model": model.config._name_or_path, "revision": getattr(model.config, "_commit_hash", None),
        "source_sha256": source_sha256, "instruction": instruction,
        "candidates": candidates, "probe_tokens": tokens, "max_memory_fraction": limit,
        "gpu_mask": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "code_sha256": hashlib.sha256((ROOT / "experiments/batch_probe.py").read_bytes()).hexdigest(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    output = output_dir / f"batch-probe-{key}.json"
    if output.exists():
        saved = json.loads(output.read_text())
        if saved["identity"] != identity or saved["selected"] != choose_batch(saved["measurements"]):
            raise ValueError(f"Saved batch probe differs: {output}")
        print(f"BATCH_PROBE reuse batch={saved['selected']} {output}", flush=True)
        return saved["selected"]

    # A fixed, varied sample prevents the first few similar prompts dominating the choice.
    sample = random.Random(42).sample(prompts, max(candidates))
    conditioned = [row + "\n\n" + instruction for row in sample]
    generate(model, tokenizer, [{"prompt": conditioned[0]}], min(tokens, 16))
    records = []
    for size in candidates:
        gc.collect()
        torch.cuda.empty_cache()  # Do not charge earlier smoke reservations to this candidate.
        for device in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device)
        start = time.monotonic()
        try:
            answers = generate(model, tokenizer, [{"prompt": text} for text in conditioned[:size]], tokens)
            for device in range(torch.cuda.device_count()):
                torch.cuda.synchronize(device)
            if len(answers) != size or any(not isinstance(answer, str) or not answer.strip() for answer in answers):
                raise ValueError(f"Empty or missing batch-probe answer at size={size}")
            elapsed = time.monotonic() - start
            peak = [torch.cuda.max_memory_reserved(device) / 2**20
                    for device in range(torch.cuda.device_count())]
            total = [torch.cuda.get_device_properties(device).total_memory / 2**20
                     for device in range(torch.cuda.device_count())]
            record = {"batch_size": size, "status": "ok", "seconds": elapsed,
                      "examples_per_second": size / elapsed,
                      "generated_tokens": sum(len(tokenizer.encode(answer, add_special_tokens=False))
                                              for answer in answers),
                      "peak_reserved_mib": [round(value) for value in peak],
                      "safe": all(used / capacity <= limit for used, capacity in zip(peak, total))}
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            record = {"batch_size": size, "status": "oom", "safe": False}
        records.append(record)
        print(f"BATCH_PROBE {record}", flush=True)
        if record["status"] == "oom":
            break
    selected = choose_batch(records)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps({"identity": identity, "measurements": records,
                                     "selected": selected}, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output)
    print(f"BATCH_PROBE selected={selected} {output}", flush=True)
    return selected
