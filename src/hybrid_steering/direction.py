"""Build a target-minus-source direction from concept pairs.

``positive_text`` is the target and ``negative_text`` is the source. The
stored matrix is the full mean; experiments apply ``rank`` when they load it.

``hybrid-direction --concept en-ru --output runs/directions/en-ru`` writes
``runs/directions/en-ru/direction``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import save_direction
from .cache import gdn_layers
from .extract import CollectedDirection, collect_direction
from .mamba import MambaRunner
from .models import DirectionManifest
from .runner import Runner
from .runtime import load_runtime, read_jsonl

CONCEPT_DATASET = "hybrid-steering/hybrid-steering-concepts"


def concept_sides(concept: str, source: str | None, target: str | None) -> tuple[str, str]:
    """Return source and target names. ``en-ru`` means English source, Russian target."""
    if source and target:
        return source, target
    parts = concept.strip().replace("->", "-").split("-")
    if len(parts) == 2 and all(parts):
        return source or parts[0], target or parts[1]
    raise ValueError("pass --source and --target, or a concept like en-ru")


def target_and_source(row: dict) -> tuple[str, str]:
    """Map one dataset row onto ``(target, source)``."""
    target, source = row.get("positive_text"), row.get("negative_text")
    if not target or not source:
        raise ValueError("pair row needs positive_text (target) and negative_text (source)")
    return str(target), str(source)


def read_pairs(path: str | Path) -> list[dict]:
    """Read pair rows and require both texts."""
    rows = read_jsonl(path)
    for row in rows:
        target_and_source(row)
    return rows


def load_concept_pairs(concept: str, name: str = "pairs.jsonl") -> list[dict]:
    """Download one concept file from the Hub dataset."""
    from huggingface_hub import hf_hub_download

    slug = concept.strip().replace("->", "-")
    path = Path(
        hf_hub_download(CONCEPT_DATASET, f"concepts/{slug}/data/{name}", repo_type="dataset")
    )
    return read_pairs(path)


def collect_from_pairs(
    model, tokenizer, pairs: list[tuple[str, str]], batch_size: int
) -> CollectedDirection:
    """Average target-minus-source over pairs. No steering is applied."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    runner = (
        MambaRunner(model, tokenizer)
        if getattr(model.config, "model_type", None) == "falcon_h1"
        else Runner(model, tokenizer, gdn_layers(model), normalize=False)
    )
    return collect_direction(runner, pairs, batch_size=batch_size)


def write_direction(
    directory: str | Path,
    model_id: str,
    target: str,
    source: str,
    rows: list[dict],
    collected: CollectedDirection,
) -> None:
    """Save the mean direction and the states it was averaged from."""
    save_direction(
        directory,
        collected.delta,
        DirectionManifest(
            model_id=model_id,
            target=target,
            source=source,
            example_ids=[
                str(row.get("pair_id") or f"pair-{index}") for index, row in enumerate(rows)
            ],
            decoder_layer_indices=sorted(collected.delta),
            state_shapes={layer: list(tensor.shape) for layer, tensor in collected.delta.items()},
        ),
        mean_target=collected.mean_target,
        mean_source=collected.mean_source,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--concept", required=True, help="dataset directory, such as en-ru")
    parser.add_argument("--jsonl", type=Path, help="local pairs; skips the Hub download")
    parser.add_argument("--pairs", type=int, default=0, help="0 uses every stored pair")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--source", help="source name when the concept slug has no hyphen")
    parser.add_argument("--target", help="target name when the concept slug has no hyphen")
    args = parser.parse_args(argv)
    source, target = concept_sides(args.concept, args.source, args.target)
    rows = read_pairs(args.jsonl) if args.jsonl else load_concept_pairs(args.concept)
    rows = rows[: args.pairs or None]
    if not rows:
        raise SystemExit("no pairs to average")
    model, tokenizer = load_runtime(args.model)
    collected = collect_from_pairs(
        model, tokenizer, [target_and_source(row) for row in rows], args.batch_size
    )
    directory = args.output / "direction"
    write_direction(directory, args.model, target, source, rows, collected)
    print(f"wrote {directory}", flush=True)


if __name__ == "__main__":
    main()
