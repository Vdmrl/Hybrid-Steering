"""Extract a target-minus-source direction and score generations with it.

``--concept en-ru`` loads pairs from the concept dataset. Evaluation prompts
come from ``hybrid_steering.detect.QUESTIONS``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from hybrid_steering import (
    DirectionManifest,
    FrobeniusDelta,
    Runner,
    collect_direction,
    concept_detector,
    gdn_layers,
    load_runtime,
    save_direction,
    truncate_direction,
)
from hybrid_steering.detect import EVAL_QUESTIONS

DATASET = "hybrid-steering/hybrid-steering-concepts"


def hub_rows(concept: str, name: str) -> list[dict]:
    slug = concept.strip().replace("->", "-")
    path = Path(hf_hub_download(DATASET, f"concepts/{slug}/data/{name}", repo_type="dataset"))
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--concept", default="en-ru")
    parser.add_argument("--pairs", type=int, default=0, help="0 uses every stored pair")
    parser.add_argument("--questions", type=int, default=0, help="0 uses every evaluation prompt")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    source, target = args.concept.replace("->", "-").split("-")
    stored = hub_rows(args.concept, "pairs.jsonl")
    pairs = [(row["positive_text"], row["negative_text"]) for row in stored[: args.pairs or None]]
    questions = list(EVAL_QUESTIONS[: args.questions or None])
    model, tokenizer = load_runtime(args.model)
    layers = gdn_layers(model)
    collected = collect_direction(
        Runner(model, tokenizer, layers, normalize=False),
        pairs,
        batch_size=args.batch_size,
        metrics=(FrobeniusDelta(),),
    )
    direction = truncate_direction(collected.delta, args.rank)
    save_direction(
        args.output / "direction",
        collected.delta,
        DirectionManifest(
            model_id=args.model,
            target=target,
            source=source,
            example_ids=[f"pair-{index}" for index in range(collected.pairs)],
            decoder_layer_indices=sorted(collected.delta),
            state_shapes={layer: list(tensor.shape) for layer, tensor in collected.delta.items()},
            rank=args.rank or None,
        ),
        mean_target=collected.mean_target,
        mean_source=collected.mean_source,
    )
    device = next(model.parameters()).device
    runner = Runner(
        model,
        tokenizer,
        sorted(direction),
        {layer: tensor.to(device) for layer, tensor in direction.items()},
        normalize=args.normalize,
    )
    detector = concept_detector(target)
    rows = []
    for scale in args.scales:
        tokens = runner.generate(
            questions,
            scale=scale,
            prompt_position=args.position,
            max_new_tokens=args.max_new_tokens,
        )
        for prompt, row_tokens in zip(questions, tokens, strict=True):
            text = tokenizer.decode(row_tokens, skip_special_tokens=True)
            rows.append(
                {
                    "scale": scale,
                    "prompt": prompt,
                    "response": text,
                    "language": detector.label(text),
                    "target_language": detector.detects(text),
                    "direction": f"{target} - {source}",
                }
            )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "generations.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
