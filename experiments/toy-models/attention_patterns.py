"""Extract attention for the four fixed causal-probe contexts."""

import argparse
import json
from pathlib import Path

import torch

from model import load_checkpoint


PROBES = [
    ("H0", "York … New → York", " York", " New"),
    ("H5", "for … a → lot", " for", " a"),
    ("H6", "No … . → 1", " No", "."),
    ("H8", "can … can → 't", " can", " can"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


@torch.inference_mode()
def extract_patterns(model, tokenizer) -> list[dict]:
    filler = tokenizer.encode(" the", add_special_tokens=False)
    patterns = []
    for head, label, source, destination in PROBES:
        token_ids = (
            tokenizer.encode(source, add_special_tokens=False)
            + filler * 8
            + tokenizer.encode(destination, add_special_tokens=False)
        )
        _, cache = model.run_with_cache(
            torch.tensor([token_ids], device=model.W_E.device)
        )
        patterns.append(
            {
                "head": head,
                "label": label,
                "tokens": [tokenizer.decode([token]) for token in token_ids],
                "attention": cache["blocks.0.attn.hook_pattern"][0].tolist(),
            }
        )
    return patterns


def main() -> None:
    args = parse_args()
    model, tokenizer, _, checkpoint = load_checkpoint(
        args.run_dir, args.checkpoint, args.device
    )
    output = args.output or args.run_dir / "reports" / "attention_patterns.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"checkpoint": str(Path(checkpoint).resolve()), "probes": extract_patterns(model, tokenizer)},
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
