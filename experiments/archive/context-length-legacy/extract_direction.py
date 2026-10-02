"""Extract a dataset-mean GDN state direction."""

import argparse
from contextlib import ExitStack
from itertools import islice
from pathlib import Path

import torch
from datasets import load_dataset

from gdn_interp import gdn_layers, load_qwen


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a dataset-mean GDN state steering direction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B", help="model ID")
    parser.add_argument(
        "--dataset",
        default="../shared/optimism_pessimism_stories.jsonl",
        help="Hugging Face dataset ID or local JSONL path",
    )
    parser.add_argument("--config", default="simplified", help="dataset config")
    parser.add_argument("--split", default="train", help="dataset split")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument(
        "--labels",
        nargs=2,
        default=("optimism (general outlook)", "pessimism (general outlook)"),
        metavar=("POSITIVE", "NEGATIVE"),
        help="labels used to compute POSITIVE minus NEGATIVE",
    )
    parser.add_argument(
        "--limit", type=int, default=128, help="examples to use per label"
    )
    parser.add_argument(
        "--text-columns",
        nargs=2,
        metavar=("POSITIVE_TEXT", "NEGATIVE_TEXT"),
        help="paired text columns; rows must also match --labels",
    )
    parser.add_argument(
        "--where",
        action="append",
        default=[],
        metavar="COLUMN=VALUE",
        help="additional exact row filter (repeatable)",
    )
    parser.add_argument(
        "--max-length", type=int, default=128, help="maximum input token count"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="timestamped artifacts/context_length run directory",
    )
    args = parser.parse_args()

    model, tokenizer = load_qwen(args.model)
    model.eval()
    dataset_path = Path(args.dataset)
    if dataset_path.is_file():
        dataset = load_dataset(
            "json", data_files=str(dataset_path), split="train", streaming=args.streaming
        )
    else:
        dataset = load_dataset(
            args.dataset,
            args.config or None,
            split=args.split,
            streaming=args.streaming,
        )
    filters = dict(item.split("=", 1) for item in args.where)
    paired_rows = None
    if args.text_columns:
        paired_rows = list(
            islice(
                (
                    row
                    for row in dataset
                    if row["concept"] == args.labels[0]
                    and row["antagonist"] == args.labels[1]
                    and all(str(row[key]) == value for key, value in filters.items())
                ),
                args.limit,
            )
        )
        if len(paired_rows) < args.limit:
            raise ValueError(f"only found {len(paired_rows)} matching paired rows")
        label_ids = (None, None)
    else:
        label_names = dataset.features["labels"].feature.names
        label_ids = [label_names.index(label) for label in args.labels]
    found = gdn_layers(model)
    physical_layers = [module.layer_idx for _, module in found]
    device = model.get_input_embeddings().weight.device
    captured: dict[int, torch.Tensor] = {}

    def capture(module, _args, kwargs, _output):
        captured[module.layer_idx] = (
            kwargs["cache_params"]
            .layers[module.layer_idx]
            .recurrent_states[0]
            .detach()
            .float()
            .cpu()
            .squeeze(0)
        )

    means, mean_state_squares, counts = [], [], []
    with ExitStack() as stack:
        for _, module in found:
            handle = module.register_forward_hook(capture, with_kwargs=True)
            stack.callback(handle.remove)

        text_columns = args.text_columns or (None, None)
        for label, label_id, text_column in zip(
            args.labels, label_ids, text_columns, strict=True
        ):
            total = None
            state_square = torch.zeros(len(found))
            count = 0
            for row in paired_rows if paired_rows is not None else dataset:
                if paired_rows is None and row["labels"] != [label_id]:
                    continue
                captured.clear()
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": row[text_column or "text"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_length,
                ).to(device)
                with torch.inference_mode():
                    model(**inputs, use_cache=True)
                states = torch.stack([captured[index] for index in physical_layers])
                total = states if total is None else total + states
                state_square += states.square().sum((1, 2, 3))
                count += 1
                if count == args.limit:
                    break
            if count == 0:
                raise ValueError(f"no single-label examples found for {label}")
            means.append(total / count)
            mean_state_squares.append(state_square / count)
            counts.append(count)
            print(f"{label}: {count}")

    direction = means[0] - means[1]
    state_rms = (0.5 * (mean_state_squares[0] + mean_state_squares[1])).sqrt()
    output = args.run_dir / "direction.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": args.model,
            "dataset": args.dataset,
            "config": args.config,
            "split": args.split,
            "streaming": args.streaming,
            "labels": tuple(args.labels),
            "counts": counts,
            "gdn_layers": [name for name, _ in found],
            "decoder_layers": physical_layers,
            "direction": direction,
            "state_rms": state_rms,
            "text_columns": args.text_columns,
            "filters": filters,
            "max_length": args.max_length,
        },
        output,
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
