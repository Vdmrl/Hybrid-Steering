"""Train the one-layer attention-only Transformer Circuits model."""

import argparse
from dataclasses import asdict
import math
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset

from model import ModelSpec, build_model, load_tokenizer

DEFAULTS = {
    "tokenizer": "openai-community/gpt2",
    "steps": 10_000,
    "batch_size": 64,
    "gradient_accumulation": 1,
    "learning_rate": 3e-4,
    "minimum_learning_rate": 3e-5,
    "warmup_steps": 1_000,
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,
    "mixed_precision": "bf16",
    "checkpoint_every": 500,
    "keep_checkpoints": 8,
    "validation_batches": 8,
    "validation_documents": 1_000,
    "seed": 42,
    "d_model": 768,
    "heads": 12,
    "d_head": 64,
    "context_length": 2_048,
    "log_every": 10,
    "snapshots": 32,
}


class PackedTokenDataset(IterableDataset):
    """Deterministically stream documents as fixed-length token blocks."""

    def __init__(
        self,
        *,
        files: list[Path],
        tokenizer,
        sequence_length: int,
        sequences: int,
        seed: int,
        skip_documents: int = 0,
        skip_sequences: int = 0,
    ):
        self.files = files
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.sequences = sequences
        self.seed = seed
        self.skip_documents = skip_documents
        self.skip_sequences = skip_sequences

    def __len__(self) -> int:
        return self.sequences

    def __iter__(self):
        import pyarrow.parquet as pq

        files = list(self.files)
        random.Random(self.seed).shuffle(files)

        def documents():
            for path in files:
                parquet = pq.ParquetFile(path)
                for batch in parquet.iter_batches(columns=["text"], batch_size=256):
                    yield from batch.column(0).to_pylist()

        tokens: list[int] = []
        offset = skipped_sequences = yielded = 0
        for document_index, text in enumerate(documents()):
            if document_index < self.skip_documents:
                continue
            tokens.extend(self.tokenizer.encode(text, add_special_tokens=False))
            tokens.append(self.tokenizer.eos_token_id)
            while len(tokens) - offset >= self.sequence_length:
                block = tokens[offset : offset + self.sequence_length]
                offset += self.sequence_length
                if skipped_sequences < self.skip_sequences:
                    skipped_sequences += 1
                    continue
                yield torch.tensor(block, dtype=torch.long)
                yielded += 1
                if yielded == self.sequences:
                    return
            if offset > 1_000_000:
                tokens, offset = tokens[offset:], 0
        raise RuntimeError(
            f"dataset ended after {yielded} of {self.sequences} requested sequences"
        )


def log_spaced(steps: int, count: int) -> list[int]:
    """Return up to ``count`` unique step numbers log-spaced over ``[1, steps]``."""
    return sorted({max(1, round(steps ** (i / (count - 1)))) for i in range(count)})


def cosine_schedule(steps: int, warmup: int, minimum_ratio: float):
    def scale(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, steps - warmup))
        return minimum_ratio + (1 - minimum_ratio) * (
            1 + math.cos(math.pi * progress)
        ) / 2

    return scale


def language_model_step(tokens, trainer):
    loss = trainer.model(tokens, return_type="loss")
    return {"loss": loss, "nll": loss.detach()}


class SnapshotHook:
    """Save weights-only snapshots at fixed, log-spaced steps for a checkpoint sweep."""

    ord = -3

    def __init__(self, save_dir: Path, steps: set[int]):
        self.save_dir, self.steps = save_dir, steps
        save_dir.mkdir(parents=True, exist_ok=True)

    def after_step(self, trainer) -> None:
        step = trainer.step_state.optimizer_step
        if not (trainer.training and trainer.is_main and step in self.steps):
            return
        model = (
            trainer.accelerator.unwrap_model(trainer.model)
            if trainer.is_distributed
            else trainer.model
        )
        torch.save({"model": model.state_dict()}, self.save_dir / f"step_{step}.pt")


def parse_args(default_layers: str = "attn") -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a one-layer attention-only model on OpenWebText."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true", help="two tiny CPU/GPU steps")
    parser.add_argument("--layers", default=default_layers, help="comma-separated attn/gdn layer order")
    parser.add_argument("--gdn-backend", choices=("auto", "fla", "reference"), default="auto")
    parser.add_argument("--steps", type=int, help="optimizer steps")
    parser.add_argument("--d-model", type=int, dest="d_model")
    parser.add_argument("--heads", type=int)
    parser.add_argument("--d-head", type=int, dest="d_head")
    parser.add_argument("--context-length", type=int, dest="context_length")
    parser.set_defaults(**DEFAULTS)
    args = parser.parse_args()
    if args.smoke:
        args.steps, args.batch_size, args.gradient_accumulation = 2, 2, 1
        args.d_model, args.heads, args.d_head, args.context_length = 64, 4, 16, 64
        args.warmup_steps, args.validation_batches, args.checkpoint_every = 1, 1, 1
        args.mixed_precision = "bf16" if torch.cuda.is_available() else "no"
    if args.minimum_learning_rate > args.learning_rate:
        parser.error("minimum learning rate cannot exceed learning rate")
    return args


def main(default_layers: str = "attn") -> None:
    from omegaconf import OmegaConf
    from trainer_tools.hooks import (
        CheckpointHook,
        Loss,
        LRSchedulerHook,
        MetricsHook,
        ProgressBarHook,
    )
    from trainer_tools.hooks.accelerate import AccelerateHook
    from trainer_tools.trainer import Trainer

    args = parse_args(default_layers)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.run_dir = args.run_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    files = sorted(args.data_dir.glob("plain_text/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no OpenWebText parquet shards under {args.data_dir}")
    validation_files = files[:1]
    train_files = files[1:] or files
    checkpoint_dir = args.run_dir / "checkpoints"
    if not args.resume and any(checkpoint_dir.glob("*.pt")):
        raise FileExistsError(f"{checkpoint_dir} already contains a run; use --resume")
    args.run_dir.mkdir(parents=True, exist_ok=True)

    resume_state = (
        torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.resume
        else {}
    )
    completed_steps = int(resume_state.get("optimizer_step", 0))
    remaining_steps = args.steps - completed_steps
    if remaining_steps < 1:
        raise ValueError(f"checkpoint already reached {completed_steps} steps")

    tokenizer = load_tokenizer(args.tokenizer)
    spec = ModelSpec(
        d_model=args.d_model,
        n_heads=args.heads,
        d_head=args.d_head,
        n_ctx=args.context_length,
        seed=args.seed,
        layer_types=tuple(args.layers.split(",")),
        gdn_backend=args.gdn_backend,
    )
    model = build_model(spec, tokenizer)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    accelerate = AccelerateHook(
        gradient_accumulation_steps=args.gradient_accumulation,
        max_grad_norm=args.max_grad_norm,
        mixed_precision=args.mixed_precision,
    )
    world_size = accelerate.accelerator.num_processes
    train_sequences = (
        remaining_steps
        * args.gradient_accumulation
        * args.batch_size
        * world_size
    )
    train_data = PackedTokenDataset(
        files=train_files,
        tokenizer=tokenizer,
        sequence_length=args.context_length,
        sequences=train_sequences,
        seed=args.seed,
        skip_documents=args.validation_documents if len(files) == 1 else 0,
        skip_sequences=int(resume_state.get("samples_seen", 0)),
    )
    valid_data = PackedTokenDataset(
        files=validation_files,
        tokenizer=tokenizer,
        sequence_length=args.context_length,
        sequences=args.validation_batches * args.batch_size * world_size,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, drop_last=True, num_workers=0, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_data, batch_size=args.batch_size, drop_last=True, num_workers=0, pin_memory=True
    )

    config = OmegaConf.create(
        {
            "paper": "https://transformer-circuits.pub/2021/framework/index.html",
            "model": asdict(spec),
            "data": {
                "data_dir": str(args.data_dir),
                "train_files": [str(path) for path in train_files],
                "validation_files": [str(path) for path in validation_files],
                "tokenizer": args.tokenizer,
                "validation_documents": args.validation_documents,
            },
            "training": {
                "steps": args.steps,
                "batch_size_per_device": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation,
                "world_size": world_size,
                "learning_rate": args.learning_rate,
                "minimum_learning_rate": args.minimum_learning_rate,
                "warmup_steps": args.warmup_steps,
                "weight_decay": args.weight_decay,
                "max_grad_norm": args.max_grad_norm,
                "mixed_precision": args.mixed_precision,
                "seed": args.seed,
                "target_tokens": args.steps
                * args.batch_size
                * args.gradient_accumulation
                * world_size
                * (args.context_length - 1),
            },
            "deviations": [
                "OpenWebText replaces the unpublished Kaplan et al. corpus.",
                "Optimizer, schedule, and token budget were not specified by the paper.",
            ],
        }
    )

    snapshot_dir = args.run_dir / "snapshots"
    snapshot_steps = set(log_spaced(args.steps, args.snapshots))
    config["training"]["snapshot_steps"] = sorted(snapshot_steps)

    minimum_ratio = args.minimum_learning_rate / args.learning_rate
    hooks = [
        SnapshotHook(snapshot_dir, snapshot_steps),
        LRSchedulerHook(
            lambda opt: torch.optim.lr_scheduler.LambdaLR(
                opt,
                cosine_schedule(args.steps, args.warmup_steps, minimum_ratio),
            )
        ),
        accelerate,
        MetricsHook(
            [Loss(loss_key="nll")],
            tracker_type="file",
            log_file=str(args.run_dir / "metrics.jsonl"),
            freq=args.log_every,
            config=OmegaConf.to_container(config, resolve=True),
        ),
        CheckpointHook(
            save_dir=str(checkpoint_dir),
            save_every_steps=args.checkpoint_every,
            keep_last=args.keep_checkpoints,
            resume_path=str(args.resume) if args.resume else None,
            save_strategy="latest",
        ),
        ProgressBarHook(freq=args.log_every),
    ]
    if accelerate.accelerator.is_main_process:
        parameters = sum(p.numel() for p in model.parameters())
        print(
            f"parameters={parameters:,} target_steps={args.steps:,} "
            f"target_tokens={config.training.target_tokens:,}"
        )
    Trainer(
        model=model,
        train_step=language_model_step,
        train_dl=train_loader,
        valid_dl=valid_loader,
        optim=optimizer,
        epochs=1,
        hooks=hooks,
        config=config,
    ).fit()


if __name__ == "__main__":
    main()
