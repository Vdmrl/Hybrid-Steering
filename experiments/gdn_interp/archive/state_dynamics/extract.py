"""Stream GDN state metrics to a sharded Parquet dataset."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import DynamicCache

from gdn_interp import gdn_layers, load_qwen
from gdn_interp.dynamics import (
    matrix_metrics,
    measurement_positions,
    states_from_cache,
)

SCALARS = (
    "frobenius_norm",
    "sigma_1",
    "effective_rank",
    "energy_rank_50",
    "energy_rank_90",
    "energy_rank_99",
    "stable_rank",
    "anisotropy",
    "relative_update",
    "state_cosine",
    "subspace_overlap_4",
    "subspace_overlap_8",
    "subspace_overlap_16",
)
SPECTRA = (
    "singular_values",
    "singular_values_normalized",
    "singular_energy_normalized",
)
MATRIX_CAPTURE_POSITIONS = (64, 256, 1024, 4096)


def rows(
    metrics: dict[str, torch.Tensor],
    *,
    record: dict,
    position: int,
    length: int,
    gdn_layers: list[int],
    decoder_layers: list[int],
    heads: list[int],
) -> dict[str, list]:
    count = len(decoder_layers) * len(heads)
    result = {
        "text_id": [record["text_id"]] * count,
        "data_type": [record["data_type"]] * count,
        "layer": [layer for layer in gdn_layers for _ in heads],
        "decoder_layer": [layer for layer in decoder_layers for _ in heads],
        "head": heads * len(decoder_layers),
        "position": [position] * count,
        "text_length": [length] * count,
    }
    for key in SCALARS:
        result[key] = metrics[key].float().cpu().reshape(-1).tolist()
    for key in SPECTRA:
        result[key] = metrics[key].float().cpu().flatten(0, 1).tolist()
    return result


def validate_orientation(modules, states: torch.Tensor, heads: list[int]) -> None:
    first = modules[0]
    expected = (len(modules), len(heads), first.head_k_dim, first.head_v_dim)
    if tuple(states.shape) != expected:
        raise RuntimeError(
            f"unexpected state orientation {tuple(states.shape)}; expected "
            f"[layer, head, key, value] = {expected}"
        )


def save_matrix_snapshot(
    matrix_dir: Path,
    *,
    record: dict,
    position: int,
    selected_layers: list[int],
    decoder_layers: list[int],
    heads: list[int],
    saved_states: dict[int, torch.Tensor],
) -> None:
    torch.save(
        {
            "text_id": record["text_id"],
            "data_type": record["data_type"],
            "position": position,
            "layers": selected_layers,
            "decoder_layers": decoder_layers,
            "heads": heads,
            "state_orientation": "[layer, head, key, value]",
            "states": torch.stack([saved_states[index] for index in selected_layers]),
        },
        matrix_dir / f"{record['text_id']}-{record['data_type']}-{position:05d}.pt",
    )


def extract_record(
    model,
    record: dict,
    output: Path,
    *,
    device: torch.device,
    selected_layers: list[int],
    modules: list,
    decoder_layers: list[int],
    heads: list[int],
    every_token: bool,
    capture_matrices: bool,
    matrix_dir: Path,
    svd_driver: str,
) -> None:
    input_ids = record["input_ids"].to(device=device, dtype=torch.long)
    length = len(input_ids)
    matrix_positions = {
        position
        for position in (*MATRIX_CAPTURE_POSITIONS, length)
        if position <= length
    }
    cache = DynamicCache(config=model.config)
    groups = None
    previous_states = {}
    previous_vh = {}
    writer = None
    start = 0
    try:
        for position in measurement_positions(length, every_token):
            with torch.inference_mode():
                model(
                    input_ids=input_ids[start:position].unsqueeze(0),
                    attention_mask=torch.ones(1, position, dtype=torch.long, device=device),
                    past_key_values=cache,
                    use_cache=True,
                )
            if groups is None:
                by_device = {}
                for index, decoder_layer in enumerate(decoder_layers):
                    state = cache.layers[decoder_layer].recurrent_states[0]
                    by_device.setdefault(state.device, []).append(index)
                groups = list(by_device.values())
            saved_states = {}
            for group in groups:
                group_gdn_layers = [selected_layers[index] for index in group]
                group_decoder_layers = [decoder_layers[index] for index in group]
                group_key = tuple(group_gdn_layers)
                states = states_from_cache(cache, group_decoder_layers, heads).detach()
                if start == 0:
                    validate_orientation([modules[index] for index in group], states, heads)
                metrics, current_vh = matrix_metrics(
                    states,
                    previous_states.get(group_key),
                    previous_vh.get(group_key),
                    svd_driver=svd_driver,
                )
                table = pa.Table.from_pydict(
                    rows(
                        metrics,
                        record=record,
                        position=position,
                        length=length,
                        gdn_layers=group_gdn_layers,
                        decoder_layers=group_decoder_layers,
                        heads=heads,
                    )
                )
                if writer is None:
                    writer = pq.ParquetWriter(output, table.schema, compression="zstd")
                writer.write_table(table)
                previous_states[group_key] = states.clone()
                previous_vh[group_key] = current_vh
                if capture_matrices and position in matrix_positions:
                    saved_states.update(
                        zip(group_gdn_layers, states.cpu(), strict=True)
                    )
            if capture_matrices and position in matrix_positions:
                save_matrix_snapshot(
                    matrix_dir,
                    record=record,
                    position=position,
                    selected_layers=selected_layers,
                    decoder_layers=decoder_layers,
                    heads=heads,
                    saved_states=saved_states,
                )
            start = position
    finally:
        if writer is not None:
            writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-head GDN state dynamics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="timestamped artifacts/state_dynamics run directory",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--heads", type=int, nargs="+")
    parser.add_argument(
        "--device-map",
        choices=("single", "auto"),
        default="single",
        help="put the model on visible GPU 0 or let Accelerate split it",
    )
    parser.add_argument("--every-token", action="store_true")
    parser.add_argument("--matrix-texts", type=int, default=5)
    parser.add_argument(
        "--svd-driver",
        choices=("gesvdj", "gesvda", "gesvd"),
        default="gesvdj",
        help="cuSOLVER driver; gesvda is approximate and much faster",
    )
    args = parser.parse_args()
    sequences = args.run_dir / "inputs" / "sequences.pt"
    output_dir = args.run_dir / "intermediate" / "state"

    artifact = torch.load(sequences, weights_only=True)
    device_map = {"": 0} if args.device_map == "single" else "auto"
    model, _ = load_qwen(artifact["model"], dtype=torch.float16, device_map=device_map)
    model.eval()
    found = gdn_layers(model)
    selected_layers = args.layers or list(range(len(found)))
    modules = [found[index][1] for index in selected_layers]
    decoder_layers = [module.layer_idx for module in modules]
    all_heads = modules[0].num_v_heads
    heads = args.heads or list(range(all_heads))
    if max(selected_layers) >= len(found) or min(selected_layers) < 0:
        raise ValueError("unknown GDN layer ordinal")
    if max(heads) >= all_heads or min(heads) < 0:
        raise ValueError("unknown value-head index")

    source_ids = list(dict.fromkeys(row["text_id"] for row in artifact["records"]))
    assigned = set(source_ids[args.shard_index :: args.shard_count])
    records = [row for row in artifact["records"] if row["text_id"] in assigned]
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir = output_dir / "matrices"
    matrix_dir.mkdir(exist_ok=True)
    device = model.get_input_embeddings().weight.device
    source_indices = {source_id: index for index, source_id in enumerate(source_ids)}
    for record_index, record in enumerate(records, 1):
        output = output_dir / f"metrics-{args.shard_index:02d}-{record_index:03d}.parquet"
        extract_record(
            model,
            record,
            output,
            device=device,
            selected_layers=selected_layers,
            modules=modules,
            decoder_layers=decoder_layers,
            heads=heads,
            every_token=args.every_token,
            capture_matrices=source_indices[record["text_id"]] < args.matrix_texts,
            matrix_dir=matrix_dir,
            svd_driver=args.svd_driver,
        )
        print(
            f"{record_index}/{len(records)} {record['text_id']} "
            f"{record['data_type']} length={len(record['input_ids'])}"
        )

    metadata = {
        "model": artifact["model"],
        "dataset": artifact["dataset"],
        "sequence_length": artifact["length"],
        "every_token": args.every_token,
        "position_rule": (
            "every token"
            if args.every_token
            else (
                "tokens 1..64; every 8 through 256; every 16 through 1024; "
                "every 32 through 4096; every 64 thereafter; plus final token"
            )
        ),
        "state_orientation": "[GDN layer, value head, key dimension, value dimension]",
        "right_subspace_axis": "value dimension (matrix columns)",
        "selected_gdn_layers": selected_layers,
        "decoder_layers": decoder_layers,
        "heads": heads,
        "matrix_shape": [modules[0].head_k_dim, modules[0].head_v_dim],
        "device_map": args.device_map,
        "fast_kernels": all(
            importlib.util.find_spec(package) is not None
            for package in ("fla", "causal_conv1d")
        ),
        "svd_driver": args.svd_driver,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "sequences": len(records),
    }
    (output_dir / f"metadata-{args.shard_index:02d}.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
