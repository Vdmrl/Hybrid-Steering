"""Run the cheap GPU smoke test for repaired candidate features."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import torch
from hybrid_steering.cache import recurrent_tensor
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).parents[2]
LEGACY_RUNNER = ROOT / "experiments" / "style-singleton-screen" / "run.py"
spec = importlib.util.spec_from_file_location("style_screen_runner", LEGACY_RUNNER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {LEGACY_RUNNER}")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

FEATURES = ("humorous", "numbered_list", "technical", "persuasive")
PROMPTS = [
    "How should a team decide whether to postpone a software release?",
    "Explain how a household can reduce its electricity use.",
    "What should a student do when two sources disagree?",
    "How can a small shop improve its inventory planning?",
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("run", "summary"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--feature", choices=FEATURES)
    parser.add_argument("--rank", choices=("full", "rank1", "rank4"), default="rank4")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--clamp-beta", type=float, default=1.0)
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tag", default="smoke")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def tensor_map(path: Path) -> dict[int, torch.Tensor]:
    return {
        int(key.removeprefix("layer_")): value
        for key, value in load_file(path, device="cpu").items()
    }


def prompts(path: Path | None, limit: int | None) -> list[tuple[str, str]]:
    if path is None:
        rows = [(f"smoke-{index:02d}", prompt) for index, prompt in enumerate(PROMPTS)]
    else:
        rows = []
        for row in read_jsonl(path):
            prompt = str(row.get("prompt") or row.get("text") or "").strip()
            if prompt:
                rows.append((str(row.get("id", f"prompt-{len(rows):03d}")), prompt))
    return rows if limit is None else rows[:limit]


def build_directions(out: Path, model: Any, tokenizer: Any) -> None:
    direction_dir = out / "directions"
    direction_dir.mkdir(parents=True, exist_ok=True)
    for feature in FEATURES:
        full_path = direction_dir / f"{feature}-full.safetensors"
        if full_path.exists():
            full = tensor_map(full_path)
        else:
            payload = json.loads(
                (out / "data" / f"{feature}_pairs.json").read_text(encoding="utf-8")
            )
            diffs = []
            for index, pair in enumerate(payload["pairs"]):
                positive = runner.extract_recurrent(
                    runner.prefill(model, tokenizer, pair["positive_text"])
                )
                negative = runner.extract_recurrent(
                    runner.prefill(model, tokenizer, pair["negative_text"])
                )
                diffs.append(runner.subtract_states(positive, negative))
                if (index + 1) % 16 == 0:
                    print(
                        f"direction {feature}: {index + 1}/{len(payload['pairs'])}",
                        flush=True,
                    )
            full = runner.mean_direction(diffs)
            save_file(
                {f"layer_{layer}": value.contiguous() for layer, value in full.items()},
                str(full_path),
            )
        for rank in (1, 4):
            rank_path = direction_dir / f"{feature}-rank{rank}.safetensors"
            if not rank_path.exists():
                low_rank = runner.low_rank(full, rank)
                save_file(
                    {
                        f"layer_{layer}": value.contiguous()
                        for layer, value in low_rank.items()
                    },
                    str(rank_path),
                )


def gram_inverse(flat_basis: torch.Tensor, ridge: float = 1e-6) -> torch.Tensor:
    gram = flat_basis @ flat_basis.T
    scale = torch.diag(gram).mean().clamp_min(1e-12)
    return torch.linalg.inv(
        gram + torch.eye(len(flat_basis), device=gram.device) * ridge * scale
    )


def make_clamp_runtime(
    cache: Any, direction: dict[int, torch.Tensor], alpha: float
) -> dict[int, dict[str, torch.Tensor]]:
    """Prepare a one-feature clamp; RSS is the identity for a singleton."""
    runtime = {}
    for layer, value in direction.items():
        state = recurrent_tensor(cache, layer)
        basis = value.to(state).float().flatten().unsqueeze(0)
        inverse = gram_inverse(basis)
        initial = inverse @ (basis @ state.float().flatten())
        runtime[layer] = {
            "basis": basis,
            "inverse": inverse,
            "target": initial + alpha,
        }
    return runtime


@torch.no_grad()
def clamp_cache(
    cache: Any, runtime: dict[int, dict[str, torch.Tensor]], beta: float
) -> None:
    for layer, values in runtime.items():
        state = recurrent_tensor(cache, layer)
        current = values["inverse"] @ (values["basis"] @ state.float().flatten())
        correction = values["target"] - current
        state.add_(
            (correction @ values["basis"]).reshape(state.shape).to(state), alpha=beta
        )


@torch.inference_mode()
def decode(
    model: Any,
    tokenizer: Any,
    cache: Any,
    max_new_tokens: int,
    clamp: tuple[dict[int, dict[str, torch.Tensor]], float] | None,
) -> str:
    output = None
    for token_id in tokenizer.encode("\n", add_special_tokens=False):
        output = model(
            input_ids=torch.tensor([[token_id]], device="cuda"),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = output.past_key_values
        if clamp is not None:
            clamp_cache(cache, *clamp)
    if output is None:
        raise RuntimeError("bridge text produced no tokens")
    logits = output.logits[:, -1, :]
    generated: list[int] = []
    eos = tokenizer.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    for _ in range(max_new_tokens):
        token_id = int(logits.argmax(dim=-1).item())
        if token_id in eos_ids:
            break
        generated.append(token_id)
        output = model(
            input_ids=torch.tensor([[token_id]], device="cuda"),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = output.past_key_values
        if clamp is not None:
            clamp_cache(cache, *clamp)
        logits = output.logits[:, -1, :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run(
    out: Path,
    model_id: str,
    feature: str | None,
    rank: str,
    alpha: float,
    max_new_tokens: int,
    prompts_file: Path | None,
    limit: int | None,
    tag: str,
    clamp_beta: float,
) -> None:
    if feature is None:
        raise ValueError("--feature is required for the sequential smoke")
    tokenizer, model = runner.load_model(model_id)
    build_directions(out, model, tokenizer)
    prompt_rows = prompts(prompts_file, limit)
    if not prompt_rows:
        raise ValueError("no prompts available")
    output = out / f"{tag}-{feature}-{rank}-alpha={alpha:g}.jsonl"
    done = {row["task_id"] for row in read_jsonl(output)}
    direction = tensor_map(out / "directions" / f"{feature}-{rank}.safetensors")
    for prompt_id, prompt in prompt_rows:
        target = runner.prefill(model, tokenizer, prompt)
        before = runner.snapshot_nonrecurrent(target)
        conditions: list[tuple[str, dict[int, torch.Tensor] | None, float]] = [
            ("baseline", None, 0.0),
            (
                f"{feature}:{rank}:alpha={alpha:g}:rss:clamp={clamp_beta:g}",
                direction,
                alpha,
            ),
        ]
        for condition, direction, strength in conditions:
            task_id = f"{tag}:{prompt_id}:{condition}"
            if task_id in done:
                continue
            cache = runner.clone_cache(target)
            clamp = None
            if direction is not None:
                runtime = make_clamp_runtime(cache, direction, strength)
                clamp_cache(cache, runtime, 1.0)
                clamp = (runtime, clamp_beta)
            runner.assert_nonrecurrent_unchanged(before, cache)
            response = decode(model, tokenizer, cache, max_new_tokens, clamp)
            append_jsonl(
                output,
                {
                    "task_id": task_id,
                    "prompt_id": prompt_id,
                    "scenario": prompt,
                    "condition": condition,
                    "response": response,
                    "generation": {
                        "rank": rank,
                        "alpha": strength,
                        "rss": True,
                        "clamp_beta": clamp_beta if direction is not None else None,
                    },
                },
            )
            print(task_id, flush=True)
        del target
        gc.collect()
        torch.cuda.empty_cache()


def has_loop(text: str) -> bool:
    words = re.findall(r"\w+", text.lower())
    chunks = [
        tuple(words[index : index + 6]) for index in range(max(0, len(words) - 5))
    ]
    return len(chunks) != len(set(chunks))


def summarize(
    out: Path, feature: str | None, rank: str, alpha: float, tag: str, count: int
) -> None:
    if feature is None:
        raise ValueError("--feature is required for the sequential smoke")
    rows = read_jsonl(out / f"{tag}-{feature}-{rank}-alpha={alpha:g}.jsonl")
    if not rows:
        raise RuntimeError("no smoke generations")
    report = []
    for row in rows:
        text = row["response"]
        report.append(
            {
                **row,
                "numbered_items": len(re.findall(r"(?m)^\s*\d+[.)]\s+", text)),
                "has_cjk": bool(re.search(r"[\u3400-\u9fff]", text)),
                "has_loop": has_loop(text),
                "word_count": len(text.split()),
            }
        )
    write_json(
        out / "smoke_summary.json",
        {
            "complete": len(rows) == count * 2,
            "n": len(rows),
            "rows": report,
        },
    )


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "run":
        run(
            args.output_dir,
            args.model,
            args.feature,
            args.rank,
            args.alpha,
            args.max_new_tokens,
            args.prompts_file,
            args.limit,
            args.tag,
            args.clamp_beta,
        )
    else:
        summarize(
            args.output_dir,
            args.feature,
            args.rank,
            args.alpha,
            args.tag,
            len(prompts(args.prompts_file, args.limit)),
        )


if __name__ == "__main__":
    main()
