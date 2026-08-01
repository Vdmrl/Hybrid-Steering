"""Prepare and run the six-feature singleton screen.

The experiment owns orchestration only. Direction arithmetic and cache
intervention are imported from the shared ``steering`` package; model loading
and deterministic decoding remain local runtime helpers because they are not
part of the shared package contract.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import subprocess
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "steering" / "src"))

from hybrid_steering import (
    add_direction,
    assert_nonrecurrent_unchanged,
    clone_cache,
    extract_recurrent,
    mean_direction,
    snapshot_nonrecurrent,
    subtract_states,
)

FEATURES = {
    "humorous": {
        "target": "humorous",
        "opposite": "factual/serious",
        "source": "FlickrStyle v0.9 humor captions; factual side is a local content-preserving rewrite because the public archive is unpaired",
    },
    "adjective_emphasis": {
        "target": "adjective emphasis",
        "opposite": "neutral adjective use",
        "source": "StylePTB AEM",
    },
    "action_emphasis": {
        "target": "verb/action emphasis",
        "opposite": "neutral action description",
        "source": "StylePTB VEM",
    },
    "technical": {
        "target": "technical vocabulary",
        "opposite": "basic vocabulary",
        "source": "synthetic-text-transformation-dataset columns; controlled local rewrites",
    },
    "persuasive": {
        "target": "persuasive style",
        "opposite": "informative style",
        "source": "synthetic-text-transformation-dataset columns; controlled local rewrites",
    },
    "narrative": {
        "target": "narrative style",
        "opposite": "analytical style",
        "source": "synthetic-text-transformation-dataset columns; controlled local rewrites",
    },
}

BASE_PROMPTS = [
    "How should a team decide whether to postpone a software release?",
    "What are the main causes of urban traffic congestion?",
    "Explain how a household can reduce its electricity use.",
    "What should a student do when two sources disagree?",
    "How can a small shop improve its inventory planning?",
    "Why do some materials rust faster than others?",
    "What is a sensible way to compare two job offers?",
    "How should a city prepare for a heat wave?",
    "What steps make a meeting more productive?",
    "Explain why backups are important for personal files.",
    "How can a researcher reduce measurement error?",
    "What should a driver check before a long trip?",
]


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=("prepare", "generate", "inputs", "summary", "all", "self-test"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--eval-prompts", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def load_model(model_id: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, local_files_only=True, trust_remote_code=True
    )
    errors = []
    for name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            cls = getattr(__import__("transformers", fromlist=[name]), name)
            model = cls.from_pretrained(
                model_id,
                dtype=torch.float16,
                device_map={"": "cuda"},
                local_files_only=True,
                trust_remote_code=True,
            )
            model.eval()
            return tokenizer, model
        except (ImportError, OSError, RuntimeError) as exc:
            errors.append(f"{name}: {exc!r}")
    raise RuntimeError("could not load local model: " + "; ".join(errors))


def chat_ids(tokenizer: Any, prompt: str, system: str) -> torch.Tensor:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    try:
        value = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        )
    except TypeError:
        value = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
    return value["input_ids"].to("cuda")


@torch.inference_mode()
def prefill(
    model: Any,
    tokenizer: Any,
    prompt: str,
    system: str = "Answer directly and naturally.",
) -> Any:
    return model(
        input_ids=chat_ids(tokenizer, prompt, system), use_cache=True, return_dict=True
    ).past_key_values


@torch.inference_mode()
def local_generate(
    model: Any, tokenizer: Any, prompt: str, system: str, max_new_tokens: int
) -> str:
    ids = chat_ids(tokenizer, prompt, system)
    output = model.generate(
        input_ids=ids,
        do_sample=False,
        use_cache=True,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0, ids.shape[1] :], skip_special_tokens=True).strip()


@torch.inference_mode()
def decode(model: Any, tokenizer: Any, cache: Any, max_new_tokens: int) -> str:
    output = None
    for token_id in tokenizer.encode("\n", add_special_tokens=False):
        output = model(
            input_ids=torch.tensor([[token_id]], device="cuda"),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = output.past_key_values
    if output is None:
        raise RuntimeError("bridge token is empty")
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
        logits = output.logits[:, -1, :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def ensure_sources(out: Path) -> Path:
    sources = out / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    style_root = sources / "StylePTB"
    if not style_root.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/lvyiwei1/StylePTB",
                str(style_root),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    for code in ("AEM", "VEM"):
        target = style_root / code / "train.tsv"
        if not target.exists():
            subprocess.run(
                [
                    sys.executable,
                    str(style_root / "single_transform_checkout.py"),
                    code,
                ],
                cwd=style_root,
                check=True,
                stdout=subprocess.DEVNULL,
            )
    flickr_zip = sources / "FlickrStyle_v0.9.zip"
    if not flickr_zip.exists():
        urllib.request.urlretrieve(
            "https://zhegan27.github.io/Papers/FlickrStyle_v0.9.zip", flickr_zip
        )
    flickr_root = sources / "FlickrStyle"
    if not flickr_root.exists():
        with zipfile.ZipFile(flickr_zip) as archive:
            archive.extractall(flickr_root)
    parquet = sources / "synthetic-text-transformation-dataset.parquet"
    if not parquet.exists():
        urllib.request.urlretrieve(
            "https://huggingface.co/datasets/sugiv/synthetic-text-transformation-dataset/resolve/main/data/train-00000-of-00001.parquet",
            parquet,
        )
    return sources


def load_style_pairs(sources: Path, code: str, count: int) -> list[dict[str, Any]]:
    rows = []
    for line in (
        (sources / "StylePTB" / code / "train.tsv")
        .read_text(encoding="utf-8-sig")
        .splitlines()
    ):
        parts = line.split("\t")
        if (
            len(parts) < 2
            or not parts[0].strip()
            or not parts[1].strip()
            or parts[0] == parts[1]
        ):
            continue
        rows.append(
            {
                "positive_text": parts[1].strip(),
                "negative_text": parts[0].strip(),
                "source_id": f"{code}:{len(rows)}",
            }
        )
        if len(rows) == count:
            break
    if len(rows) < count:
        raise RuntimeError(f"{code} supplied only {len(rows)} clean pairs")
    return rows


def load_humor_texts(sources: Path, count: int) -> list[str]:
    path = next((sources / "FlickrStyle").glob("**/humor/funny_train.txt"))
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ][:count]


def rewrite_pair(
    model: Any, tokenizer: Any, text: str, target: str, max_new_tokens: int = 96
) -> str:
    prompt = (
        f"Rewrite the caption as {target}. Preserve every fact, entity, event, and approximate length. "
        "Return only the rewritten text; do not mention this instruction.\n\n" + text
    )
    return local_generate(
        model,
        tokenizer,
        prompt,
        "You are a careful style-transfer editor.",
        max_new_tokens,
    )


def prepare(args_: argparse.Namespace) -> None:
    out = args_.output_dir
    sources = ensure_sources(out)
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_model(args_.model)
    humor_texts = load_humor_texts(sources, args_.pairs)
    humor = [
        {
            "source_id": f"FlickrStyle:{i}",
            "positive_text": text,
            "negative_text": rewrite_pair(
                model, tokenizer, text, "factual and serious"
            ),
        }
        for i, text in enumerate(humor_texts)
    ]
    write_json(
        data / "humorous_pairs.json",
        {
            "feature": "humorous",
            "source": FEATURES["humorous"]["source"],
            "pairs": humor,
        },
    )
    for code, feature in (("AEM", "adjective_emphasis"), ("VEM", "action_emphasis")):
        pairs = load_style_pairs(sources, code, args_.pairs)
        write_json(
            data / f"{feature}_pairs.json",
            {"feature": feature, "source": FEATURES[feature]["source"], "pairs": pairs},
        )
    frame = pd.read_parquet(
        sources / "synthetic-text-transformation-dataset.parquet",
        columns=[
            "input_text",
            "Tone",
            "Verbosity",
            "Style",
            "Complexity Level",
            "Emotion",
            "Purpose",
            "Vocabulary Range",
        ],
    )
    neutral = [
        str(x)
        for x in frame["input_text"].dropna().drop_duplicates().tolist()
        if len(str(x).split()) >= 8
    ][: args_.pairs]
    if len(neutral) < args_.pairs:
        neutral = (neutral * ((args_.pairs // len(neutral)) + 1))[: args_.pairs]
    specs = {
        "technical": ("technical and domain-precise", "basic and accessible"),
        "persuasive": ("persuasive but evidence-based", "informative and neutral"),
        "narrative": ("narrative and chronological", "analytical and structured"),
    }
    for feature, (target, opposite) in specs.items():
        pairs = []
        for index, text in enumerate(neutral):
            positive = rewrite_pair(model, tokenizer, text, target)
            negative = rewrite_pair(model, tokenizer, text, opposite)
            pairs.append(
                {
                    "source_id": f"synthetic:{index}",
                    "positive_text": positive,
                    "negative_text": negative,
                    "input_text": text,
                }
            )
        write_json(
            data / f"{feature}_pairs.json",
            {"feature": feature, "source": FEATURES[feature]["source"], "pairs": pairs},
        )
    prompts = [
        {"id": f"neutral-{i:02d}", "prompt": prompt}
        for i, prompt in enumerate(BASE_PROMPTS)
    ]
    write_json(data / "heldout_prompts.json", prompts)
    write_json(
        out / "rubrics.json",
        {"rubric_version": "exploratory-style-screen-1", "features": FEATURES},
    )
    review = {}
    rng = random.Random(20260801)
    for feature in FEATURES:
        rows = json.loads((data / f"{feature}_pairs.json").read_text(encoding="utf-8"))[
            "pairs"
        ]
        sample = rng.sample(rows, min(5, len(rows)))
        review[feature] = [
            {
                "source_id": row["source_id"],
                "positive": row["positive_text"],
                "negative": row["negative_text"],
                "length_ratio": round(
                    len(row["positive_text"].split())
                    / max(len(row["negative_text"].split()), 1),
                    3,
                ),
            }
            for row in sample
        ]
    write_json(out / "manual_review.json", review)
    del model, tokenizer
    torch.cuda.empty_cache()


def low_rank(direction: dict[int, torch.Tensor], rank: int) -> dict[int, torch.Tensor]:
    result = {}
    for layer, value in direction.items():
        matrix = value.squeeze(0).float()
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        k = min(rank, s.shape[-1])
        result[layer] = (
            (u[..., :k] * s[..., :k].unsqueeze(-2)) @ vh[..., :k, :]
        ).unsqueeze(0)
    return result


def load_pairs(out: Path, feature: str) -> list[dict[str, Any]]:
    return json.loads(
        (out / "data" / f"{feature}_pairs.json").read_text(encoding="utf-8")
    )["pairs"]


def generate(args_: argparse.Namespace) -> None:
    out = args_.output_dir
    (out / "directions").mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_model(args_.model)
    prompts = json.loads(
        (out / "data" / "heldout_prompts.json").read_text(encoding="utf-8")
    )[: args_.eval_prompts]
    directions: dict[str, dict[int, torch.Tensor]] = {}
    for feature in FEATURES:
        diffs = []
        for index, pair in enumerate(load_pairs(out, feature)):
            pos = extract_recurrent(
                prefill(
                    model,
                    tokenizer,
                    pair["positive_text"],
                    "Preserve this response's style in your internal state.",
                )
            )
            neg = extract_recurrent(
                prefill(
                    model,
                    tokenizer,
                    pair["negative_text"],
                    "Preserve this response's style in your internal state.",
                )
            )
            diffs.append(subtract_states(pos, neg))
            if (index + 1) % 8 == 0:
                print(
                    f"direction {feature} {index + 1}/{len(load_pairs(out, feature))}",
                    flush=True,
                )
        directions[feature] = mean_direction(diffs)
        save_file(
            {
                f"layer_{layer}": value.contiguous()
                for layer, value in directions[feature].items()
            },
            str(out / "directions" / f"{feature}-full.safetensors"),
        )
        rank1 = low_rank(directions[feature], 1)
        save_file(
            {f"layer_{layer}": value.contiguous() for layer, value in rank1.items()},
            str(out / "directions" / f"{feature}-rank1.safetensors"),
        )
    rows = out / "generations.jsonl"
    done = {row["task_id"] for row in jsonl(rows)}
    for p_index, item in enumerate(prompts):
        target = prefill(model, tokenizer, item["prompt"])
        before = snapshot_nonrecurrent(target)
        # Keep method labels separate from the hidden intervention metadata.
        conditions = [("baseline", None)] + [
            (f"{feature}:{kind}", (feature, kind))
            for feature in FEATURES
            for kind in ("full", "rank1")
        ]
        for name, choice in conditions:
            task_id = f"{item['id']}:{name}"
            if task_id in done:
                continue
            branch = clone_cache(target)
            if choice is not None:
                feature, kind = choice
                direction = load_file(
                    out / "directions" / f"{feature}-{kind}.safetensors", device="cpu"
                )
                add_direction(
                    branch,
                    {
                        int(key.removeprefix("layer_")): value
                        for key, value in direction.items()
                    },
                    args_.alpha,
                )
            assert_nonrecurrent_unchanged(before, branch)
            text = decode(model, tokenizer, branch, args_.max_new_tokens)
            append_jsonl(
                rows,
                {
                    "task_id": task_id,
                    "prompt_id": item["id"],
                    "scenario": item["prompt"],
                    "condition": name,
                    "response": text,
                },
            )
            print(f"generation {p_index + 1}/{len(prompts)} {name}", flush=True)
        del target
        gc.collect()
        torch.cuda.empty_cache()
    write_json(
        out / "generation_metadata.json",
        {
            "model": args_.model,
            "alpha": args_.alpha,
            "pairs": args_.pairs,
            "eval_prompts": len(prompts),
            "conditions": [
                "baseline",
                *(f"{f}:{k}" for f in FEATURES for k in ("full", "rank1")),
            ],
        },
    )


def prepare_judge_inputs(args_: argparse.Namespace) -> None:
    out = args_.output_dir
    records = jsonl(out / "generations.jsonl")
    by_prompt = defaultdict(list)
    for row in records:
        by_prompt[row["prompt_id"]].append(row)
    for feature in FEATURES:
        rows = []
        for prompt_id, values in by_prompt.items():
            rows.extend(
                {
                    "prompt_id": f"{prompt_id}:{feature}:{condition}",
                    "scenario": next(x["scenario"] for x in values),
                    "answers": [
                        {
                            "answer_id": "answer_0",
                            "text": next(
                                x["response"]
                                for x in values
                                if x["condition"] == condition
                            ),
                        }
                    ],
                    "metadata": {"condition": condition, "feature": feature},
                }
                for condition in ["baseline", f"{feature}:full", f"{feature}:rank1"]
            )
        (out / "judge-inputs").mkdir(parents=True, exist_ok=True)
        (out / "judge-inputs" / f"{feature}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
    quality = []
    for prompt_id, values in by_prompt.items():
        for row in values:
            quality.append(
                {
                    "prompt_id": f"{prompt_id}:quality:{row['condition']}",
                    "scenario": row["scenario"],
                    "answers": [{"answer_id": "answer_0", "text": row["response"]}],
                    "metadata": {
                        "condition": row["condition"],
                        "feature": "answer_quality",
                    },
                }
            )
    (out / "judge-inputs" / "answer_quality.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in quality) + "\n",
        encoding="utf-8",
    )


def summary(args_: argparse.Namespace) -> None:
    out = args_.output_dir
    compact = out / "compact-results.jsonl"
    results = (
        [
            json.loads(line)
            for line in compact.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if compact.exists()
        else []
    )
    by_feature: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in results:
        by_feature[row["feature"]][row.get("condition", "unknown")].append(
            float(row.get("expected_score", row.get("trait_score", 0)))
        )
    # Judge rows are compacted by the shell postprocessor below; this remains safe
    # when a provider run is partial.
    table = []
    for feature in FEATURES:
        values = by_feature.get(feature, {})
        baseline = sum(values.get("baseline", [0])) / max(
            len(values.get("baseline", [])), 1
        )
        full = sum(values.get(f"{feature}:full", [0])) / max(
            len(values.get(f"{feature}:full", [])), 1
        )
        rank = sum(values.get(f"{feature}:rank1", [0])) / max(
            len(values.get(f"{feature}:rank1", [])), 1
        )
        q = by_feature.get("answer_quality", {})
        q0 = sum(q.get("baseline", [0])) / max(len(q.get("baseline", [])), 1)
        qf = sum(q.get(f"{feature}:full", [0])) / max(
            len(q.get(f"{feature}:full", [])), 1
        )
        table.append(
            {
                "feature": feature,
                "source": FEATURES[feature]["source"],
                "full_delta": round(full - baseline, 4),
                "rank1_delta": round(rank - baseline, 4),
                "quality_delta": round(qf - q0, 4),
                "visual_check": "see manual_review.json and generated samples",
                "verdict": "PASS"
                if full - baseline >= 0.5 and qf - q0 >= -0.5
                else "RANK1_WEAK"
                if full - baseline >= 0.5
                else "FAIL",
            }
        )
    ranked = sorted(
        table, key=lambda row: (row["full_delta"], row["rank1_delta"]), reverse=True
    )
    write_json(
        out / "summary.json",
        {
            "experiment": "style-singleton-screen",
            "model": args_.model,
            "pairs_per_feature": args_.pairs,
            "heldout_prompts": args_.eval_prompts,
            "alpha": args_.alpha,
            "table": table,
            "ranking": [row["feature"] for row in ranked],
            "recommended_for_combination_with_russian_and_casualness": [
                row["feature"] for row in ranked[:3]
            ],
            "dangerous_pairs": [],
            "limitations": [
                "No human calibration for exploratory rubrics.",
                "FlickrStyle archive supplies humorous captions without paired factual captions; factual side was locally rewritten.",
            ],
        },
    )


def self_test() -> None:
    assert len(FEATURES) == 6
    assert all(
        set(value) == {"target", "opposite", "source"} for value in FEATURES.values()
    )
    print("style singleton screen self-test passed")


def main() -> None:
    parsed = args()
    parsed.output_dir.mkdir(parents=True, exist_ok=True)
    if parsed.phase == "self-test":
        self_test()
    elif parsed.phase == "prepare":
        prepare(parsed)
    elif parsed.phase == "generate":
        generate(parsed)
    elif parsed.phase == "inputs":
        prepare_judge_inputs(parsed)
    elif parsed.phase == "summary":
        summary(parsed)
    elif parsed.phase == "all":
        prepare(parsed)
        generate(parsed)
        prepare_judge_inputs(parsed)


if __name__ == "__main__":
    main()
