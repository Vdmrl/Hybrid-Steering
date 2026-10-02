"""Language steering on code tasks: does the language reach the code itself?

Each of the 50 prompts in ``prompts.json`` asks for a Python function and a 2-3
sentence explanation. Two steering methods push the answer toward a target
language; each answer is scored for how much of the prose is in the target
language and whether the language leaked into the code.

Methods (``method`` column; the strength is ``scale`` for both):

- ``gdn``: rank-1 clamp of every GDN head, from the first prompt token on and
  after every generated token. With ``(u, w, sigma)`` the head's top singular
  triple of the direction, ``S <- S - u (u^T S) + c * sigma * u w^T``, where
  ``c = scale * c_ref`` and ``c_ref`` is the natural recurrent-state norm on
  five of the prompts divided by the norm of all sigma. The clamp holds the
  state's component along ``u`` at a fixed value rather than adding to it.
- ``caa``: residual steering at one decoder layer, the same vector added to
  every position (prompt included): ``h <- h + scale * h_ref * r / |r|``, with
  ``r`` the target-minus-source mean of that layer's output at the last token
  over the same pairs as the direction, and ``h_ref`` the layer's mean
  activation norm on eight held-out questions (``holdout.json``).

Scores per answer (script-based, so ``--language`` must be ``ru``, ``zh``,
``ar`` or ``hi``):

- ``prose_share``: target-script letters over all letters outside code fences.
- ``code_leak_any``: 1 if code proper -- the code blocks with ``#`` comments,
  docstrings and every other string literal blanked -- still contains a
  target-script letter; ``code_leak_share`` is that script's share of its letters.
- ``has_code``, ``code_parses`` (``ast.parse`` accepts every block), ``rep5``
  (5-gram repetition of the answer tokens; above 0.1 the text is degenerate).

The ``en-*-gen`` pools behind ``results/`` are no longer on the Hub dataset's
``main``; see ``results/README.md`` for the revision that has them.

    uv run hybrid-direction --concept en-ru-gen --source en --target ru --pairs 100 \\
        --output runs/directions/en-ru-gen
    uv run python experiments/code-leak/run.py --direction runs/directions/en-ru-gen/direction \\
        --concept en-ru-gen --language ru --caa-layer 9 --output runs/code-leak/ru
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import tokenize
from pathlib import Path

import torch

from hybrid_steering import final_states, gdn_layers, load_direction, load_runtime
from hybrid_steering.direction import load_concept_pairs, read_pairs, target_and_source
from hybrid_steering.runner import Runner
from hybrid_steering.runtime import chat_prompts, write_jsonl

PROMPTS = json.loads((Path(__file__).parent / "prompts.json").read_text(encoding="utf-8"))
# eight generic questions, never used to fit a direction: the norm reference for CAA
HOLDOUT = json.loads((Path(__file__).parent / "holdout.json").read_text(encoding="utf-8"))
SCRIPTS = {
    "ru": r"[Ѐ-ӿ]",
    "zh": r"[㐀-䶿一-鿿豈-﫿]",
    "ar": r"[ؠ-يٮ-ۓۺ-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-ﻼ]",
    # letters only: vowel signs and viramas are combining marks, not letters
    "hi": r"[ऄ-हऽॐक़-ॡॱ-ॿ]",
}
LETTER = re.compile(r"[^\W\d_]")
FENCE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL)


# ---------------------------------------------------------------- scoring
def blank(lines: list[str], start: tuple[int, int], end: tuple[int, int]) -> None:
    """Replace a (row, column) span with spaces; rows are 1-based, columns in characters."""
    for row in range(start[0], end[0] + 1):
        line = lines[row - 1]
        a = start[1] if row == start[0] else 0
        b = end[1] if row == end[0] else len(line)
        lines[row - 1] = line[:a] + " " * (b - a) + line[b:]


def blank_comments(code: str) -> str:
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # unparsable block: cut each line at the first # outside quotes
        out = []
        for line in code.split("\n"):
            quote, cut = None, len(line)
            for index, character in enumerate(line):
                if quote:
                    quote = None if character == quote else quote
                elif character in "\"'":
                    quote = character
                elif character == "#":
                    cut = index
                    break
            out.append(line[:cut] + " " * (len(line) - cut))
        return "\n".join(out)
    lines = code.split("\n")
    for token in tokens:
        if token.type == tokenize.COMMENT:
            blank(lines, token.start, token.end)
    return "\n".join(lines)


def code_proper(code: str) -> str:
    """The block with comments, docstrings and string literals replaced by spaces."""
    text = blank_comments(code)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        text = re.sub(
            r"(\"\"\"|\'\'\')(.*?)\1", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S
        )
        return re.sub(r"\"[^\"\n]*\"|'[^'\n]*'", lambda m: " " * len(m.group(0)), text)
    source = text.split("\n")

    def column(row: int, byte_offset: int) -> int:
        # ast columns are UTF-8 byte offsets into the text it parsed
        return len(source[row - 1].encode()[:byte_offset].decode(errors="ignore"))

    lines = list(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr) or (
            isinstance(node, ast.Constant) and isinstance(node.value, str)
        ):
            blank(
                lines,
                (node.lineno, column(node.lineno, node.col_offset)),
                (node.end_lineno, column(node.end_lineno, node.end_col_offset)),
            )
    return "\n".join(lines)


def share(text: str, script: re.Pattern) -> float | None:
    letters = LETTER.findall(text)
    return len(script.findall(text)) / len(letters) if letters else None


def score(text: str, script: re.Pattern) -> dict:
    blocks = FENCE.findall(text)
    row = {"prose_share": share(FENCE.sub(" ", text), script) or 0.0, "has_code": int(bool(blocks))}
    if not blocks:
        return {**row, "code_leak_any": None, "code_leak_share": None, "code_parses": 0}
    proper = "\n".join(code_proper(block) for block in blocks)
    parses = 1
    for block in blocks:
        try:
            ast.parse(block)
        except SyntaxError:
            parses = 0
    return {
        **row,
        "code_leak_any": int(bool(script.search(proper))),
        "code_leak_share": share(proper, script) or 0.0,
        "code_parses": parses,
    }


def rep5(ids: list[int]) -> float:
    if len(ids) < 10:
        return 0.0
    grams = [tuple(ids[i : i + 5]) for i in range(len(ids) - 4)]
    return 1.0 - len(set(grams)) / len(grams)


# ---------------------------------------------------------------- steering
def rank1(direction: dict[int, torch.Tensor]):
    """Per layer and head: top singular triple of the direction."""
    out = {}
    for layer, matrix in direction.items():
        left, singular, right = torch.linalg.svd(matrix.float(), full_matrices=False)
        out[layer] = (left[..., :, 0], right[..., 0, :], singular[..., 0])
    return out


def clamp(cache, triples, factor: float) -> None:
    for layer, (u, w, sigma) in triples.items():
        state = cache.layers[layer].recurrent_states[0]
        uu, ww, ss = (x.to(state.device, state.dtype) for x in (u, w, sigma))
        along = torch.einsum("hk,bhkv->bhv", uu, state)
        state.sub_(torch.einsum("hk,bhv->bhkv", uu, along))
        state.add_(torch.einsum("hk,hv->hkv", uu, ww)[None] * (factor * ss)[None, :, None, None])


@torch.no_grad()
def generate(model, tokenizer, prompts, max_new, step=None):
    """Greedy decode, feeding the prompt one token at a time so ``step(cache)``
    runs after every forward call from position 0; ``step=None`` still decodes
    token by token, so steered and unsteered runs share the same numerics."""
    from transformers import DynamicCache

    encoded = tokenizer(chat_prompts(tokenizer, prompts), return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    ids, mask = encoded.input_ids.to(device), encoded.attention_mask.to(device)
    cache = DynamicCache(config=model.config)
    logits = None
    for position in range(ids.shape[1]):
        out = model(
            input_ids=ids[:, position : position + 1],
            attention_mask=mask[:, : position + 1],
            past_key_values=cache,
            use_cache=True,
        )
        if step:
            step(cache)
        logits = out.logits[:, -1]
    done = torch.zeros(len(prompts), dtype=torch.bool, device=device)
    tokens: list[list[int]] = [[] for _ in prompts]
    for _ in range(max_new):
        chosen = logits.argmax(-1)
        for row, token in enumerate(chosen.tolist()):
            if not done[row]:
                if token == tokenizer.eos_token_id:
                    done[row] = True
                else:
                    tokens[row].append(token)
        if done.all():
            break
        feed = chosen.masked_fill(done, tokenizer.pad_token_id)[:, None]
        mask = torch.cat([mask, (~done).long()[:, None]], dim=1)
        out = model(input_ids=feed, attention_mask=mask, past_key_values=cache, use_cache=True)
        if step:
            step(cache)
        logits = out.logits[:, -1]
    return tokens


class LayerOutput:
    """Last-token output of one decoder layer, or an additive edit to all of it."""

    def __init__(self, model, layer: int):
        self.last, self.add = None, None
        self.handle = model.model.layers[layer].register_forward_hook(self.hook, with_kwargs=True)

    def hook(self, module, args, kwargs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self.last = hidden.detach().float()
        if self.add is None:
            return output
        hidden = hidden + self.add.to(hidden.dtype)
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden


@torch.no_grad()
def residual_direction(model, tokenizer, capture, pairs, prompts):
    """Unit target-minus-source last-token direction, and the layer's mean norm on prompts."""
    device = next(model.parameters()).device
    total = None
    for sign, texts in ((1.0, [t for t, _ in pairs]), (-1.0, [s for _, s in pairs])):
        for text in texts:
            model(**tokenizer(text, return_tensors="pt").to(device))
            vector = sign * capture.last[0, -1] / len(texts)
            total = vector if total is None else total + vector
    norms = []
    for prompt in chat_prompts(tokenizer, prompts):
        model(**tokenizer(prompt, return_tensors="pt").to(device))
        norms.append(capture.last[0].norm(dim=-1).mean())
    return total / total.norm(), float(torch.stack(norms).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--concept", required=True, help="pairs for the CAA vector, Hub directory")
    parser.add_argument("--jsonl", type=Path, help="local pairs; skips the Hub download")
    parser.add_argument("--language", required=True, choices=sorted(SCRIPTS))
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--caa-layer", type=int, required=True)
    parser.add_argument("--gdn-scales", type=float, nargs="+", default=[0.1, 0.3, 0.5, 1.0])
    parser.add_argument("--caa-scales", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--prompts", type=int, default=0, help="0 uses all 50")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    args = parser.parse_args()
    script = re.compile(SCRIPTS[args.language])
    prompts = PROMPTS[: args.prompts or None]
    model, tokenizer = load_runtime(args.model)
    tokenizer.padding_side = "left"

    direction, manifest, _, _ = load_direction(args.direction)
    triples = rank1(direction)
    runner = Runner(model, tokenizer, gdn_layers(model), normalize=False)
    natural = final_states(runner, chat_prompts(tokenizer, prompts[:5]))
    natural_norm = float(
        torch.stack([states.float().pow(2).sum((1, 2, 3)) for states in natural.values()])
        .sum(0)
        .sqrt()
        .mean()
    )
    c_ref = natural_norm / float(torch.cat([t[2].flatten() for t in triples.values()]).norm())

    rows_in = read_pairs(args.jsonl) if args.jsonl else load_concept_pairs(args.concept)
    pairs = [target_and_source(row) for row in rows_in[: args.pairs]]
    capture = LayerOutput(model, args.caa_layer)
    unit, h_ref = residual_direction(model, tokenizer, capture, pairs, HOLDOUT)

    conditions = [("none", 0.0, None)]
    conditions += [
        ("gdn", s, lambda cache, f=s * c_ref: clamp(cache, triples, f)) for s in args.gdn_scales
    ]
    conditions += [("caa", s, None) for s in args.caa_scales]
    rows = []
    for method, scale, step in conditions:
        capture.add = (scale * h_ref) * unit if method == "caa" else None
        tokens = generate(model, tokenizer, prompts, args.max_new_tokens, step)
        for index, answer_tokens in enumerate(tokens):
            text = tokenizer.decode(answer_tokens, skip_special_tokens=True)
            rows.append(
                {
                    "method": method,
                    "scale": scale,
                    "prompt_index": index,
                    "language": args.language,
                    "target": manifest.target,
                    **score(text, script),
                    "rep5": rep5(answer_tokens),
                }
            )
        print(f"{method} scale={scale}: done", flush=True)
    capture.handle.remove()
    write_jsonl(args.output / "rows.jsonl", rows)
    (args.output / "calibration.json").write_text(
        json.dumps({"c_ref": c_ref, "caa_layer": args.caa_layer, "h_ref": h_ref}) + "\n"
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
