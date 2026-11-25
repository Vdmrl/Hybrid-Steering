import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_job_configs import (
    CAA_SHAPE,
    HYS_SHAPE,
    PACKAGE,
    calibration,
    load_directions,
    read,
    read_rows,
    validate_banks,
)
from interventions import (
    FEATURES,
    LAYERS,
    RecurrentWrite,
    apply_clamp,
    caa_phases,
    hys_coefficients,
    make_clamp_runtime,
    rss_sum,
)
from mock_runtime import MockTokenizer, mock_backend, smoke_config
from native_runtime import generate
from run_experiment import resolve_job, run_job

torch.set_num_threads(2)


@pytest.mark.parametrize("piece", ([0.0, 0.0], [1.0, -2.0], [1e-16, 0.0]))
def test_rss_singleton_exactly_one(piece):
    x = torch.tensor(piece)
    out, gamma = rss_sum([x])
    assert gamma == 1.0 and torch.equal(out, x)


@pytest.mark.parametrize(
    "pieces",
    [
        [[1.0, 2.0], [0.0, 3.0]],
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [0.0, -2.0]],
        [[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [-2.0, 0.0], [0.0, 1e-14]],
    ],
)
def test_rss_eq27(pieces):
    ts = [torch.tensor(p) for p in pieces]
    raw = torch.stack(ts).sum(0)
    denom = max(math.hypot(*raw.tolist()), 1e-12)
    expected = math.sqrt(sum(math.fsum(v * v for v in p) for p in ts)) / denom
    scaled, actual = rss_sum(ts)
    assert actual == pytest.approx(expected)
    assert torch.isfinite(scaled).all() and math.isfinite(actual)
    torch.testing.assert_close(scaled, raw * expected, atol=1e-5, rtol=1e-6)
    raw_out, raw_gamma = rss_sum(ts, "raw")
    torch.testing.assert_close(raw_out, raw)
    assert raw_gamma == 1.0


def cache(state):
    return SimpleNamespace(layers=[SimpleNamespace(recurrent_states=[state.clone()])])


@pytest.mark.parametrize("tag", list(FEATURES))
def test_no_gram_singleton_parity(tag):
    basis = {tag: {0: torch.tensor([[[[0.8, -0.9, 0.5]]]])}}
    initial = torch.tensor([[[[0.2, 0.8, -0.3]]]])
    correction = torch.tensor([[[[1.0, -0.5, 0.9]]]])
    outputs = []
    for ablated in (False, True):
        c = cache(initial)
        runtime = make_clamp_runtime(c, basis, {0: {tag: 2.5}}, no_gram=ablated)
        apply_clamp(c, runtime, {tag: 1.0}, first=True)
        state = c.layers[0].recurrent_states[0]
        first = state.clone()
        state.add_(correction)
        apply_clamp(c, runtime, {tag: 0.25})
        outputs.append(torch.stack([first, state]))
    torch.testing.assert_close(*outputs, atol=1e-5, rtol=0)


def test_no_gram_expected_skew_nonorthogonal():
    basis = {
        m: {0: x.reshape(1, 1, 1, -1)}
        for m, x in {
            "N": torch.tensor([1.0, 0.2]),
            "P": torch.tensor([0.5, 1.0]),
        }.items()
    }
    d = {0: {"N": 1.5, "P": -1.0}}
    base = torch.tensor([[[[0.2, 0.1]]]])
    perturb = torch.tensor([[[[1.3, 0.6]]]])
    u = torch.stack([d[0][0].flatten() for d in basis.values()])
    expected_first = base.flatten() + torch.tensor(list(d[0].values())) @ u
    full = cache(base)
    ablated = cache(base)
    for c, no_gram in ((full, False), (ablated, True)):
        runtime = make_clamp_runtime(c, basis, d, no_gram=no_gram)
        apply_clamp(c, runtime, {"N": 1.0, "P": 1.0}, first=True)
        state = c.layers[0].recurrent_states[0]
        torch.testing.assert_close(state.flatten(), expected_first, atol=1e-5, rtol=0)
        state.add_(perturb)
        apply_clamp(c, runtime, {"N": 1.0, "P": 1.0})
    s_full = full.layers[0].recurrent_states[0].flatten()
    s_sep = ablated.layers[0].recurrent_states[0].flatten()
    torch.testing.assert_close(u @ s_full, u @ expected_first, atol=1e-5, rtol=0)
    shifted = expected_first + perturb.flatten()
    expected_sep = shifted - ((u @ perturb.flatten()) / u.square().sum(1)) @ u
    torch.testing.assert_close(s_sep, expected_sep, atol=1e-5, rtol=0)
    assert not torch.allclose(u @ s_sep, u @ expected_first)


def test_no_clamp_first_write_only_and_fresh_cache():
    model, tok, device, dirs = mock_backend()
    alpha = {"N": 2.5, "P": 1.5}
    deltas, _ = hys_coefficients(dirs["hys"], alpha, 0.75, "no_clamp")
    outputs = []
    for _ in range(2):
        writer = RecurrentWrite(dirs["hys"], alpha, deltas, "no_clamp", device)
        result = generate(model, tok, device, "hi", "sys", 32, recurrent=writer)
        assert writer.writes == 1 and writer.audit_first[0]["phase"] == "bulk"
        assert result["audit"]["decode_forward_count"] == 31
        outputs.append(result["token_ids"])
    assert outputs[0] == outputs[1]
    assert len(model.forward_records) == 66
    assert torch.count_nonzero(model.forward_records[0]["incoming"]) == 0
    assert torch.count_nonzero(model.forward_records[33]["incoming"]) == 0
    assert torch.count_nonzero(model.forward_records[1]["incoming"]) > 0


@pytest.mark.parametrize("mode", ["rss", "raw"])
def test_caa24_hooks_skip_bulk_change_bridge_and_decode(mode):
    model, tok, device, dirs = mock_backend()
    alpha = {"N": 1.0, "P": 0.5}
    phases, stats = caa_phases(dirs["caa24"], alpha, 1.0, "natural", mode, False, device)
    observed = [[], []]
    untouched = {i: [] for i in set(range(32)) - set(LAYERS)}
    handles = []
    for i, block in enumerate(model.blocks):
        if i in untouched:
            handles.append(
                block.register_forward_hook(
                    lambda m, inp, out, i=i: untouched[i].append(out.shape[1])
                )
            )
        if i == 0:

            def record(m, inp, out):
                # Hooks added later see this layer's result re-enter at the next layer.
                observed[0].append(out.detach().clone())

            handles.append(block.register_forward_hook(record))
        if i == 1:

            def next_input(m, inputs):
                observed[1].append(inputs[0].detach().clone())

            handles.append(block.register_forward_pre_hook(next_input))
    result = generate(model, tok, device, "hi", "sys", 32, caa=phases)
    for h in handles:
        h.remove()
    assert set(result["audit"]["caa_layer_calls"]) == set(LAYERS)
    assert set(result["audit"]["caa_layer_calls"].values()) == {32}
    assert all(s == [3] + [1] * 32 for s in untouched.values())
    torch.testing.assert_close(observed[0][0], observed[1][0])
    for j in range(1, 33):
        torch.testing.assert_close(
            observed[1][j] - observed[0][j],
            phases[1][0].reshape(1, 1, -1),
            atol=1e-7,
            rtol=1e-5,
        )
    if mode == "raw":
        assert stats[0]["min"] == stats[0]["max"] == 1


def test_windows_change_active_set_and_gamma():
    model, tok, device, dirs = mock_backend()
    alpha = {"N": 1.0, "P": 0.5}
    phases, stats = caa_phases(dirs["caa24"], alpha, 0.75, "natural", "rss", True, device)
    assert list(phases) == [1, 29, 33] and not phases[33]
    assert [e["active"] for e in stats] == ["NP", "P", ""]
    assert stats[1]["min"] == stats[1]["max"] == 1
    for i in LAYERS:
        pieces = [dirs["caa24"][f][i] * (0.75 * a) for f, a in alpha.items()]
        want, scale = rss_sum(pieces)
        torch.testing.assert_close(phases[1][i], want)
        assert stats[0]["per_layer"][i] == scale
        torch.testing.assert_close(phases[29][i], pieces[1])
    # Synthetic mock verifies the exact 28/32 boundary at three adjacent forwards.
    # The packaged CLI smoke separately remains capped at 32 tokens.
    for tag, count in (("N", 28), ("P", 32)):
        only, _ = caa_phases(dirs["caa24"], {tag: 1.0}, 1.0, "natural", "rss", True, device)
        result = generate(model, tok, device, "hi", "sys", 34, caa=only)
        assert set(result["audit"]["caa_layer_calls"].values()) == {count}


def test_natural_and_unit_keep_the_intended_dose():
    _, _, device, dirs = mock_backend()
    caa = dirs["caa24"]
    natural, _ = caa_phases(caa, {"N": 2.0}, 0.75, "natural", "rss", False, device)
    unit, _ = caa_phases(caa, {"N": 2.0}, 0.75, "unit", "rss", False, device)
    for i in LAYERS:
        torch.testing.assert_close(natural[1][i], caa["N"][i] * 1.5)
        assert unit[1][i].norm().item() == pytest.approx(1.5 / math.sqrt(24))


@pytest.mark.parametrize(("method", "shape"), [("caa24", CAA_SHAPE), ("hys", HYS_SHAPE)])
def test_real_9b_directions_shape_sha_finite(method, shape):
    actual = load_directions(method)
    assert set(actual) == set(FEATURES)
    for vectors in actual.values():
        assert tuple(vectors) == LAYERS and all(
            v.shape == shape and v.dtype == torch.float32 for v in vectors.values()
        )


def test_job_counts_grid_and_disjoint_banks():
    names = ("selection8", "independent16", "heldout32", "simple50")
    banks = {name: read_rows(PACKAGE / "banks" / f"{name}.jsonl") for name in names}
    validate_banks(banks)
    grid = calibration(load_directions("caa24"))
    selected = {m: block["grid"][2] for m, block in grid.items()}
    counts = [
        (240, {"rss": 30}),
        (80, {"rss": 5}),
        (1216, {"no_rss": 12, "no_gram": 12, "no_clamp": 12, "full": 2}),
        (1440, {"rss": 33, "raw": 12}),
        (300, {"full": 3, "rss": 3}),
    ]
    for f, (n, modes) in zip(sorted((PACKAGE / "jobs").glob("job*.json")), counts, strict=True):
        cfg = read(f)
        assert cfg["expected_answers"] == n
        assert Counter(c["mode"] for c in cfg["conditions"]) == modes
        resolve_job(cfg, banks[cfg["bank"]], selected)


def test_frozen_calibration_grid_and_doses():
    expected = {
        "N": [0.05, 0.09, 0.16, 0.28, 0.50, 0.85],
        "F": [0.03, 0.05, 0.085, 0.14, 0.24, 0.40],
        "T": [0.04, 0.07, 0.12, 0.20, 0.33, 0.55],
        "R": [0.04, 0.07, 0.13, 0.23, 0.40, 0.70],
        "P": [0.08, 0.15, 0.28, 0.52, 0.95, 1.75],
    }
    v = load_directions("caa24")
    generated = calibration(v)
    saved = read(PACKAGE / "jobs/calibration_grid.json")
    job = read(PACKAGE / "jobs/job1_caa24_calib_selection8.json")
    assert job["norm_mode"] == "natural" and job["use_windows"] is False
    for tag, grid in expected.items():
        assert generated[tag]["grid"] == saved[tag]["grid"] == grid
        assert [c["alpha"][tag] for c in job["conditions"] if c["active"] == tag] == grid
        doses = saved[tag]["dose_l2_at_grid_endpoints"]
        assert set(doses) == {"0", "8", "16", "30"}
        for layer, dose in doses.items():
            norm = float(v[tag][int(layer)].double().norm())
            assert dose == pytest.approx(
                {
                    "alpha_min": grid[0],
                    "alpha_max": grid[-1],
                    "l2_at_min": grid[0] * norm,
                    "l2_at_max": grid[-1] * norm,
                }
            )


def test_resume_and_joint_judge_schema(tmp_path):
    cfg, prompts = smoke_config()
    job = resolve_job(cfg, prompts)
    run_job(job, prompts, tmp_path, mock=True)
    raw = tmp_path / "generations.jsonl"
    complete = read_rows(raw)
    assert len(complete) == 12 and all(row["audit"]["finite_logits"] for row in complete)
    # Simulate a stopped worker with the first two intact records.
    raw.write_text("".join(json.dumps(r) + "\n" for r in complete[:2]), encoding="utf-8")
    run_job(job, prompts, tmp_path, resume=True, mock=True)
    resumed = read_rows(raw)
    assert len(resumed) == 12 and len({r["answer_id"] for r in resumed}) == 12
    assert resumed[:2] == complete[:2]
    assert [r["token_ids"] for r in resumed] == [r["token_ids"] for r in complete]
    exported = read_rows(tmp_path / "judge_input.jsonl")
    assert len(exported) == len({r["prompt_id"] for r in exported}) == 2
    assert all(len(r["answers"]) == 6 for r in exported)
    assert all(set(r) == {"prompt_id", "scenario", "answers"} for r in exported)
    assert all(set(a) == {"answer_id", "text"} for r in exported for a in r["answers"])
    assert {a["answer_id"] for r in exported for a in r["answers"]} == {
        r["answer_id"] for r in complete
    }
    assert read(tmp_path / "manifest.json")["status"] == "complete"
    with pytest.raises(ValueError, match="identical"):
        run_job({**job, "max_new_tokens": 16}, prompts, tmp_path, resume=True, mock=True)


def test_tiny_real_qwen35_bf16_cache_contract():
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    torch.manual_seed(20260911)
    cfg = Qwen3_5Config(
        text_config={
            "vocab_size": 32,
            "hidden_size": 64,
            "intermediate_size": 96,
            "num_hidden_layers": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "eos_token_id": 0,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.25,
                "mrope_section": [2, 1, 1],
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "num_heads": 2,
            "out_hidden_size": 64,
            "intermediate_size": 64,
            "num_position_embeddings": 16,
        },
    )
    model = Qwen3_5ForConditionalGeneration(cfg).eval().bfloat16()
    tok = MockTokenizer()
    device = torch.device("cpu")
    caa = {"N": {i: torch.ones(64) * 0.01 for i in LAYERS}}
    hys = {"N": {i: torch.ones(1, 2, 4, 4) * 0.01 for i in LAYERS}}
    phases, _ = caa_phases(caa, {"N": 1.0}, 1.0, "natural", "rss", False, device)
    for mode in ("full", "no_rss", "no_gram", "no_clamp"):
        deltas, _ = hys_coefficients(hys, {"N": 2.5}, 1.0, mode)
        state = RecurrentWrite(hys, {"N": 2.5}, deltas, mode, device)
        result = generate(model, tok, device, "hi", "sys", 4, recurrent=state)
        assert result["audit"]["finite_logits"]
        assert state.writes == (
            1 if mode == "no_clamp" else 2 + result["audit"]["decode_forward_count"]
        )
    result = generate(model, tok, device, "hi", "sys", 4, caa=phases)
    assert set(result["audit"]["caa_layer_calls"]) == set(LAYERS)
