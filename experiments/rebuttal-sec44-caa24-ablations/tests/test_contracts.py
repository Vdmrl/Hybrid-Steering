import json
import math
import os
import subprocess
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
    LAYER_SETS,
    LAYERS,
    RecurrentWrite,
    Step,
    apply_clamp,
    caa_hook,
    caa_phases,
    hys_coefficients,
    make_clamp_runtime,
    rss_sum,
)
from jobs import run_parallel
from mock_runtime import MockTokenizer, mock_backend, smoke_config
from native_runtime import generate
from run_experiment import export_judge, opaque_id, plan, resolve_job, run_job, scheduled_tasks

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
        assert tuple(vectors) == (tuple(range(32)) if method == "caa24" else LAYERS) and all(
            v.shape == shape and v.dtype == torch.float32 for v in vectors.values()
        )
        if method == "caa24":
            assert all(torch.isfinite(v).all() and v.double().norm() > 0 for v in vectors.values())


def test_job_counts_grid_and_disjoint_banks():
    names = ("selection8", "independent16", "heldout32", "simple50")
    banks = {name: read_rows(PACKAGE / "banks" / f"{name}.jsonl") for name in names}
    validate_banks(banks)
    grid = calibration(load_directions("caa24"))
    selected = {m: block["grid"][2] for m, block in grid.items()}
    counts = [
        (240, {"rss": 30}),
        (3832, {"rss": 479}),
        (80, {"rss": 5}),
        (1216, {"no_rss": 12, "no_gram": 12, "no_clamp": 12, "full": 2}),
        (1440, {"rss": 33, "raw": 12}),
        (300, {"full": 3, "rss": 3}),
        (200, {"full": 4}),
    ]
    names = (
        "job1_caa24_calib_selection8",
        "job1_caa_mega_calib_selection8",
        "job1b_caa24_val_independent16",
        "job2_hys_ablations_and_ctrl",
        "job3_caa24_final_template",
        "job4_external_lambda_simple50",
        "job4a_hys_external_simple50",
    )
    for name, (n, modes) in zip(names, counts, strict=True):
        f = PACKAGE / "jobs" / f"{name}.json"
        cfg = read(f)
        assert cfg["expected_answers"] == n
        assert Counter(c["mode"] for c in cfg["conditions"]) == modes
        resolve_job(cfg, banks[cfg["bank"]], selected)


@pytest.mark.parametrize("name", tuple(LAYER_SETS))
@pytest.mark.parametrize("norm", ["natural", "unit"])
def test_caa_layer_sets_rss_and_hooks(name, norm):
    layers = LAYER_SETS[name]
    expected = {
        "all32": tuple(range(32)),
        "mid12_gdn": (8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22),
        "mid16_all": tuple(range(8, 24)),
        "early_mid12_gdn": (4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18),
        "gdn24": LAYERS,
    }
    assert layers == expected[name]
    model, tokenizer, device, dirs = mock_backend()
    bank = dirs["caa24"]
    mixed, stats = caa_phases(bank, {"N": 2, "F": 3}, 0.75, norm, "rss", False, device, layers)
    single, single_stats = caa_phases(bank, {"N": 2}, 0.75, norm, "rss", False, device, layers)
    assert set(mixed[1]) == set(stats[0]["per_layer"]) == set(layers)
    for i in layers:

        def contribution(tag, dose):
            d = bank[tag][i]
            return (
                0.75 * dose * d
                if norm == "natural"
                else 0.75 * dose * d / (d.norm() * math.sqrt(len(layers)))
            )

        a, b = contribution("N", 2), contribution("F", 3)
        g = math.sqrt(float(a.square().sum() + b.square().sum())) / float((a + b).norm())
        assert stats[0]["per_layer"][i] == pytest.approx(g)
        assert single_stats[0]["per_layer"][i] == 1.0
        torch.testing.assert_close(mixed[1][i], (a + b) * g)
        torch.testing.assert_close(single[1][i], a)
    result = generate(model, tokenizer, device, "hi", "sys", 2, caa=mixed)
    assert result["audit"]["caa_layer_calls"] == dict.fromkeys(layers, 2)


def test_mega_and_selected_caa_config_contract():
    jobs = PACKAGE / "jobs"
    mega = read(jobs / "job1_caa_mega_calib_selection8.json")
    cases = mega["conditions"]
    assert sum(not c["active"] for c in cases) == 1
    assert mega["max_new_tokens"] == 1024
    assert len(cases) == 479 and len({c["id"] for c in cases}) == 479
    singleton = [c for c in cases if c["active"]]
    assert {len(c["active"]) for c in singleton} == {1}
    assert len({c["family"] for c in singleton}) == 13
    assert sum(c["supplemental"] for c in singleton) == 10
    for c in singleton:
        assert c["layers"] == list(LAYER_SETS[c["layer_set"]])
        assert c["alpha_or_gain"] == c["alpha"][c["active"]]
        assert not c["use_windows"] or c["active"] in "NP"
    external = read(jobs / "job4a_hys_external_simple50.json")
    assert external["bank"] == "simple50"
    assert [(c["active"], c["lambda"]) for c in external["conditions"]] == [
        ("", 1),
        (FEATURES, 0.5),
        (FEATURES, 0.75),
        (FEATURES, 1),
    ]
    selected = {
        **read(jobs / "selected_caa_config.json"),
        "family": "mid12_gdn_unit_win",
        "norm_mode": "unit",
        "use_windows": True,
        "windows": {"N": 28, "P": 32},
        "layers": list(LAYER_SETS["mid12_gdn"]),
        "values": dict.fromkeys(FEATURES, 2.5),
    }
    for name in (
        "job1b_caa24_val_independent16",
        "job3_caa24_final_template",
        "job4_external_lambda_simple50",
    ):
        cfg = read(jobs / f"{name}.json")
        prompts = read_rows(PACKAGE / "banks" / f"{cfg['bank']}.jsonl")
        result = resolve_job(cfg, prompts, selected)
        for c in result["conditions"]:
            if c["method"] == "caa24":
                assert all(
                    c[k] == selected[k]
                    for k in ("family", "layers", "norm_mode", "use_windows", "windows")
                )
                assert c["alpha"] == {m: 2.5 for m in c["active"]}
            else:
                assert "family" not in c
        with pytest.raises(ValueError, match="Invalid CAA"):
            resolve_job(cfg, prompts, {**selected, "layers": [0, 0]})


DROP2 = {
    "job2b_hys_full_singlegpu_parity": (384, "heldout32", 4096),
    "job1b_caa_val_independent16": (176, "independent16", 1024),
    "job4b_caa_best_multilayer_simple50": (150, "simple50", 4096),
    "job3_caa_best_multilayer_heldout32": (1088, "heldout32", 4096),
    "job3b_caa_gdn24_ctrl_heldout32": (416, "heldout32", 4096),
    "job1c_gdn24_micro_selection8": (96, "selection8", 1024),
}


def test_drop2_counts_condition_sets_and_unique_answer_ids():
    total, all_ids = 0, set()
    for name, (count, bank, cap) in DROP2.items():
        cfg = read(PACKAGE / "jobs" / f"{name}.json")
        assert cfg["bank"] == bank and cfg["max_new_tokens"] == cap
        prompts = read_rows(PACKAGE / "banks" / f"{bank}.jsonl")
        job = resolve_job(cfg, prompts)
        ids = {opaque_id(name, key) for key in scheduled_tasks(job, prompts)}
        assert len(ids) == count == job["expected_answers"] and not (all_ids & ids)
        all_ids.update(ids)
        total += count
        c = job["conditions"]
        cells = Counter((len(row["active"]), row["lambda"]) for row in c)
        if "job2b" in name:
            assert cells == {(1, 1): 5, (4, 1): 5, (5, 1): 1, (5, 0.5): 1}
            assert {(row["method"], row["mode"]) for row in c} == {("hys", "full")}
        elif "job1b" in name:
            assert cells == {(0, 1): 1, (1, 1): 10}
            assert Counter(row.get("caa_config") for row in c) == {
                None: 1,
                "best_multilayer": 5,
                "gdn24_strict": 5,
            }
        elif "job4b" in name:
            assert cells == {(5, 1): 1, (5, 0.75): 1, (5, 0.5): 1}
        elif "job3b" in name:
            assert cells == {(1, 1): 5, (4, 1): 5, (5, 1): 1, (5, 0.5): 1, (5, 0.75): 1}
            assert all(row["layers"] == list(LAYERS) for row in c)
        elif "job3_" in name:
            assert cells == {
                (0, 1): 1,
                (1, 1): 5,
                (2, 1): 10,
                (3, 1): 10,
                (4, 1): 5,
                (5, 1): 1,
                (5, 0.75): 1,
                (5, 0.5): 1,
            }
        else:
            assert cells == {(1, 1): 12}
            assert {row["per_concept"]["N"]["window"] for row in c if row["active"] == "N"} == {28}
        # The parallel launcher embeds resolved CAA conditions in its shards.
        portable = [
            {k: v for k, v in row.items() if k not in ("alpha", "beta")}
            if row["method"] == "hys"
            else row
            for row in c
        ]
        assert resolve_job({**job, "conditions": portable}, prompts) == job
    assert len(all_ids) == total == 2310


def test_drop2_heterogeneous_rss_at_window_boundary():
    cfg = read(PACKAGE / "jobs/job3_caa_best_multilayer_heldout32.json")
    job = resolve_job(cfg, read_rows(PACKAGE / "banks/heldout32.jsonl"))
    c = next(row for row in job["conditions"] if row["active"] == FEATURES and row["lambda"] == 0.5)
    model, tokenizer, device, bank = mock_backend()
    specs = c["per_concept"]
    assert {t: specs[t]["window"] for t in FEATURES} == dict(N=28, F=None, T=None, R=None, P=None)
    assert {t: specs[t]["layers"] for t in FEATURES} == {
        t: list(LAYER_SETS[name])
        for t, name in zip(
            FEATURES, ("mid12_gdn", "all32", "mid12_gdn", "early_mid12_gdn", "all32"), strict=True
        )
    }
    assert {t: specs[t]["value"] for t in FEATURES} == dict(
        N=7.44337, F=1.05893, T=0.254482, R=5.34724, P=3.05788
    )
    phase, stats = plan({**job, "conditions": [c]}, bank, device)[c["id"]]
    assert list(phase) == [1, 29]
    step, audit = Step("decode", 28), {}
    zeros = torch.zeros(1, 1, 8)
    for t, idx in ((27, 0), (28, 1)):
        step.token_call = t + 1
        for layer in (0, 4, 8, 19, 22, 31):
            pieces = []
            for tag, s in specs.items():
                if layer in s["layers"] and (s["window"] is None or t < s["window"]):
                    d = bank["caa24"][tag][layer]
                    if s["norm_mode"] == "unit":
                        d = d / (d.norm() * math.sqrt(len(s["layers"])))
                    pieces.append(d * (0.5 * s["value"]))
            u = torch.stack(pieces).sum(dim=0)
            gamma = math.sqrt(sum(float(p.square().sum()) for p in pieces)) / float(u.norm())
            assert stats[idx]["per_layer"][layer] == pytest.approx(gamma)
            torch.testing.assert_close(
                caa_hook(step, layer, phase, audit)(None, (), zeros)[0, 0], u * gamma
            )
    assert torch.equal(phase[1][31], phase[29][31])
    assert not torch.equal(phase[1][8], phase[29][8])
    # Disjoint window: L8 has no delta after t=27, L0 has one concept
    # throughout and gamma=1; the native loop must audit unequal counts.
    specs = {
        "N": {**specs["N"], "layers": [8]},
        "F": {**specs["F"], "layers": [0]},
    }
    offsets, gamma = caa_phases(
        bank["caa24"], {}, 0.5, "unit", "rss", False, device, per_concept=specs
    )
    assert set(offsets[29]) == {0} and gamma[0]["per_layer"] == {0: 1.0, 8: 1.0}
    assert generate(model, tokenizer, device, "hi", "sys", 30, caa=offsets)["audit"][
        "caa_layer_calls"
    ] == {0: 30, 8: 28}


def test_external_heterogeneous_selection_preserves_legacy_math():
    cfg = read(PACKAGE / "jobs/job1b_caa24_val_independent16.json")
    prompts = read_rows(PACKAGE / "banks/independent16.jsonl")
    caps = {"N": 28, "P": 32}
    values = dict.fromkeys(FEATURES, 2.5)
    flat = {
        "family": "gdn24_natural_win",
        "layers": list(LAYERS),
        "norm_mode": "natural",
        "use_windows": True,
        "windows": caps,
        "values": values,
    }
    heterogeneous = {
        tag: {
            "layers": list(LAYERS),
            "norm_mode": "natural",
            "value": value,
            "window": caps.get(tag),
        }
        for tag, value in values.items()
    }
    cfg["conditions"] = [
        {
            "id": "parity",
            "method": "caa24",
            "mode": "rss",
            "active": FEATURES,
            "lambda": 0.75,
            "alpha": None,
        }
    ]
    cfg["expected_answers"] = len(prompts)
    _, _, device, bank = mock_backend()
    old, new = [
        plan(resolve_job(cfg, prompts, sel), bank, device)["parity"]
        for sel in (flat, {"per_concept": heterogeneous})
    ]
    assert old[1] == new[1]
    for k, deltas in old[0].items():
        assert set(deltas) == set(new[0][k])
        assert all(torch.equal(d, new[0][k][layer]) for layer, d in deltas.items())
    with pytest.raises(ValueError, match="Invalid per-concept"):
        resolve_job(cfg, prompts, {**heterogeneous, "N": {**heterogeneous["N"], "layers": [0, 0]}})


def test_drop2_ascii_blind_both_judges(tmp_path):
    text = "Français\u0085新\u2028ligne\u2029fin"
    rows = [
        {
            "answer_id": "opaque-1",
            "prompt_id": "q1",
            "prompt": "décrire",
            "response": text,
            "family": "secret condition",
        }
    ]
    export_judge(tmp_path, rows, "test")
    f = tmp_path / "blind.jsonl"
    raw = f.read_bytes()
    assert raw.isascii() and b"\\u0085" in raw and b"\\u2028" in raw and b"\\u2029" in raw
    assert len(raw.decode().splitlines()) == 1
    judge = Path(os.environ.get("DROP2_TRAIT_JUDGE_PATH", PACKAGE.parents[1] / "judge"))
    quality = os.environ.get("DROP2_QUALITY_JUDGE_PATH")
    code = "import sys;from pathlib import Path;from ready_judge.core import input_rows; r=input_rows(Path(sys.argv[1])); assert len(r)==1 and r[0]['text']==sys.argv[2]"
    subprocess.run(
        [sys.executable, "-B", "-c", code, str(f), text],
        check=True,
        env={**os.environ, "PYTHONPATH": str(judge), "PYTHONUTF8": "1"},
    )
    if quality:
        code = "import sys;from pathlib import Path;from hybrid_judge.runner import read_jsonl; r=read_jsonl(Path(sys.argv[1])); assert len(r)==1 and r[0].answers[0].text==sys.argv[2]"
        subprocess.run(
            [sys.executable, "-B", "-c", code, str(f), text],
            check=True,
            env={**os.environ, "PYTHONPATH": quality, "PYTHONUTF8": "1"},
        )
    assert "secret condition" not in raw.decode()


def test_parallel_cpu_mock_failure_resume_and_blind(tmp_path, monkeypatch):
    cfg, _ = smoke_config()
    cfg.update(bank="selection8", max_new_tokens=2, expected_answers=48)
    source = tmp_path / "tiny.json"
    source.write_text(json.dumps(cfg), encoding="utf-8")
    output = tmp_path / "mock"
    args = [
        "run_parallel.py",
        "--runner",
        "caa24",
        "--job",
        str(source),
        "--output",
        str(output),
        "--mock",
        "--gpus",
        "0",
        "--workers-per-gpu",
        "3",
        "--stagger-sec",
        "0",
    ]
    monkeypatch.setattr(sys, "argv", [*args, "--dry-run"])
    assert run_parallel.main() == 0 and not output.exists()
    real_popen = subprocess.Popen
    wrapper = """import sys, run_experiment as r
generate = r.generate
def fail_after_one(*args, **kwargs):
    r.generate = lambda *a, **kw: sys.exit('intentional after one durable row')
    return generate(*args, **kwargs)
r.generate = fail_after_one
r.main()
"""

    def popen(cmd, **kwargs):
        if Path(cmd[cmd.index("--output") + 1]).name == "worker_01":
            cmd = [*cmd[:2], "-c", wrapper, *cmd[3:]]
        return real_popen(cmd, **kwargs)

    monkeypatch.setattr(run_parallel.subprocess, "Popen", popen)
    monkeypatch.setattr(sys, "argv", args)
    assert run_parallel.main() == 1
    assert read(output / "manifest.json")["completed"] == 33
    assert not (output / "blind.jsonl").exists()
    workers = output / "workers"
    prompts = read_rows(PACKAGE / "banks/selection8.jsonl")
    keys = [[c["id"], p["id"]] for c in cfg["conditions"] for p in prompts]
    for i in range(3):
        shard = read(workers / f"worker_{i:02d}" / "shard_job.json")
        assert shard["tasks"] == keys[i::3]
    raw = workers / "worker_01" / "generations.jsonl"
    survived = read_rows(raw)[0]
    with raw.open("ab") as stream:  # Simulate a killed process's torn final line.
        stream.write(b'{"incomplete":')

    monkeypatch.setattr(run_parallel.subprocess, "Popen", real_popen)
    args[args.index("--workers-per-gpu") + 1] = "2"
    monkeypatch.setattr(sys, "argv", [*args, "--resume"])
    assert run_parallel.main() == 0
    rows = read_rows(output / "generations.jsonl")
    assert [[r["condition_id"], r["question_id"]] for r in rows] == keys
    assert len({r["answer_id"] for r in rows}) == 48 and survived in rows
    manifest = read(output / "manifest.json")
    assert manifest["status"] == "complete" and manifest["completed"] == 48
    assert set(manifest["gamma_by_condition"]) == {c["id"] for c in cfg["conditions"]}
    assert [len(read(workers / f"worker_{i:02d}" / "shard_job.json")["tasks"]) for i in (3, 4)] == [
        8,
        7,
    ]
    blind = read_rows(output / "blind.jsonl")
    assert len(blind) == 8
    assert all(set(p) == {"prompt_id", "scenario", "answers"} for p in blind)
    assert all(set(a) == {"answer_id", "text"} for p in blind for a in p["answers"])
    assert {a["answer_id"] for p in blind for a in p["answers"]} == {r["answer_id"] for r in rows}
    assert (output / "blind.jsonl").read_bytes() == (output / "judge_input.jsonl").read_bytes()
    assert run_parallel.main() == 0
    assert read_rows(output / "generations.jsonl") == rows
    assert len(list(workers.iterdir())) == 5


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
