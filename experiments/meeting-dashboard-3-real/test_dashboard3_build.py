import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("build.py")
SPEC = importlib.util.spec_from_file_location("dashboard3_real_build", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_uses_real_exp3_summary() -> None:
    root = Path(__file__).parent
    summary = json.loads((root / "exp3_summary.json").read_text(encoding="utf-8"))
    selection = json.loads((root / "exp3_selection.json").read_text(encoding="utf-8"))
    html = MODULE.build(summary, selection)
    assert "composition-normalization-v3" in html
    assert "gdn_rss_r4_1111" in html
    assert "GDN raw, rank 1" in html
    assert "Тройки: сила и сохранение качества" in html
    assert "__DATA__" not in html


def test_bit_masks_follow_experiment_contract() -> None:
    expected = {
        "0001": ("joy",),
        "0010": ("concrete",),
        "0100": ("optimism",),
        "1000": ("candor",),
        "0011": ("joy", "concrete"),
        "1110": ("concrete", "optimism", "candor"),
    }
    for mask, features in expected.items():
        assert MODULE.mask_features(mask) == features


def test_dashboard3_uses_data_driven_labels_and_visuals() -> None:
    root = Path(__file__).parent
    html = MODULE.build(
        json.loads((root / "exp3_summary.json").read_text(encoding="utf-8")),
        json.loads((root / "exp3_selection.json").read_text(encoding="utf-8")),
    )
    assert "active_features" in html
    assert "Forest plot paired-контрастов" in html
    assert "Профиль признаков, шкала 1–5" in html
    assert "scoreDelta" in html
    assert "п.п." not in html
    assert "__CONDITION_COUNT__" not in html
    assert "__TEST_N__" not in html
    assert "балла" in html
