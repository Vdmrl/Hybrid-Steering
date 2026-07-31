import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD_PATH = Path(__file__).with_name("build.py")
SPEC = importlib.util.spec_from_file_location("interim_dashboard", BUILD_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_dashboard_uses_completed_summaries() -> None:
    exp3 = json.loads(
        (ROOT / "experiments/meeting-dashboard-3-real/exp3_summary.json").read_text(
            encoding="utf-8"
        )
    )
    exp4 = json.loads(
        (ROOT / "outputs/strong-composition-exp4/summary.json").read_text(
            encoding="utf-8"
        )
    )

    page = MODULE.build(exp3, exp4)

    assert "Промежуточные результаты" in page
    assert "GDN против activation steering" in page
    assert "Радость + Оптимизм" in page
    assert "Короткие выводы" in page
    assert "Эксперимент" not in page
    assert "__DATA__" not in page
    assert "__GENERATED__" not in page
