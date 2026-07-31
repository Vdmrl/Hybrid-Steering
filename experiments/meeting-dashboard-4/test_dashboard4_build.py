import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("build.py")
SPEC = importlib.util.spec_from_file_location("dashboard4_build", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_describes_exp4() -> None:
    html = MODULE.build()
    assert "Strong composition" in html
    assert "rank4" in html
    assert "4608" in html
    assert "RSS" in html
    assert "Тройки признаков" in html
    assert "__GENERATED__" not in html
