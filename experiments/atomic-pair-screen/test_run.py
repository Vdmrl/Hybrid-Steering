import ast
from pathlib import Path

PATH = Path(__file__).with_name("run.py")


def test_pair_conditions_include_both_singletons_and_composition():
    tree = ast.parse(PATH.read_text(encoding="utf-8"))
    assert any(
        isinstance(node, ast.For)
        and isinstance(node.iter, ast.Tuple)
        and len(node.iter.elts) == 4
        for node in ast.walk(tree)
    )
