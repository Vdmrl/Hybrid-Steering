import ast
from pathlib import Path

PATH = Path(__file__).with_name("run.py")


def test_features_are_distinct_and_complete():
    tree = ast.parse(PATH.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(name, ast.Name) and name.id == "FEATURES"
            for name in node.targets
        )
    )
    features = ast.literal_eval(assignment.value)
    assert set(features) == {
        "atomic_sentences",
        "explicit_connectives",
        "rhetorical_questions",
    }
    assert all(target != opposite for target, opposite in features.values())
