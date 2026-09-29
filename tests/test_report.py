from hybrid_steering.report import group_mean, report_main, summary_section


def test_group_mean_averages_a_score_within_each_cell() -> None:
    rows = [
        {"scale": 1, "mode": "initial", "concept_score": 1},
        {"scale": 1, "mode": "initial", "concept_score": 0},
        {"scale": 0, "mode": "initial", "concept_score": 0},
    ]
    grouped = group_mean(rows, "scale", "concept_score", "mode")
    cell = next(item for item in grouped if item["scale"] == 1)
    assert cell["concept_score"] == 0.5
    assert cell["n"] == 2
    assert "initial" in summary_section(rows, "scale", "concept_score", "mode", "Score")


def test_report_main_writes_the_named_columns(tmp_path, monkeypatch) -> None:
    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"scale": 1, "concept_score": 1}\n', encoding="utf-8")
    output = tmp_path / "report.html"
    monkeypatch.setattr(
        "sys.argv",
        ["report", "--rows", str(rows), "--output", str(output)],
    )
    report_main("Steering", [("scale", "concept_score", None, "Concept score by scale")])
    text = output.read_text(encoding="utf-8")
    assert "Concept score by scale" in text
    assert "1.000" in text
