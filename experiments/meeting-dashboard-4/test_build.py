from build import build


def test_build_describes_exp4() -> None:
    html = build()
    assert "Strong composition" in html
    assert "rank4" in html
    assert "4608" in html
    assert "RSS" in html
    assert "__GENERATED__" not in html
