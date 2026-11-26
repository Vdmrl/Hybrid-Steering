import importlib.util
import json
from pathlib import Path

import pytest
import torch

_PATH = Path(__file__).resolve().parents[1] / "experiments/forgetting/russian.py"
_SPEC = importlib.util.spec_from_file_location("russian_alignment", _PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load {_PATH}")
RUSSIAN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(RUSSIAN)


def test_prompt_columns_count_real_tokens_after_left_padding() -> None:
    mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
    columns = RUSSIAN.prompt_columns(torch.tensor([2, 1]), mask)
    torch.testing.assert_close(columns, torch.tensor([4, 2]))


def test_prompt_columns_rejects_a_position_past_the_prompt() -> None:
    with pytest.raises(ValueError):
        RUSSIAN.prompt_columns(4, torch.tensor([[1, 1, 1]]))


def test_write_cosine_matches_the_injected_matrix() -> None:
    reference = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
    same = reference.unsqueeze(0).repeat(2, 1, 1, 1)
    orthogonal = torch.zeros_like(same)
    orthogonal[..., 0, 1] = 1.0
    torch.testing.assert_close(RUSSIAN.write_cosine(same, reference), torch.ones(2))
    torch.testing.assert_close(RUSSIAN.write_cosine(orthogonal, reference), torch.zeros(2))


def test_state_cell_averages_the_head_norms_and_the_cosine() -> None:
    base = torch.zeros(2, 1, 1, 2, 2)
    steered = base.clone()
    steered[0, 0, 0] = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
    reference = steered[0, 0]
    cell = RUSSIAN.state_cell(base, steered, reference)
    torch.testing.assert_close(cell["state_delta"], torch.tensor([[2.5]]))
    torch.testing.assert_close(cell["write_cosine"], torch.tensor(0.5), atol=1e-5, rtol=0)


def test_figure_averages_both_fillers(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    rows = []
    for method, scale in (("baseline", 0.0), ("rank1", 1.0), ("rank2", 1.0), ("full", 1.0)):
        for filler in (0, 1):
            for length in (0, 4):
                for index in (0, 1):
                    rows.append(
                        {
                            "method": method,
                            "scale": scale,
                            "filler": filler,
                            "length": length,
                            "index": index,
                            "hit": int(method == "full"),
                            "language_pct": 100.0 if method == "full" else 0.0,
                        }
                    )
    cells = {}
    for method in ("rank1", "rank2", "full"):
        for filler, origin in ((0, 4.0), (1, 2.0)):
            for length, ratio in ((0, 1.0), (4, 0.5)):
                cells[(method, 1.0, filler, length)] = {
                    "state_delta": torch.tensor([[origin * ratio]]),
                    "write_cosine": torch.tensor(0.25 if length else 1.0),
                }
    (output / "rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    torch.save({"cells": cells, "layers": [0], "factors": {}}, output / "heads.pt")
    (output / "meta.json").write_text(
        json.dumps({"model_type": "qwen3_5", "lengths": [0, 4]}) + "\n", encoding="utf-8"
    )

    RUSSIAN.plot_figure(output)

    curves = json.loads((output / "curves.json").read_text(encoding="utf-8"))
    assert curves["state_ratio"]["full"][1]["score"] == 0.5
    assert curves["write_cosine"]["rank1"][1]["score"] == 0.25
    assert curves["behavior"]["full"][0]["score"] == 1.0
    assert (output / "figure.png").is_file()
