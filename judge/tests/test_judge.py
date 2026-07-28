from pathlib import Path

from hybrid_judge.config import load_configs
from hybrid_judge.models import PairwiseResponse, ScalarResponse
from hybrid_judge.runner import pairwise_tasks, read_jsonl, render_prompt

ROOT = Path(__file__).parents[1]


def test_contracts_and_blinding() -> None:
    features, config = load_configs(ROOT, "v1")
    rows = read_jsonl(ROOT / "examples" / "input.example.jsonl")
    feature = features.features["optimism"]
    prompt = render_prompt(
        (ROOT / "prompts" / config.evaluation.scalar_prompt).read_text(
            encoding="utf-8"
        ),
        feature,
        config.scale.anchors,
    )
    assert "optimism" in prompt
    assert "baseline" not in prompt
    assert (
        len(pairwise_tasks(rows, feature_name="optimism", both_orders=True, seed=1))
        == 2
    )

    ScalarResponse.model_validate(
        {
            "scores": [
                {
                    "answer_id": "answer_0",
                    "target_score": 4,
                    "opposite_score": 0,
                    "task_correctness": 4,
                    "coherence": 4,
                    "content_preservation": 4,
                    "reason": "clear",
                }
            ]
        }
    )
    PairwiseResponse.model_validate(
        {
            "feature_winner": "A",
            "quality_winner": "tie",
            "reason": "clear",
        }
    )
