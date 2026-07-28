from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .models import PairwiseAggregateV2, PairwiseResultV2


def _winner_id(result: PairwiseResultV2, field: str) -> str:
    winner = getattr(result, field)
    if winner == "left":
        return result.left_answer_id
    if winner == "right":
        return result.right_answer_id
    return "tie"


def _decision(
    group: list[PairwiseResultV2], field: str, complete: bool
) -> tuple[bool | None, str | None]:
    if not complete:
        return None, None
    winners = [_winner_id(result, field) for result in group]
    consistent = winners[0] == winners[1]
    return consistent, winners[0] if consistent else "tie"


def aggregate_pairwise_v2(
    results: list[PairwiseResultV2],
) -> list[PairwiseAggregateV2]:
    groups: dict[tuple[str, str, tuple[str, str]], list[PairwiseResultV2]] = (
        defaultdict(list)
    )
    for result in results:
        answer_ids = tuple(sorted([result.left_answer_id, result.right_answer_id]))
        groups[(result.prompt_id, result.feature, answer_ids)].append(result)

    aggregates = []
    for (prompt_id, feature, answer_ids), group in sorted(groups.items()):
        group = sorted(group, key=lambda result: result.orientation)
        complete = len(group) == 2 and {result.orientation for result in group} == {
            "one",
            "reverse",
        }

        trait_consistent, trait_winner = _decision(group, "trait_winner", complete)
        quality_consistent, quality_winner = _decision(
            group, "quality_winner", complete
        )
        pair_key = ":".join(answer_ids)
        aggregates.append(
            PairwiseAggregateV2(
                aggregate_id=f"pairwise-v2:{feature}:{prompt_id}:{pair_key}",
                prompt_id=prompt_id,
                feature=feature,
                answer_ids=list(answer_ids),
                orientation_count=len(group),
                status="complete" if complete else "incomplete",
                trait_order_consistent=trait_consistent,
                quality_order_consistent=quality_consistent,
                trait_winner_answer_id=trait_winner,
                quality_winner_answer_id=quality_winner,
                task_ids=[result.task_id for result in group],
            )
        )
    return aggregates


def aggregate_pairwise_file(input_path: Path, output_path: Path) -> int:
    results = [
        PairwiseResultV2.model_validate_json(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    aggregates = aggregate_pairwise_v2(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(result.model_dump_json() + "\n" for result in aggregates),
        encoding="utf-8",
    )
    return len(aggregates)
