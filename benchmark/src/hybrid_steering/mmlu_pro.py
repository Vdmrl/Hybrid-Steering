"""Pinned zero-shot MMLU-Pro protocol (not the upstream five-shot CoT score)."""

import re


def task(row: dict) -> dict:
    options = [item for item in row["options"] if item != "N/A"]
    answer = row["answer"]
    if (not row.get("question_id") or not isinstance(row.get("question"), str)
            or not 2 <= len(options) <= 16 or not all(isinstance(x, str) for x in options)
            or not isinstance(answer, str) or answer not in "ABCDEFGHIJKLMNOP"[:len(options)]):
        raise ValueError("Invalid MMLU-Pro row")
    choices = "\n".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(options))
    return {"key": row["question_id"], "prompt":
            f"Question: {row['question']}\nOptions:\n{choices}\n"
            "Give the final answer as 'Answer: X', where X is one option letter.",
            "answer": answer, "option_count": len(options), "category": row.get("category")}


def score(response: str, row: dict) -> dict:
    letters = "ABCDEFGHIJKLMNOP"[:row["option_count"]]
    matches = re.findall(r"\banswer\s*(?:is|:)\s*\(?([A-P])\)?\b", response, re.IGNORECASE)
    prediction = matches[-1].upper() if matches else None
    valid = prediction is not None and prediction in letters
    return {"mmlu_prediction": prediction if valid else None,
            "mmlu_format_valid": valid, "mmlu_correct": valid and prediction == row["answer"]}
