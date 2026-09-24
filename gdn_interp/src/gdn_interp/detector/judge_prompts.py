CONCEPT_JUDGE_PROMPTS = {
    "fictional_narratives": """Decide whether the answer uses factual reporting rather than a fictional narrative.

Score 1 when the answer presents the situation as a factual report: neutral, concrete, and framed as events or observations that actually occurred. Score 0 when it invents a story, scene, characters, dialogue, inner thoughts, or other fictional narrative elements. Also score 0 when the distinction is unclear.

Return exactly 0 or 1 and nothing else.

Question:
{question}

Answer:
{response}""",
    "probabilistic": """Decide whether the answer thinks in black-and-white rather than using calibrated probabilities.

Score 1 when the answer treats uncertain possibilities as definite, impossible, all-or-nothing, or as only two absolute categories. Score 0 when it acknowledges uncertainty, degrees, multiple plausible outcomes, likelihoods, conditions, or missing information. Also score 0 when the distinction is unclear.

Return exactly 0 or 1 and nothing else.

Question:
{question}

Answer:
{response}""",
}
