import re
from collections.abc import Callable

from litellm import batch_completion


JUDGE_PROMPT = """You are a factual-equivalence judge. Compare the model answer with the baseline answer for the given question.

Return `<verdict>1</verdict>` only when both answers give the same factual answer. Answers may use different languages, wording, formatting, or detail. Return `<verdict>0</verdict>` if they conflict, either answer does not answer the question, either answer is factually wrong, or equivalence is uncertain.

Your entire response must be exactly one of these two strings, with no other text:
<verdict>1</verdict>
<verdict>0</verdict>

Question:
{question}

Model answer:
{response}

Baseline answer:
{baseline_response}
"""

VERDICT_PATTERN = re.compile(r"\s*<verdict>([01])</verdict>\s*\Z")


class AnswerEquivalenceDetector:
    """Judge whether a steered answer factually matches its baseline."""

    def prompt(self, context: str, question: str, response: str, baseline_response: str) -> str:
        return JUDGE_PROMPT.format(
            context=context,
            question=question,
            response=response,
            baseline_response=baseline_response,
        )

    def parse(self, raw: str) -> int:
        match = VERDICT_PATTERN.fullmatch(raw)
        return int(match.group(1)) if match else 0

    def score(self, context: str, question: str, response: str, baseline_response: str, judge: Callable[[str], str]) -> tuple[int, str]:
        raw = judge(self.prompt(context, question, response, baseline_response))
        return self.parse(raw), raw


class LLMJudge:
    """Generate verdicts through LiteLLM."""

    def __init__(
        self,
        *,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 16,
        max_workers: int = 16,
    ) -> None:
        if not model:
            raise ValueError("model must not be empty")
        if max_tokens < 1 or max_workers < 1:
            raise ValueError("max_tokens and max_workers must be positive")

        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.max_workers = max_workers

    def __call__(self, prompts: list[str]) -> list[str]:
        if not prompts:
            return []

        responses = batch_completion(
            model=self.model,
            messages=[[{"role": "user", "content": prompt}] for prompt in prompts],
            temperature=0,
            api_base=self.api_base,
            api_key=self.api_key,
            max_tokens=self.max_tokens,
            max_workers=self.max_workers,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return [str(response.choices[0].message.content or "") for response in responses]
