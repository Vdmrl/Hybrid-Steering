import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gdn_interp import AnswerEquivalenceDetector, LLMJudge, LanguageDetector


class LanguageDetectorTest(unittest.TestCase):
    def test_blank_text_is_unknown(self) -> None:
        detector = LanguageDetector("ru")

        self.assertEqual(detector.label(" \n\t"), "unknown")
        self.assertEqual(detector.confidence(""), 0.0)

    def test_score_measures_target_language_gain(self) -> None:
        detector = LanguageDetector("ru")
        base = "The quick brown fox jumps over the lazy dog."
        steered = "Быстрая коричневая лиса перепрыгивает через ленивую собаку."

        self.assertEqual(detector.label(base), "en")
        self.assertEqual(detector.label(steered), "ru")
        self.assertEqual(detector.score(base, base), 0.0)
        self.assertEqual(detector.score(base, steered), 1.0)


class AnswerEquivalenceDetectorTest(unittest.TestCase):
    def test_score_accepts_only_tagged_verdict(self) -> None:
        detector = AnswerEquivalenceDetector()
        score, raw = detector.score(
            "", "What color is A?", "A is blue.", "A is синий.", lambda _: "<verdict>1</verdict>"
        )

        self.assertEqual(score, 1)
        self.assertEqual(raw, "<verdict>1</verdict>")
        self.assertEqual(detector.parse("<verdict>1</verdict> because yes"), 0)

    def test_litellm_judge_returns_one_response_per_prompt(self) -> None:
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="<verdict>1</verdict>"))])
        judge = LLMJudge(model="openai/Qwen/Qwen3.8-27B", api_base="http://127.0.0.1:8000/v1", api_key="EMPTY")

        with patch("gdn_interp.detector.answer_equivalence.batch_completion", return_value=[response, response]) as completion:
            self.assertEqual(judge(["first", "second"]), ["<verdict>1</verdict>", "<verdict>1</verdict>"])

        self.assertEqual(completion.call_args.kwargs["model"], "openai/Qwen/Qwen3.8-27B")
        self.assertEqual(completion.call_args.kwargs["api_base"], "http://127.0.0.1:8000/v1")
        self.assertEqual(completion.call_args.kwargs["extra_body"], {"chat_template_kwargs": {"enable_thinking": False}})


if __name__ == "__main__":
    unittest.main()
