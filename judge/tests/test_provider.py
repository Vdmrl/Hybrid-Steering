from pathlib import Path
from types import SimpleNamespace

from hybrid_judge import provider
from hybrid_judge.config import load_configs


class QuotaError(Exception):
    status_code = 402


def test_openrouter_switches_to_fallback_key(monkeypatch) -> None:
    calls = []

    class FakeOpenAI:
        def __init__(self, api_key, **kwargs):
            def create(**request):
                calls.append((api_key, request))
                if api_key == "primary":
                    raise QuotaError("quota exhausted")
                return "fallback response"

            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    monkeypatch.setenv("OPENROUTER_FALLBACK_API_KEY", "fallback")
    monkeypatch.setattr(provider, "OpenAI", FakeOpenAI)
    _, config = load_configs(Path(__file__).parents[1])

    client = provider.openrouter_client(config, "primary")

    assert client.chat.completions.create(model="judge") == "fallback response"
    assert client.chat.completions.create(model="judge") == "fallback response"
    assert [key for key, _ in calls] == ["primary", "fallback", "fallback"]
