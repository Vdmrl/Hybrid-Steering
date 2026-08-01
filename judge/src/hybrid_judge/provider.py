from __future__ import annotations

import os
from threading import Lock
from types import SimpleNamespace
from typing import Any

import httpx
from openai import OpenAI

from .models import JudgeConfig


class _FallbackCompletions:
    def __init__(self, clients: list[OpenAI]) -> None:
        self._clients = clients
        self._index = 0
        self._lock = Lock()

    def create(self, **kwargs: Any) -> Any:
        while True:
            with self._lock:
                index = self._index
            try:
                return self._clients[index].chat.completions.create(**kwargs)
            except Exception as exc:
                if getattr(exc, "status_code", None) not in {402, 429}:
                    raise
                with self._lock:
                    if self._index == index and index + 1 < len(self._clients):
                        self._index += 1
                        print(
                            "OpenRouter primary quota exhausted; using fallback key",
                            flush=True,
                        )
                    if self._index == index:
                        raise


def openrouter_client(config: JudgeConfig, api_key: str) -> Any:
    proxy = os.environ.get("OPENROUTER_PROXY", "").strip() or None
    fallback = os.environ.get("OPENROUTER_FALLBACK_API_KEY", "").strip()
    keys = [api_key] + ([fallback] if fallback and fallback != api_key else [])
    clients = [
        OpenAI(
            api_key=key,
            base_url=config.base_url,
            timeout=config.generation.timeout_seconds,
            max_retries=config.generation.max_retries,
            http_client=httpx.Client(proxy=proxy),
        )
        for key in keys
    ]
    if len(clients) == 1:
        return clients[0]
    return SimpleNamespace(
        chat=SimpleNamespace(completions=_FallbackCompletions(clients))
    )
