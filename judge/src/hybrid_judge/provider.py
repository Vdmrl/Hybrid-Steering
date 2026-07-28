from __future__ import annotations

import os

import httpx
from openai import OpenAI

from .models import JudgeConfigV2


def openrouter_client(config: JudgeConfigV2, api_key: str) -> OpenAI:
    proxy = os.environ.get("OPENROUTER_PROXY", "").strip() or None
    return OpenAI(
        api_key=api_key,
        base_url=config.base_url,
        timeout=config.generation.timeout_seconds,
        max_retries=config.generation.max_retries,
        http_client=httpx.Client(proxy=proxy),
    )
