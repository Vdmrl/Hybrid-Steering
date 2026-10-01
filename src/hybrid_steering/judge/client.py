"""LiteLLM completions for judge calls. The key and proxy stay in the environment."""

from __future__ import annotations

import os

from .config import load_settings

JSON_OBJECT = {"type": "json_object"}


def _endpoint() -> tuple[str, str]:
    """OpenRouter or a self-hosted vLLM server, both OpenAI-compatible."""
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not base_url or not api_key:
        raise RuntimeError("set OPENAI_BASE_URL and OPENAI_API_KEY")
    return base_url, api_key


def complete_batch(
    message_lists: list[list[dict[str, str]]],
    *,
    max_tokens: int | None = None,
    json_object: bool = False,
    batch_size: int = 8,
    thinking: bool | None = None,
) -> list[str | None]:
    """Send every message list together. ``batch_size`` is how many run at once.

    ``max_tokens`` and ``thinking`` default to ``config/judge.yaml``. A failed
    request, or one cut at the token limit, is ``None``. The exception text is
    not printed.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not message_lists:
        return []
    import litellm

    base_url, api_key = _endpoint()
    proxy = os.environ.get("OPENROUTER_PROXY", "").strip()
    if proxy:
        os.environ.setdefault("HTTPS_PROXY", proxy)
    settings = load_settings()
    kwargs: dict = {
        "model": settings.model,
        "temperature": 0,
        "max_tokens": settings.max_tokens if max_tokens is None else max_tokens,
        "api_key": api_key,
        "base_url": base_url,
        "max_workers": batch_size,
    }
    if json_object:
        kwargs["response_format"] = JSON_OBJECT
    if not (settings.thinking if thinking is None else thinking):
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    responses = litellm.batch_completion(messages=message_lists, **kwargs)
    contents: list[str | None] = []
    failed = truncated = 0
    for response in responses:
        if isinstance(response, BaseException):
            failed += 1
            if failed == 1:
                print(f"judge request failed: {type(response).__name__}", flush=True)
            contents.append(None)
            continue
        choice = response.choices[0]
        content = choice.message.content
        if getattr(choice, "finish_reason", None) == "length":
            truncated += 1
            content = None
        contents.append(content.strip() if isinstance(content, str) and content.strip() else None)
        if contents[-1] is None:
            failed += 1
    if failed:
        print(
            f"judge missed {failed} of {len(responses)} responses, {truncated} at the token limit",
            flush=True,
        )
    return contents
