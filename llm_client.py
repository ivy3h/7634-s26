"""OpenAI-compatible wrapper around a local vLLM server.

Single point of LLM access for the whole project. Every other module
must route calls through `chat()` or `chat_json()` so the backend can
be swapped without touching the rest of the code.

Endpoint and model come from env vars, never hard-coded:
    LLM_ENDPOINT   default http://localhost:8000/v1
    LLM_MODEL      default Qwen/Qwen3-8B
    LLM_API_KEY    default EMPTY (vLLM ignores it)
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

_client: OpenAI | None = None
_DEFAULT_MODEL = "Qwen/Qwen3-8B"


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        base_url = os.environ.get("LLM_ENDPOINT", "http://localhost:8000/v1")
        api_key = os.environ.get("LLM_API_KEY", "EMPTY")
        _client = OpenAI(base_url=base_url, api_key=api_key, timeout=600.0)
    return _client


def _resolve_model(model: str | None) -> str:
    return model or os.environ.get("LLM_MODEL", _DEFAULT_MODEL)


def chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    retries: int = 8,
    **kwargs: Any,
) -> str:
    """Send a chat request and return the assistant message content.

    Retries transient connection / timeout / rate-limit errors with
    exponential backoff so the caller can fire requests while the vLLM
    server is still warming up.
    """
    client = _get_client()
    resolved_model = _resolve_model(model)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            return (resp.choices[0].message.content or "").strip()
        except (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError) as err:
            last_err = err
            backoff = min(60.0, 2.0 ** attempt)
            time.sleep(backoff)
    raise RuntimeError(f"chat failed after {retries} retries: {last_err!r}")


def chat_simple(prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
    """Convenience wrapper for a single-turn user prompt."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat(messages, **kwargs)


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_safe(text: str) -> Any:
    """Strip markdown fences and parse JSON. Fallback: find first `{` / `[`."""
    text = text.strip()
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for opener, closer in (("{", "}"), ("[", "]")):
            start = text.find(opener)
            end = text.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue
        raise


def chat_json(
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.3,
    max_parse_retries: int = 3,
    **kwargs: Any,
) -> Any:
    """Ask for JSON, parse it, reprompt briefly if it fails to parse."""
    last_raw = ""
    for attempt in range(max_parse_retries):
        raw = chat_simple(
            prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature if attempt == 0 else max(0.1, temperature - 0.1 * attempt),
            **kwargs,
        )
        last_raw = raw
        try:
            return parse_json_safe(raw)
        except (json.JSONDecodeError, ValueError):
            continue
    raise ValueError(f"could not parse JSON after {max_parse_retries} attempts. Last raw:\n{last_raw[:400]}")


def health_check() -> dict[str, Any]:
    """Return `{ok, endpoint, model}` — fast smoke-test."""
    endpoint = os.environ.get("LLM_ENDPOINT", "http://localhost:8000/v1")
    model = _resolve_model(None)
    try:
        out = chat_simple("Reply with just: ok", max_tokens=16, temperature=0.0, retries=2)
        return {"ok": True, "endpoint": endpoint, "model": model, "reply": out}
    except Exception as err:  # noqa: BLE001 — surface any failure to caller
        return {"ok": False, "endpoint": endpoint, "model": model, "error": repr(err)}


if __name__ == "__main__":
    print(json.dumps(health_check(), indent=2))
