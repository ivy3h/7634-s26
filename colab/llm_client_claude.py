"""Drop-in replacement for llm_client.py that routes every call to the
Anthropic Claude API instead of a local vLLM server.

Same public surface (`chat`, `chat_simple`, `chat_json`, `health_check`,
`parse_json_safe`) as the vLLM-backed version, so none of the other modules
(phase1_story_generator, story_to_plan, world_builder, action_interpreter,
drama_manager, game_engine) need any change.

Env vars:
    ANTHROPIC_API_KEY   required
    LLM_MODEL           default claude-sonnet-4-5
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import anthropic

_client: anthropic.Anthropic | None = None
_DEFAULT_MODEL = "claude-sonnet-4-5"


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. In Colab, add it to "
                "'Secrets' (key icon) as ANTHROPIC_API_KEY and grant the "
                "notebook access."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def _resolve_model(model: str | None) -> str:
    return model or os.environ.get("LLM_MODEL", _DEFAULT_MODEL)


def _split_system(messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, str]]]:
    """OpenAI-style messages -> (system_prompt, claude_messages). Claude takes
    the system prompt as a separate top-level argument, not as a role=system
    message."""
    system_parts: list[str] = []
    claude_msgs: list[dict[str, str]] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            if content:
                system_parts.append(content)
        else:
            claude_msgs.append({"role": role, "content": content})
    system = "\n\n".join(system_parts) if system_parts else None
    return system, claude_msgs


def chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    retries: int = 8,
    enable_thinking: bool = False,
    extra_body: dict[str, Any] | None = None,
    **kwargs: Any,
) -> str:
    """Send a chat request to Claude and return the assistant message content.

    Accepts the same keyword arguments as the vLLM-backed wrapper. vLLM-only
    options (`enable_thinking`, `extra_body`, `chat_template_kwargs`) are
    accepted and ignored so callers don't need to branch on backend.
    """
    _ = enable_thinking, extra_body, kwargs  # vLLM-only knobs; ignored

    resolved_model = _resolve_model(model)
    system, claude_msgs = _split_system(messages)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            call_kwargs: dict[str, Any] = dict(
                model=resolved_model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=claude_msgs,
            )
            if system is not None:
                call_kwargs["system"] = system
            resp = _get_client().messages.create(**call_kwargs)
            text = "".join(
                block.text for block in resp.content if getattr(block, "type", None) == "text"
            )
            return text.strip()
        except (anthropic.APIConnectionError, anthropic.APITimeoutError,
                anthropic.InternalServerError, anthropic.RateLimitError) as err:
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
    raise ValueError(
        f"could not parse JSON after {max_parse_retries} attempts. Last raw:\n{last_raw[:400]}"
    )


def health_check() -> dict[str, Any]:
    """Return `{ok, backend, model}` -- fast smoke-test."""
    model = _resolve_model(None)
    try:
        out = chat_simple("Reply with just: ok", max_tokens=16, temperature=0.0, retries=2)
        return {"ok": True, "backend": "anthropic", "model": model, "reply": out}
    except Exception as err:  # noqa: BLE001
        return {"ok": False, "backend": "anthropic", "model": model, "error": repr(err)}


if __name__ == "__main__":
    print(json.dumps(health_check(), indent=2))
