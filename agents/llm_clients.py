from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any


def strip_think(text: str) -> str:
    """Remove qwen-style <think>...</think> reasoning blocks from output."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def _messages(system_prompt: str, user_payload: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
    ]


def _read_http_error(exc: urllib.error.HTTPError) -> str:
    try:
        detail = exc.read().decode("utf-8", errors="replace")
    except Exception:
        detail = str(exc)
    return f"HTTP {exc.code}: {detail}"


def _post_json(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout: int,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(_read_http_error(exc)) from exc


def _openrouter_headers() -> dict[str, str]:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("No backup model credential is configured.")

    headers = {"Authorization": f"Bearer {api_key}"}
    site_url = os.getenv("OPENROUTER_SITE_URL", "").strip()
    app_name = os.getenv("OPENROUTER_APP_NAME", "Lebanon Export Dashboard").strip()
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name
    return headers


def _openrouter_fallback_model() -> str:
    return os.getenv("OPENROUTER_MODEL", "openrouter/free").strip() or "openrouter/free"


def _has_openrouter_fallback() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY", "").strip())


def _raise_combined_failure(primary_exc: Exception, fallback_exc: Exception) -> None:
    raise RuntimeError(
        "The primary and backup model services both failed. "
        f"Primary error: {primary_exc}. Backup error: {fallback_exc}."
    ) from fallback_exc


def call_groq(system_prompt: str, payload: dict[str, Any], model: str) -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("No primary model credential is configured.")
    data = _post_json(
        "https://api.groq.com/openai/v1/chat/completions",
        {
            "model": model,
            "messages": _messages(system_prompt, payload),
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "include_reasoning": False,
        },
        {"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    return strip_think(data["choices"][0]["message"]["content"])


def call_openrouter(system_prompt: str, payload: dict[str, Any], model: str) -> str:
    """Call OpenRouter's OpenAI-compatible chat-completions endpoint.

    The diagnostic prompt already requires one JSON object. We intentionally do
    not force response_format because not every OpenRouter model/router supports
    the same structured-output options. The schema validator remains the final
    guardrail and falls back deterministically when needed.
    """
    data = _post_json(
        "https://openrouter.ai/api/v1/chat/completions",
        {
            "model": model,
            "messages": _messages(system_prompt, payload),
            "temperature": 0.1,
            "max_tokens": 1800,
        },
        _openrouter_headers(),
        timeout=120,
    )
    return strip_think(data["choices"][0]["message"]["content"])


def call_ollama(system_prompt: str, payload: dict[str, Any], model: str) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    data = _post_json(
        f"{base_url.rstrip('/')}/api/chat",
        {
            "model": model,
            "messages": _messages(system_prompt, payload),
            "stream": False,
            "think": False,
            "format": "json",
            "options": {"temperature": 0.1},
        },
        {},
        timeout=120,
    )
    return strip_think(data["message"]["content"])


def call_lmstudio(system_prompt: str, payload: dict[str, Any], model: str) -> str:
    base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    data = _post_json(
        f"{base_url.rstrip('/')}/chat/completions",
        {
            "model": model,
            "messages": _messages(system_prompt, payload),
            "temperature": 0.1,
        },
        {},
        timeout=120,
    )
    return strip_think(data["choices"][0]["message"]["content"])


def call_llm(provider: str, system_prompt: str, payload: dict[str, Any], model: str) -> str:
    """Run a structured LLM call.

    Groq is the primary provider. When it raises an error and a backup API key
    is configured, the same prompt is retried once through the backup route.
    """
    provider = provider.lower().strip()
    if provider == "groq":
        try:
            return call_groq(system_prompt, payload, model)
        except Exception as primary_exc:
            if not _has_openrouter_fallback():
                raise
            try:
                return call_openrouter(
                    system_prompt,
                    payload,
                    _openrouter_fallback_model(),
                )
            except Exception as fallback_exc:
                _raise_combined_failure(primary_exc, fallback_exc)
    if provider == "openrouter":
        return call_openrouter(system_prompt, payload, model)
    if provider == "ollama":
        return call_ollama(system_prompt, payload, model)
    if provider == "lmstudio":
        return call_lmstudio(system_prompt, payload, model)
    raise ValueError(f"Unsupported provider: {provider}")


# ------------------------------------------------------------------
# Free-text chat completions (with history) for the conversational layer
# ------------------------------------------------------------------


def chat_completion(
    provider: str,
    system_prompt: str,
    messages: list[dict[str, str]],
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 800,
) -> str:
    """Plain-text chat completion with conversation history."""
    provider = provider.lower().strip()
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    if provider == "groq":
        try:
            api_key = os.getenv("GROQ_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("No primary model credential is configured.")
            data = _post_json(
                "https://api.groq.com/openai/v1/chat/completions",
                {
                    "model": model,
                    "messages": full_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "include_reasoning": False,
                },
                {"Authorization": f"Bearer {api_key}"},
                timeout=60,
            )
            return strip_think(data["choices"][0]["message"]["content"])
        except Exception as primary_exc:
            if not _has_openrouter_fallback():
                raise
            try:
                data = _post_json(
                    "https://openrouter.ai/api/v1/chat/completions",
                    {
                        "model": _openrouter_fallback_model(),
                        "messages": full_messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    _openrouter_headers(),
                    timeout=120,
                )
                return strip_think(data["choices"][0]["message"]["content"])
            except Exception as fallback_exc:
                _raise_combined_failure(primary_exc, fallback_exc)

    if provider == "openrouter":
        data = _post_json(
            "https://openrouter.ai/api/v1/chat/completions",
            {
                "model": model,
                "messages": full_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            _openrouter_headers(),
            timeout=120,
        )
        return strip_think(data["choices"][0]["message"]["content"])

    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        data = _post_json(
            f"{base_url.rstrip('/')}/api/chat",
            {
                "model": model,
                "messages": full_messages,
                "stream": False,
                "think": False,
                "options": {"temperature": temperature},
            },
            {},
            timeout=180,
        )
        return strip_think(data["message"]["content"])

    if provider == "lmstudio":
        base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        data = _post_json(
            f"{base_url.rstrip('/')}/chat/completions",
            {
                "model": model,
                "messages": full_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            {},
            timeout=180,
        )
        return strip_think(data["choices"][0]["message"]["content"])

    raise ValueError(f"Unsupported provider: {provider}")
