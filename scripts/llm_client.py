#!/usr/bin/env python3
"""LLM client helpers (OpenAI / Azure Responses) with retry and JSON parsing."""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyserini_client import parse_env_file, project_root

KEY_NAMES = ("OPENAI_API_KEY", "OPEN_AI_KEY", "AZURE_OPENAI_API_KEY")
MODEL_NAMES = ("OPENAI_MODEL", "OPEN_AI_MODEL", "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_MODEL")
PHASE1_MODEL_NAMES = ("PHASE1_OPENAI_MODEL", "PHASE1_LLM_MODEL")
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_PHASE1_MODEL = "gpt-4.1-mini"
DEFAULT_AZURE_API_VERSION = "2025-04-01-preview"


@dataclass
class LLMConfig:
    url: str
    headers: dict[str, str]
    model: str
    mode: str  # "chat" or "responses"


def load_env(root: Path | None = None) -> dict[str, str]:
    root = root or project_root()
    values = parse_env_file(root / ".env.local")
    values.update(parse_env_file(root / ".env"))
    return values


def resolve_api_key(root: Path | None = None) -> str | None:
    root = root or project_root()
    env = load_env(root)
    for name in KEY_NAMES:
        value = env.get(name) or os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def resolve_model(root: Path | None = None, *, override: str | None = None) -> str:
    if override:
        return override.strip()
    root = root or project_root()
    env = load_env(root)
    for name in MODEL_NAMES:
        value = env.get(name) or os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return DEFAULT_MODEL


def resolve_phase1_model(root: Path | None = None, *, override: str | None = None) -> str:
    if override:
        return override.strip()
    root = root or project_root()
    env = load_env(root)
    for name in PHASE1_MODEL_NAMES:
        value = env.get(name) or os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return resolve_model(root)


def _get(env: dict[str, str], name: str) -> str:
    return (env.get(name) or os.environ.get(name) or "").strip()


def _openai_base_url(env: dict[str, str]) -> str | None:
    for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_API_URL"):
        value = _get(env, name)
        if value:
            return value.rstrip("/")
    return None


def resolve_config(root: Path | None = None, *, model: str | None = None) -> LLMConfig:
    root = root or project_root()
    env = load_env(root)
    model_name = resolve_model(root, override=model)
    api_key = resolve_api_key(root)
    if not api_key:
        raise RuntimeError(
            "Missing LLM key. Set OPEN_AI_KEY (or OPENAI_API_KEY) in .env.local"
        )

    # 1. Standard OpenAI key.
    if api_key.startswith("sk-"):
        return LLMConfig(
            url="https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            model=model_name,
            mode="chat",
        )

    # 2. Azure Responses API (preferred for cognitiveservices endpoints).
    endpoint = _get(env, "AZURE_OPENAI_ENDPOINT") or _get(env, "AZURE_OPENAI_API_BASE")
    if endpoint:
        endpoint = endpoint.rstrip("/")
        api_version = _get(env, "AZURE_OPENAI_API_VERSION") or DEFAULT_AZURE_API_VERSION
        # Allow the user to paste a full responses URL directly.
        if "/openai/responses" in endpoint:
            url = endpoint
        else:
            url = f"{endpoint}/openai/responses?api-version={api_version}"
        return LLMConfig(
            url=url,
            headers={"api-key": api_key, "Content-Type": "application/json"},
            model=model_name,
            mode="responses",
        )

    # 3. Custom base URL (self-hosted / proxy) using chat/completions.
    base_url = _openai_base_url(env)
    if base_url:
        if base_url.endswith("/chat/completions"):
            url = base_url
        elif base_url.endswith("/v1"):
            url = f"{base_url}/chat/completions"
        else:
            url = f"{base_url}/v1/chat/completions"
        return LLMConfig(
            url=url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            model=model_name,
            mode="chat",
        )

    # 4. Fallback.
    return LLMConfig(
        url="https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        model=model_name,
        mode="chat",
    )


def _post(url: str, headers: dict[str, str], body: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_with_retry(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    *,
    timeout: float,
) -> dict[str, Any]:
    """POST once; on HTTP 400 retry without ``temperature`` (some 5.x models reject it)."""
    try:
        return _post(url, headers, body, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 400 and "temperature" in detail.lower() and "temperature" in body:
            retry_body = {k: v for k, v in body.items() if k != "temperature"}
            try:
                return _post(url, headers, retry_body, timeout=timeout)
            except urllib.error.HTTPError as exc2:
                detail = exc2.read().decode("utf-8", errors="replace")[:400]
                raise RuntimeError(f"LLM HTTP {exc2.code}: {detail}") from exc2
        raise RuntimeError(f"LLM HTTP {exc.code}: {detail[:400]}") from exc


def _extract_responses_text(payload: dict[str, Any]) -> str:
    """Pull plain text out of an Azure/OpenAI Responses API payload."""
    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return payload["output_text"].strip()

    chunks: list[str] = []
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        # Prefer final assistant messages; skip empty reasoning blocks.
        if item.get("type") == "reasoning":
            continue
        content = item.get("content")
        if isinstance(content, list):
            for c in content:
                if not isinstance(c, dict):
                    continue
                text = c.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text)
        elif isinstance(content, str) and content.strip():
            chunks.append(content)
    return "".join(chunks).strip()


def _extract_chat_text(payload: dict[str, Any]) -> str:
    return payload["choices"][0]["message"]["content"].strip()


def llm_text(
    prompt: str,
    *,
    model: str | None = None,
    system: str | None = None,
    temperature: float | None = 0.2,
    max_output_tokens: int = 900,
    timeout: float = 90.0,
    want_json: bool = False,
) -> str:
    """Single-turn completion returning raw text (works for chat and responses)."""
    cfg = resolve_config(model=model)

    if cfg.mode == "responses":
        input_messages: list[dict[str, str]] = []
        if system:
            input_messages.append({"role": "system", "content": system})
        input_messages.append({"role": "user", "content": prompt})
        body: dict[str, Any] = {
            "model": cfg.model,
            "input": input_messages,
            "max_output_tokens": max_output_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if want_json:
            body["text"] = {"format": {"type": "json_object"}}
        payload = _post_with_retry(cfg.url, cfg.headers, body, timeout=timeout)
        return _extract_responses_text(payload)

    # chat/completions
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = {
        "model": cfg.model,
        "messages": messages,
        "max_tokens": max_output_tokens,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if want_json:
        body["response_format"] = {"type": "json_object"}
    payload = _post_with_retry(cfg.url, cfg.headers, body, timeout=timeout)
    return _extract_chat_text(payload)


def _loads_lenient(text: str) -> Any:
    """Parse JSON, tolerating code fences or leading/trailing prose."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        # remove an optional leading language tag like ``json``
        if "\n" in text:
            text = text.split("\n", 1)[1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to locate the outermost JSON object or array.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("Could not parse JSON from model output")


def llm_json(
    prompt: str,
    *,
    model: str | None = None,
    system: str = "Return only valid JSON. No prose, no code fences.",
    temperature: float | None = 0.2,
    max_output_tokens: int = 1500,
    timeout: float = 120.0,
) -> Any:
    """Single-turn completion returning parsed JSON (object or array)."""
    text = llm_text(
        prompt,
        model=model,
        system=system,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        timeout=timeout,
        want_json=True,
    )
    return _loads_lenient(text)


# --- Backward-compatible wrappers (used by query_plan.py) ---------------------

def chat_completion_json(
    prompt: str,
    *,
    model: str | None = None,
    system: str = "Return only valid JSON.",
    temperature: float = 0.2,
    timeout: float = 60.0,
) -> dict[str, Any]:
    result = llm_json(
        prompt,
        model=model,
        system=system,
        temperature=temperature,
        timeout=timeout,
    )
    if not isinstance(result, dict):
        raise RuntimeError("Expected a JSON object from LLM")
    return result


def ping_openai(*, model: str | None = None) -> str:
    """Minimal connectivity check; returns model reply text."""
    return llm_text(
        "Reply with exactly: ok",
        model=model,
        system=None,
        temperature=0,
        max_output_tokens=16,
        timeout=60.0,
    )


# --- Bounded-backoff wrapper (used by quote-first pipeline) -------------------

_RETRYABLE_HTTP = {408, 409, 425, 429, 500, 502, 503, 504}


def _looks_retryable(exc: BaseException) -> bool:
    """Whether the exception looks like a transient failure worth retrying."""
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in _RETRYABLE_HTTP
    if isinstance(exc, urllib.error.URLError):
        return True
    msg = str(exc).lower()
    if "timeout" in msg or "timed out" in msg:
        return True
    if "getaddrinfo failed" in msg or "connection reset" in msg or "temporarily" in msg:
        return True
    if "http 429" in msg or "http 500" in msg or "http 502" in msg or "http 503" in msg or "http 504" in msg:
        return True
    return False


def llm_json_with_retry(
    prompt: str,
    *,
    model: str | None = None,
    system: str = "Return only valid JSON. No prose, no code fences.",
    temperature: float | None = 0.2,
    max_output_tokens: int = 1500,
    timeout: float = 120.0,
    max_attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    seed: int | None = None,
) -> Any:
    """``llm_json`` with bounded exponential backoff on transient failures.

    Does not retry on JSON parse errors (those indicate model output problems,
    not transient network issues). Callers should treat parse errors as topic-
    level failures rather than silently accepting empty results.

    ``seed`` is accepted for signature compatibility; current backends do not
    surface a deterministic seed, so it is recorded in caller manifests only.
    """
    _ = seed  # signature only; retained for reproducibility metadata.
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return llm_json(
                prompt,
                model=model,
                system=system,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                timeout=timeout,
            )
        except ValueError:
            # JSON parse failure; not transient.
            raise
        except (RuntimeError, urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt >= max_attempts or not _looks_retryable(exc):
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay += random.uniform(0, base_delay)
            time.sleep(delay)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("llm_json_with_retry exhausted attempts without exception")
