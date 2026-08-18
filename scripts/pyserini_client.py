#!/usr/bin/env python3
"""Pyserini REST client for ClimbMix search and document fetch."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "http://api.castorini.uwaterloo.ca"
DEFAULT_INDEX = "climbmix-400b"
AUTH_HEADER_RE = re.compile(r"Authorization:\s*([^\"'\n]+)", re.IGNORECASE)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def resolve_auth_header(root: Path | None = None) -> str:
    root = root or project_root()
    curlrc = root / ".curlrc.pyserini-rest"
    if curlrc.exists():
        match = AUTH_HEADER_RE.search(curlrc.read_text(encoding="utf-8"))
        if match:
            return match.group(1).strip()

    token = os.environ.get("PYSERINI_API_TOKEN")
    if not token:
        token = parse_env_file(root / ".env.local").get("PYSERINI_API_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing Pyserini token. Set PYSERINI_API_TOKEN in .env.local "
            "or create .curlrc.pyserini-rest."
        )
    token = token.strip()
    if token.lower().startswith("bearer "):
        return token
    return f"Bearer {token}"


def search(
    query: str,
    *,
    hits: int = 100,
    base_url: str = DEFAULT_BASE_URL,
    index: str = DEFAULT_INDEX,
    auth_header: str | None = None,
    timeout: float = 60.0,
    retries: int = 3,
) -> list[dict[str, Any]]:
    auth_header = auth_header or resolve_auth_header()
    params = urllib.parse.urlencode({"query": query, "hits": hits})
    url = f"{base_url.rstrip('/')}/v1/{index}/search?{params}"
    request = urllib.request.Request(url, headers={"Authorization": auth_header})

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload.get("candidates") or []
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429 and attempt + 1 < retries:
                time.sleep(min(60.0, 5.0 * (2**attempt)))
                continue
            if attempt + 1 < retries:
                time.sleep(2**attempt)
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Search failed for query after {retries} attempts: {last_error}")


def fetch_document(
    docid: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    index: str = DEFAULT_INDEX,
    auth_header: str | None = None,
    timeout: float = 60.0,
    retries: int = 3,
) -> dict[str, Any]:
    """Fetch one ClimbMix document by ID via GET /v1/{index}/doc/{docid}."""
    auth_header = auth_header or resolve_auth_header()
    url = f"{base_url.rstrip('/')}/v1/{index}/doc/{urllib.parse.quote(docid, safe='')}"
    request = urllib.request.Request(url, headers={"Authorization": auth_header})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429 and attempt + 1 < retries:
                time.sleep(min(60.0, 5.0 * (2**attempt)))
                continue
            if attempt + 1 < retries:
                time.sleep(2**attempt)
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Doc fetch failed for {docid} after {retries} attempts: {last_error}")


def load_topics_tsv(path: Path) -> list[tuple[str, str]]:
    topics: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        topic_id, _, text = line.partition("\t")
        if not topic_id or not text:
            raise ValueError(f"Invalid topic line in {path}: {line[:80]!r}")
        topics.append((topic_id.strip(), text.strip()))
    return topics
