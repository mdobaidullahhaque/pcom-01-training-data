"""Polite HTTP GET with retries, backoff and an on-disk response cache."""
import hashlib
import json
import logging
import os
import time

import requests

import config

log = logging.getLogger("http_client")

_session = requests.Session()
_session.headers["User-Agent"] = config.USER_AGENT
RETRY_STATUSES = {429, 500, 502, 503, 504}


def cache_path(url, cache_dir):
    return cache_dir / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}.json"


def get_json(url, cache_dir=None):
    """Return (status, parsed body). Final answers, including 4xx, are cached; failures raise."""
    cache_dir = cache_dir or config.HTTP_CACHE_DIR
    path = cache_path(url, cache_dir)
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        return cached["status"], cached["body"]

    for attempt in range(config.MAX_RETRIES + 1):
        wait = config.REQUEST_DELAY_SECONDS * 2 ** (attempt + 2)
        try:
            response = _session.get(url, timeout=60)
        except requests.RequestException as error:
            reason = repr(error)
        else:
            if response.status_code not in RETRY_STATUSES:
                break
            reason = f"HTTP {response.status_code}"
            wait = max(wait, float(response.headers.get("Retry-After") or 0))
        if attempt == config.MAX_RETRIES:
            raise RuntimeError(f"Giving up on {url[:200]}: {reason}")
        log.warning("Retry %d in %.0f s for %s: %s", attempt + 1, wait, url[:120], reason)
        time.sleep(wait)

    try:
        body = response.json()
    except ValueError:
        body = None
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"url": url, "status": response.status_code, "body": body}), encoding="utf-8"
    )
    os.replace(temporary, path)
    time.sleep(config.REQUEST_DELAY_SECONDS)
    return response.status_code, body
