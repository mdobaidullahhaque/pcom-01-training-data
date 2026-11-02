"""Polite HTTP GET with retries, backoff and an on-disk response cache."""
import gzip
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
TIMEOUT_SECONDS = 180


def cache_path(url, cache_dir, compress=False):
    suffix = ".json.gz" if compress else ".json"
    return cache_dir / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}{suffix}"


def read_cache(path, compress):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if compress else data)


def write_cache(path, record, compress):
    data = json.dumps(record).encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(gzip.compress(data) if compress else data)
    os.replace(temporary, path)


def get_json(url, cache_dir=None, headers=None, compress=False):
    """Return (status, parsed body). Final answers, including 4xx, are cached; failures raise."""
    cache_dir = cache_dir or config.HTTP_CACHE_DIR
    path = cache_path(url, cache_dir, compress)
    if path.exists():
        cached = read_cache(path, compress)
        return cached["status"], cached["body"]

    for attempt in range(config.MAX_RETRIES + 1):
        wait = config.REQUEST_DELAY_SECONDS * 2 ** (attempt + 2)
        try:
            response = _session.get(url, timeout=TIMEOUT_SECONDS, headers=headers)
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
    write_cache(path, {"url": url, "status": response.status_code, "body": body}, compress)
    if compress:
        log.info(
            "Fetched %s: %.2f MB, cached as %.2f MB",
            url[:120], len(response.content) / 1024**2, path.stat().st_size / 1024**2,
        )
    time.sleep(config.REQUEST_DELAY_SECONDS)
    return response.status_code, body
