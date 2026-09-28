"""Fetching and caching what the repositories serve.

Every source in :mod:`hallsim.search` reaches the network through here:
one JSON getter carrying the package's user agent, a retry with backoff
for the keyless services that answer 429 under load, a concurrent fetch
for the index builds that are thousands of small requests, and a disk
cache under :data:`CACHE_ROOT` for what is worth keeping — a repository's
index, a deposit's file list, an ontology lookup.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import tempfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

log = logging.getLogger(__name__)

USER_AGENT = "hallsim-search"
#: Every cache lives under here, one directory per kind of thing kept.
CACHE_ROOT = Path.home() / ".cache" / "hallsim"

CACHE_TTL_DAYS = 30.0
INDEX_WORKERS = 16
#: Tries per URL during an index build, backing off between them.
INDEX_ATTEMPTS = 4
#: Gap between straggler refetches, once concurrency has been given up on.
SERIAL_RETRY_PAUSE = 0.5


def cache_dir(name: str) -> Path:
    """``CACHE_ROOT/name``, created."""
    path = CACHE_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_json(url: str, params: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        f"{url}?{urllib.parse.urlencode(params)}",
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def retrying(fetch, tries: int = 4, pause: float = 2.0, fatal=()):
    """Call ``fetch`` again after a transient failure, waiting longer each
    time; the keyless services answer 429 under load. An exception of a
    ``fatal`` type is raised at once."""
    for attempt in range(tries):
        try:
            return fetch()
        except Exception as exc:  # noqa: BLE001 - retried, then raised
            if isinstance(exc, fatal) or attempt == tries - 1:
                raise
            log.info("retrying after %s", str(exc)[:160])
            time.sleep(pause * (2**attempt))


def cached_json(key: str, fetch):
    """``fetch()``'s answer, kept on disk under ``key`` for good: a file
    list, an ontology lookup, anything a repository states once. A fetch
    that raises writes nothing, so a timeout is never a stored miss."""
    path = cache_dir("datasets") / (
        hashlib.sha1(key.encode()).hexdigest() + ".json"
    )
    if path.exists():
        return json.loads(path.read_text())
    value = fetch()
    path.write_text(json.dumps(value))
    return value


def index_path(name: str) -> Path:
    """Where a repository's cached index lives."""
    return cache_dir("discovery") / f"{name}.json"


def cached_index(
    name: str,
    build: "callable",
    ttl_days: float = CACHE_TTL_DAYS,
    refresh: bool = False,
) -> list[dict]:
    """``build()``'s records, cached on disk at :func:`index_path`.

    Rebuilt when older than ``ttl_days`` or when ``refresh``. A build failure
    with a stale cache present returns the stale cache rather than nothing —
    a repository being down should degrade the search, not empty it. A build
    that returns nothing counts as a failure: an empty listing is what a
    repository serves while it is warming up, and cached it would read as
    the repository holding nothing for a month.
    """
    path = index_path(name)
    fresh = (
        path.exists()
        and (time.time() - path.stat().st_mtime) < ttl_days * 86400.0
    )
    if fresh and not refresh:
        return json.loads(path.read_text())
    try:
        records = build()
        if not records:
            raise RuntimeError("the build returned no records")
    except Exception as exc:
        if path.exists():
            log.warning(
                "%s index rebuild failed (%s); using stale cache", name, exc
            )
            return json.loads(path.read_text())
        raise
    path.write_text(json.dumps(records))
    log.info("%s index built: %d records -> %s", name, len(records), path)
    return records


def fetch_many(
    urls: list[str],
    timeout: float = 30.0,
    workers: int = INDEX_WORKERS,
    attempts: int = INDEX_ATTEMPTS,
) -> list[dict | None]:
    """Fetch JSON from many URLs concurrently, preserving order.

    Index builds are thousands of small requests against a public API; serial
    fetching makes the first search a coffee break. A failed record is None
    rather than an exception — one bad row must not lose the index.

    Overload is retried with backoff. A loaded repository answers a request it
    would otherwise serve with 429 *or* with 500 — JWS returns 500 — so a
    status-code allowlist would drop two thirds of that index on the floor and
    still look like a complete build. Callers must therefore check how many
    rows came back None; :func:`cached_index` writes whatever it is handed.

    Stragglers get a final serial pass. Retrying in place keeps every worker
    hammering a source that is shedding load, and JWS's failures were measured
    to be transient rather than per-URL: the same 40 URLs that failed 29 times
    inside a concurrent batch all succeeded when spaced out.
    """

    def one(url, tries):
        for attempt in range(tries):
            try:
                return get_json(url, {}, timeout)
            except Exception:
                if attempt == tries - 1:
                    return None
                time.sleep(2.0**attempt * (0.5 + random.random()))
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        out = list(pool.map(lambda u: one(u, attempts), urls))

    missing = [i for i, rec in enumerate(out) if rec is None]
    if missing and len(missing) < len(urls):
        log.info("retrying %d straggler(s) serially", len(missing))
        for i in missing:
            time.sleep(SERIAL_RETRY_PAUSE)
            out[i] = one(urls[i], attempts)
    return out


def atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` beside ``path`` and rename it into place, so a reader
    in another process sees the old file or the new one, never a
    half-written one."""
    fd, tmp = tempfile.mkstemp(
        dir=path.parent, prefix=".tmp-", suffix=path.suffix
    )
    os.close(fd)
    try:
        Path(tmp).write_bytes(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def is_junk_archive_entry(name: str) -> bool:
    """Whether a zip entry is packaging debris rather than content.

    A zip made on macOS carries an AppleDouble resource fork beside every file
    (`._model.ode`, under `__MACOSX/`). They have the right extension and none
    of the content, so an extractor that trusts the suffix hands the importer
    a binary metadata blob — which is where ModelDB 35358's
    `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa2` came from.
    """
    parts = name.replace("\\", "/").split("/")
    return any(
        p == "__MACOSX" or p.startswith("._") or p == ".DS_Store"
        for p in parts
    )
