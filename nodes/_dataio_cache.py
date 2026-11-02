"""DataFrame cache for the data_io reader family (``comfydl.dataio.cache``).

Caches the **raw parsed ``DataFrame``** (not the tensor dataset) so that every
cache hit rebuilds fresh tensors via ``CdlDataset.from_dataframe`` — the
cached object is never handed to the graph, which removes two whole classes
of aliasing bugs (in-place mutation poisoning the cache, and inference-mode
tensors leaking into non-inference contexts).

Correctness invariants (each one fixes a real NNT defect, see refs_survey +
smallplans §0.82c):

1. the cache dict is initialised at import time (NNT's ``_data_cache`` never
   was — first use crashed);
2. the cache key covers **everything that affects the parsed DataFrame**:
   reader name, normalised path, ``mtime_ns``+size of the source file, and
   every parse parameter (delimiter/header/sheet/query text) — a changed file
   or a changed parameter therefore produces a new key (NNT's key skipped
   both file contents and several parameters);
3. the shard parameters (``start_idx``/``num_samples``) are deliberately NOT
   part of the key: the *full* table is cached and the shard is applied
   afterwards, so a shard change can never return stale data — correctness
   holds by construction instead of by key discipline, and N shards share one
   cached parse instead of multiplying memory N times.

Memory discipline (RAM is expensive — this is a budget, not a pagefile):

* the cache holds at most ``MAX_ENTRIES`` entries totalling at most
  ``MAX_TOTAL_BYTES`` of DataFrame memory (FIFO eviction, insertion order);
* a single DataFrame larger than the whole budget is not cached at all;
* accounting uses ``df.memory_usage(deep=True).sum()`` computed once at
  store time.

Switch/telemetry (ADR-D1): the cache registers the hot switch
``comfydl.dataio.cache`` in ``comfy.profiling.switch_registry`` (default ON,
self-metered).  Only the cache layer's *own* work (key building, stat,
lookup, store, eviction) is metered via ``record_self_overhead`` — never the
underlying file read.  When the registry is unavailable (bare-submodule
import, unit-test sandboxes) the cache degrades gracefully: lookups are
skipped, readers behave exactly as before, nothing raises.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Dict, Tuple

SWITCH_ID = "comfydl.dataio.cache"

#: Memory budget for the whole cache (bytes of DataFrame payload).
MAX_TOTAL_BYTES = 512 * 1024 * 1024
#: Hard cap on entry count (secondary guard against metadata bloat).
MAX_ENTRIES = 64

_LOCK = threading.Lock()
_CACHE: "OrderedDict[str, Any]" = OrderedDict()
_CACHED_BYTES = 0
_STATS: Dict[str, int] = {
    "hits": 0,
    "misses": 0,
    "evictions": 0,
    "rejected_too_large": 0,
}


def _registry():
    """The switch registry, or ``None`` when running outside the host."""
    try:
        from comfy.profiling import switch_registry

        return switch_registry
    except Exception:  # noqa: BLE001 - bare-submodule / sandbox import
        return None


def register() -> None:
    """Declare the switch (idempotent; no-op outside the host)."""
    reg = _registry()
    if reg is None:
        return
    reg.register_switch(
        SWITCH_ID,
        default=True,
        hot_toggle=True,
        self_meter=True,
        hard_disable_check=None,
    )


def enabled() -> bool:
    """Whether the cache participates in the run path (never raises)."""
    reg = _registry()
    if reg is None:
        return False
    try:
        return bool(reg.is_enabled(SWITCH_ID))
    except Exception:  # noqa: BLE001
        return False


def note_overhead(seconds: float) -> None:
    """Feed the self-metering ledger (never raises)."""
    reg = _registry()
    if reg is None:
        return
    try:
        reg.record_self_overhead(SWITCH_ID, seconds)
    except Exception:  # noqa: BLE001
        pass


@contextmanager
def measure():
    """Time the cache layer's own work and bill it to the switch (ADR-D1)."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        note_overhead(time.perf_counter() - t0)


# --------------------------------------------------------------------- #
# Key construction
# --------------------------------------------------------------------- #


def md5_hex(text: str) -> str:
    """Stable content hash for inline payloads (query text, inline tables)."""
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


def file_stamp(path: str) -> Tuple[Any, ...]:
    """``(normcase abspath, mtime_ns, size)`` — ``None`` fields when unknown."""
    try:
        norm = os.path.normcase(os.path.abspath(path))
        st = os.stat(norm)
        return (norm, st.st_mtime_ns, st.st_size)
    except OSError:
        return (os.path.normcase(os.path.abspath(path)), None, None)


def cache_key(*parts: Any) -> str:
    """md5 over the repr of every part that affects the parsed DataFrame."""
    return hashlib.md5(repr(parts).encode("utf-8", errors="replace")).hexdigest()


# --------------------------------------------------------------------- #
# Lookup / store
# --------------------------------------------------------------------- #


def lookup(key: str) -> Tuple[bool, Any]:
    """Return ``(hit, df)``; a hit is refreshed to the FIFO's hot end."""
    with _LOCK:
        df = _CACHE.get(key)
        if df is None:
            _STATS["misses"] += 1
            return False, None
        _CACHE.move_to_end(key)
        _STATS["hits"] += 1
        return True, df


def store(key: str, df: Any) -> None:
    """Insert/refresh a DataFrame under the byte budget (FIFO eviction)."""
    global _CACHED_BYTES
    try:
        nbytes = int(df.memory_usage(deep=True).sum())
    except Exception:  # noqa: BLE001 - unaccountable frame: do not cache
        return
    with _LOCK:
        old = _CACHE.pop(key, None)
        if old is not None:
            try:
                _CACHED_BYTES -= int(old.memory_usage(deep=True).sum())
            except Exception:  # noqa: BLE001
                _CACHED_BYTES = 0
        if nbytes > MAX_TOTAL_BYTES:
            _STATS["rejected_too_large"] += 1
            return
        while _CACHE and (
            len(_CACHE) >= MAX_ENTRIES
            or _CACHED_BYTES + nbytes > MAX_TOTAL_BYTES
        ):
            _, evicted = _CACHE.popitem(last=False)
            _CACHED_BYTES -= int(evicted.memory_usage(deep=True).sum())
            _STATS["evictions"] += 1
        _CACHE[key] = df
        _CACHED_BYTES += nbytes


def clear() -> None:
    """Drop everything (tests / operator escape hatch)."""
    global _CACHED_BYTES
    with _LOCK:
        _CACHE.clear()
        _CACHED_BYTES = 0


def stats() -> Dict[str, int]:
    """A snapshot of the counters plus the current memory ledger."""
    with _LOCK:
        out = dict(_STATS)
        out["entries"] = len(_CACHE)
        out["cached_bytes"] = int(_CACHED_BYTES)
        out["budget_bytes"] = MAX_TOTAL_BYTES
        out["max_entries"] = MAX_ENTRIES
        return out
