"""Caching for the pipeline inspector.

The console re-runs stages as a user drags a slider, so the same frame is
processed many times with one parameter different. Recomputing every stage each
time would make the interface feel like a batch job; caching the stages that did
not change makes it feel like an instrument.

The cache key is a hash of the frame identity plus **every upstream parameter**.
Hashing only the stage's own parameters would be wrong in a way that is very
hard to notice: the cached S5 output for one shadow threshold would be reused
after S3's learning rate changed, so the user would see a stale mask and
attribute it to the parameter they just moved.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

__all__ = ["StageCache", "cache_key", "LruCache"]

T = TypeVar("T")


def cache_key(source: str, frame_index: int, upstream_params: list[dict[str, Any]]) -> str:
    """A stable key for one stage's output on one frame.

    ``upstream_params`` must include the parameters of *every* stage up to and
    including this one, in order. JSON with sorted keys makes the encoding
    canonical, so two dictionaries that differ only in insertion order do not
    produce two cache entries.
    """
    payload = json.dumps(
        {"source": source, "frame": frame_index, "params": upstream_params},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


@dataclass(slots=True)
class _Entry(Generic[T]):
    value: T
    cost_ms: float


class LruCache(Generic[T]):
    """A bounded least-recently-used cache.

    Bounded because the values are frames: a 720p mask is 0.9 MB and a colour
    frame 2.7 MB, so an unbounded cache of stage outputs exhausts memory in a
    couple of minutes of slider dragging. The bound is in entries rather than
    bytes because every entry here is within a small factor of the same size.
    """

    def __init__(self, capacity: int = 64) -> None:
        if capacity < 1:
            raise ValueError(f"cache capacity must be at least 1, got {capacity}")
        self.capacity = capacity
        self._entries: OrderedDict[str, _Entry[T]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> T | None:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return entry.value

    def put(self, key: str, value: T, cost_ms: float = 0.0) -> None:
        self._entries[key] = _Entry(value=value, cost_ms=cost_ms)
        self._entries.move_to_end(key)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict[str, float]:
        return {
            "entries": float(len(self._entries)),
            "capacity": float(self.capacity),
            "hits": float(self.hits),
            "misses": float(self.misses),
            "hit_rate": self.hit_rate,
            "saved_ms": float(sum(e.cost_ms for e in self._entries.values())),
        }


class StageCache:
    """Per-stage caches keyed on the whole upstream parameter chain."""

    def __init__(self, capacity_per_stage: int = 24) -> None:
        self._caches: dict[str, LruCache[Any]] = {}
        self._capacity = capacity_per_stage

    def cache_for(self, stage_id: str) -> LruCache[Any]:
        if stage_id not in self._caches:
            self._caches[stage_id] = LruCache(self._capacity)
        return self._caches[stage_id]

    def get(self, stage_id: str, key: str) -> Any | None:
        return self.cache_for(stage_id).get(key)

    def put(self, stage_id: str, key: str, value: Any, cost_ms: float = 0.0) -> None:
        self.cache_for(stage_id).put(key, value, cost_ms)

    def invalidate_from(self, stage_id: str, order: list[str]) -> None:
        """Drop this stage's cache and every stage after it.

        Changing a parameter invalidates that stage and everything downstream,
        and nothing upstream. Clearing the whole cache instead would be correct
        but would throw away exactly the entries that make the next re-run fast.
        """
        if stage_id not in order:
            self._caches.clear()
            return
        for later in order[order.index(stage_id) :]:
            self._caches.pop(later, None)

    def clear(self) -> None:
        self._caches.clear()

    def stats(self) -> dict[str, dict[str, float]]:
        return {stage_id: cache.stats() for stage_id, cache in self._caches.items()}
