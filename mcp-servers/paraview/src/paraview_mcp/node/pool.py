"""Readers of a growing source, shared by the views of one node half.

Views of the same growing source (one per variable, say) read it through one
reader: each new step is looked for on disk, re-read and measured once for the
source rather than once per view. A view that finds its source already re-read
by another view moves to the new step without looking at the disk again.

The reader caches the data of one time, so views of one source at different
steps make it read again as each draws; following views all ask for the latest
step, which is read once.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

#: Seconds a look at a shared source's disk stands for the other views of it.
CHECK_SHARED = 1.5
#: Colour range measurements kept, newest last.
RANGES_KEPT = 64


@dataclass
class Source:
    """One shared reader and what its views know of its disk."""

    key: str
    proxy: Any
    users: dict[int, Any] = field(default_factory=dict)
    #: Bumped each time the reader re-reads its files.
    generation: int = 0
    #: The per-reader growth state the views share: the disk key the reader holds,
    #: the steps its follow copy holds, a settling folder, and the last disk look.
    state: dict[str, Any] = field(default_factory=dict)
    checked_at: float = 0.0
    disk_changed: float | None = None


#: The engine dicts a shared reader's growth state lives in, per item id.
STATE = ("_growth", "_settling", "_held", "_disk_seen")


class Readers:
    """The node half's shared readers, by what they read."""

    def __init__(self) -> None:
        self._sources: dict[str, Source] = {}
        self.ranges: dict[tuple[Any, ...], dict[str, list[float]] | None] = {}

    @staticmethod
    def key_of(item: dict[str, Any]) -> str:
        return json.dumps(
            [item.get("path"), item.get("reader"), item.get("reader_properties")],
            sort_keys=True,
            default=str,
        )

    @staticmethod
    def folder_of(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def get(self, key: str, simple: Any) -> Source | None:
        """The live shared reader of ``key``, if one is open."""
        source = self._sources.get(key)
        if source is None:
            return None
        live = {id(p) for p in simple.GetSources().values()}
        if id(source.proxy) not in live:
            del self._sources[key]
            return None
        return source

    def of(self, proxy: Any) -> Source | None:
        for source in self._sources.values():
            if source.proxy is proxy:
                return source
        return None

    def named(self, key: str) -> Source | None:
        return self._sources.get(key)

    def add(self, key: str, proxy: Any, user: Any) -> Source:
        source = Source(key, proxy, {id(user): user})
        self._sources[key] = source
        return source

    def join(self, source: Source, user: Any) -> None:
        source.users[id(user)] = user

    def leave(self, proxy: Any, user: Any) -> bool:
        """``user`` no longer shows ``proxy``; whether other views still do."""
        source = self.of(proxy)
        if source is None:
            return False
        source.users.pop(id(user), None)
        if source.users:
            return True
        self._sources.pop(source.key, None)
        self.ranges = {k: v for k, v in self.ranges.items() if k[0] != source.key}
        return False

    def pull(self, source: Source, engine: Any, item_id: str) -> None:
        """Give ``engine`` the shared growth state of its reader ``item_id``."""
        for name in STATE:
            mine = getattr(engine, name)
            if name in source.state:
                mine[item_id] = source.state[name]
            else:
                mine.pop(item_id, None)
        if source.disk_changed is not None:
            engine.disk_changed = max(engine.disk_changed or 0.0, source.disk_changed)

    def push(self, source: Source, engine: Any, item_id: str) -> None:
        """Keep what ``engine`` learnt of its reader ``item_id`` for the others."""
        for name in STATE:
            mine = getattr(engine, name)
            if item_id in mine:
                source.state[name] = mine[item_id]
            else:
                source.state.pop(name, None)
        if engine.disk_changed is not None:
            source.disk_changed = max(source.disk_changed or 0.0, engine.disk_changed)

    def looked(self, source: Source) -> bool:
        """Whether another view looked at this source's disk just now."""
        return time.monotonic() - source.checked_at < CHECK_SHARED

    # -- colour ranges ---------------------------------------------------------
    def range_of(self, key: tuple[Any, ...]) -> tuple[bool, Any]:
        if key in self.ranges:
            return True, self.ranges[key]
        return False, None

    def keep_range(self, key: tuple[Any, ...], value: Any) -> None:
        self.ranges.pop(key, None)
        self.ranges[key] = value
        while len(self.ranges) > RANGES_KEPT:
            del self.ranges[next(iter(self.ranges))]
