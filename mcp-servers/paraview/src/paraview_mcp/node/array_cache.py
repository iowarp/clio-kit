"""How long the arrays of a scene sent to the browser stay ready for it to fetch.

trame-vtk caches every array of a scene it serializes, by hash, and the browser
fetches them by hash once it has the scene. After each serialization trame-vtk drops
the arrays only its cache holds that it last serialized over 20 s before. The merged
polydata ParaView 6's datasets go out as (serializers.py) are new arrays each time,
held by the cache alone, so over a slow link a large scene still being fetched lost
its arrays as soon as a later step was serialized: the fetch failed with a KeyError,
and the browser never drew that scene or any after it.

Here an array goes only once it is in none of the last two serializations (one
update sends a scene twice: as a change, and whole for a page that opens later), and
either no fetch has come for ``QUIET`` seconds or it is in none of the last ``CAP``.
A large scene over a slow link arrives late, but whole.
"""

from __future__ import annotations

import time
from typing import Any

#: Serializations whose arrays are always kept: the newest update's two.
KEEP = 2
#: Serializations whose arrays are kept at most, while fetches go on.
CAP = 8
#: Seconds without a fetch after which arrays older than ``KEEP`` go.
QUIET = 30.0


def install() -> bool:
    """Patch trame-vtk's array cache, once per process; whether it was patched now."""
    from trame_vtk.modules.vtk.serializers.synchronization_context import (
        SynchronizationContext,
    )

    if getattr(SynchronizationContext, "clio_retention", False):
        return False
    cache = SynchronizationContext.cache_data_array
    fetch = SynchronizationContext.get_cached_data_array

    def cache_data_array(self: Any, p_md5: str, data: Any) -> None:
        if isinstance(data, dict):
            data["clio_gen"] = getattr(self, "clio_gen", 0)
        cache(self, p_md5, data)

    def get_cached_data_array(self: Any, p_md5: str, *args: Any, **kwargs: Any) -> Any:
        self.clio_fetched = time.time()
        if p_md5 not in self.data_array_cache:
            print(f"the browser asked for an array no longer kept: {p_md5}", flush=True)
        return fetch(self, p_md5, *args, **kwargs)

    def check_for_arrays_to_release(self: Any, time_window: float = 20) -> None:
        now = time.time()
        gen = getattr(self, "clio_gen", 0)
        self.clio_gen = gen + 1
        quiet = now - getattr(self, "clio_fetched", 0.0) >= QUIET
        cut_off = now - time_window
        gone = []
        for sha, record in self.data_array_cache.items():
            if record["array"].GetReferenceCount() != 1 or record["ts"] >= cut_off:
                continue
            age = gen - record.get("clio_gen", gen - CAP)
            if age >= CAP or (age >= KEEP and quiet):
                gone.append(sha)
        for sha in gone:
            del self.data_array_cache[sha]

    SynchronizationContext.cache_data_array = cache_data_array
    SynchronizationContext.get_cached_data_array = get_cached_data_array
    SynchronizationContext.check_for_arrays_to_release = check_for_arrays_to_release
    SynchronizationContext.clio_retention = True
    return True
