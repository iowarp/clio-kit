"""How long the view takes to draw a frame, so a slow view can say how slow, and why."""

from __future__ import annotations

import time
from typing import Any

#: Renders closer together than this belong to one burst (a screenshot's passes, a drag).
BURST_GAP_SECONDS = 0.25


class FrameClock:
    """Times the renders of one view's window: captures and remote frames alike.

    ``seconds`` is the slowest single render of the latest burst. It stays ``None``
    until a frame is drawn, and for a view whose window lives in another process (a
    pvserver session), where no frame is timed here.
    """

    def __init__(self, view: Any) -> None:
        self.seconds: float | None = None
        self._started = 0.0
        self._ended = 0.0
        window = view.GetRenderWindow()
        if window is None:
            return
        window.AddObserver("StartEvent", self._start)
        window.AddObserver("EndEvent", self._end)

    def _start(self, *_: Any) -> None:
        self._started = time.perf_counter()

    def _end(self, *_: Any) -> None:
        if not self._started:
            return
        now = time.perf_counter()
        took = round(now - self._started, 2)
        same_burst = self._started - self._ended < BURST_GAP_SECONDS
        self.seconds = max(took, self.seconds or 0.0) if same_burst else took
        self._ended = now


def volume_grids(scene: dict[str, Any], proxies: dict[str, Any]) -> list[str]:
    """The data set types of the visible objects drawn as a volume."""

    grids = set()
    for item in scene.get("objects", []):
        display = item.get("display") or {}
        proxy = proxies.get(item["id"])
        if proxy is None or not display.get("visible", True):
            continue
        if display.get("representation") == "Volume":
            grids.add(str(proxy.GetDataInformation().GetDataSetTypeAsString()))
    return sorted(grids)
