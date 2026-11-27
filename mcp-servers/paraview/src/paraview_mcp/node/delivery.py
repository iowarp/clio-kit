"""The image stream a remote view's pages share, put right for a page that has
no picture yet.

trame-vtk keeps one image stream per view for all its pages: the size images are
drawn at, their quality, whether a drag is under way, and the time of the last
image sent. A page that goes without saying goodbye (the host remounted the app,
the frame was dropped mid-drag) leaves that state as it was, and the next page can
get no picture at all: a drag that never ended holds back every still image, and
an image the stream thinks it already sent is not sent again."""

from __future__ import annotations

from typing import Any

PUSH = "viewport.image.push"


def _delivery(server: Any) -> Any:
    protocol = getattr(server, "protocol", None)
    if protocol is None:
        return None
    pair = protocol.getRPCMethod(PUSH)
    return pair[0] if pair else None


def fresh_stream(server: Any, view: Any, width: int, height: int) -> bool:
    """Size the view's stream to the asking page and clear what a page that went
    left behind, so the next push sends a full still image. True if it did."""
    delivery = _delivery(server)
    if delivery is None:
        return False
    view_id = view.GetGlobalIDAsString()
    tracking = getattr(delivery, "tracking_views", {}).get(view_id)
    if tracking is None:
        return False
    animating = getattr(delivery, "views_in_animations", None)
    if isinstance(animating, list):
        animating[:] = [v for v in animating if v != view_id]
    if width >= 10 and height >= 10:
        tracking["originalSize"] = (int(width), int(height))
    tracking["ratio"] = 1
    tracking["quality"] = 100
    tracking["mtime"] = 0
    tracking["enabled"] = True
    return True
