"""A frame is rendered at its own size, never tiled from a small viewer's view."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from paraview_mcp.node.reading import SceneReading


class _View:
    def __init__(self, size: list[int]) -> None:
        self.ViewSize = size


class _Simple:
    def __init__(self, view: _View, fail: bool = False) -> None:
        self.view = view
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def SaveScreenshot(self, path: str, view: _View, **kw: Any) -> None:
        self.calls.append({"size": list(view.ViewSize), **kw})
        if self.fail:
            raise RuntimeError("render failed")
        Path(path).write_bytes(b"png")


def _reader(size: list[int], fail: bool = False) -> SceneReading:
    reader = SceneReading.__new__(SceneReading)
    reader.view = _View(size)
    reader.simple = _Simple(reader.view, fail)
    return reader


def test_frame_renders_once_at_its_own_size(tmp_path: Path) -> None:
    reader = _reader([3, 2])
    png = reader._render_png(tmp_path / "f.png", 320, 200)
    assert png == b"png"
    (call,) = reader.simple.calls
    assert call["size"] == [320, 200]
    assert call["ImageResolution"] == [320, 200]
    assert reader.view.ViewSize == [3, 2]


def test_view_gets_its_size_back_when_the_render_fails(tmp_path: Path) -> None:
    reader = _reader([640, 400], fail=True)
    with pytest.raises(RuntimeError):
        reader.screenshot(tmp_path / "f.png", 1280, 800)
    assert reader.view.ViewSize == [640, 400]
