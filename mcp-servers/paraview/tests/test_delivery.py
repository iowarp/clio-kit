"""A page with no picture puts the view's shared image stream right."""

from __future__ import annotations

from paraview_mcp.node.delivery import PUSH, fresh_stream


class _View:
    def GetGlobalIDAsString(self) -> str:
        return "7"


class _Delivery:
    def __init__(self) -> None:
        self.views_in_animations = ["7", "9", "7"]
        self.tracking_views = {
            "7": {
                "originalSize": (1398, 862),
                "ratio": 0.5,
                "quality": 50,
                "mtime": 41,
                "enabled": False,
            },
        }


class _Protocol:
    def __init__(self, delivery: _Delivery | None) -> None:
        self.delivery = delivery

    def getRPCMethod(self, name: str):
        return (self.delivery, None) if name == PUSH and self.delivery else None


class _Server:
    def __init__(self, protocol: _Protocol | None) -> None:
        self.protocol = protocol


def test_a_page_that_went_mid_drag_does_not_hold_back_the_next_pages_picture() -> None:
    delivery = _Delivery()
    assert fresh_stream(_Server(_Protocol(delivery)), _View(), 348, 806)
    assert delivery.views_in_animations == ["9"]
    assert delivery.tracking_views["7"] == {
        "originalSize": (348, 806),
        "ratio": 1,
        "quality": 100,
        "mtime": 0,
        "enabled": True,
    }


def test_a_page_too_small_to_draw_keeps_the_streams_size() -> None:
    delivery = _Delivery()
    assert fresh_stream(_Server(_Protocol(delivery)), _View(), 0, 0)
    assert delivery.tracking_views["7"]["originalSize"] == (1398, 862)


def test_nothing_to_put_right_before_a_page_subscribes() -> None:
    assert not fresh_stream(_Server(None), _View(), 348, 806)
    assert not fresh_stream(_Server(_Protocol(None)), _View(), 348, 806)
    empty = _Delivery()
    empty.tracking_views = {}
    assert not fresh_stream(_Server(_Protocol(empty)), _View(), 348, 806)
