"""A blank frame says why: the camera checks need no ParaView."""

from __future__ import annotations

from typing import Any

import pytest

from paraview_mcp.node.camera import _check_camera
from paraview_mcp.node.errors import CommandError
from paraview_mcp.node.reading import SceneReading

BOX = [[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]]


class Looking(SceneReading):
    def __init__(self, **camera: Any) -> None:
        self.camera = {
            "view_up": [0.0, 1.0, 0.0],
            "view_angle": 30.0,
            "parallel_scale": 1.0,
            "parallel_projection": False,
            **camera,
        }

    def read_camera(self) -> dict[str, Any]:
        return self.camera


def test_a_camera_facing_the_data_misses_nothing() -> None:
    eye = Looking(position=[0.5, 0.5, 5.0], focal_point=[0.5, 0.5, 0.5])
    assert eye._camera_misses(BOX) == ""


def test_a_camera_turned_away_or_inside_says_so() -> None:
    away = Looking(position=[0.5, 0.5, 5.0], focal_point=[0.5, 0.5, 9.0])
    assert away._camera_misses(BOX) == "the data is behind the camera"
    aside = Looking(position=[0.5, 0.5, 5.0], focal_point=[9.0, 0.5, 5.0])
    assert "field of view" in aside._camera_misses(BOX)
    inside = Looking(position=[0.5, 0.5, 0.5], focal_point=[0.5, 0.5, 0.0])
    assert "inside" in inside._camera_misses(BOX)


def test_a_camera_that_draws_nothing_is_refused() -> None:
    with pytest.raises(CommandError):
        _check_camera(
            {"position": [1, 1, 1], "focal_point": [1, 1, 1], "view_up": [0, 1, 0]}
        )
    with pytest.raises(CommandError):
        _check_camera(
            {"position": [0, 0, 5], "focal_point": [0, 0, 0], "view_up": [0, 0, 1]}
        )
    _check_camera(
        {"position": [0, 0, 5], "focal_point": [0, 0, 0], "view_up": [0, 1, 0]}
    )
