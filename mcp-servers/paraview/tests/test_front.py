"""The front without ParaView: tools, the view resource, attach decisions."""

from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path

import pytest
from fastmcp import Client

from paraview_mcp import attach, config, present
from paraview_mcp.config import Settings, Target
from paraview_mcp.errors import ParaViewError, no_paraview
from paraview_mcp.server import mcp
from paraview_mcp.transport import LocalTransport, classify_ssh_failure

EXPECTED_TOOLS = {
    "open_view",
    "create_filter",
    "remove_scene_object",
    "set_representation",
    "set_timestep",
    "set_camera",
    "fit_camera",
    "inspect_selection",
    "measure_field",
    "capture",
    "export_artifact",
    "export_scene",
    "scene_state",
}


async def test_tools_list_without_paraview() -> None:
    async with Client(mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
    assert set(tools) == EXPECTED_TOOLS
    assert tools["open_view"].meta["ui"]["resourceUri"] == "ui://paraview/view"
    # Only the opening call carries the view.
    assert all("ui" not in (t.meta or {}) for n, t in tools.items() if n != "open_view")
    assert not any("connect" in name for name in tools)


async def test_view_resource_is_an_mcp_app() -> None:
    async with Client(mcp) as client:
        listed = await client.list_resources()
        content = await client.read_resource("ui://paraview/view")
    assert [r.mime_type for r in listed] == ["text/html;profile=mcp-app"]
    html = content[0].text
    assert "ui/initialize" in html
    assert "ai.iowarp.clio/service" in html
    assert "ui/update-model-context" in html
    assert "'fullscreen'" in html
    # Context reports are throttled to ten a second.
    assert "100 - (Date.now() - lastSent)" in html
    assert "127.0.0.1" not in html and "localhost" not in html


def test_settings_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARAVIEW_ADDRESS", "node08")
    monkeypatch.setenv("PARAVIEW_VIA", "${need.via}")
    monkeypatch.setenv("PARAVIEW_PREPARE", "  module load paraview ")
    monkeypatch.setenv("PARAVIEW_IDLE_SECONDS", "5")
    settings = config.from_environment()
    assert settings.target == Target("node08", None, "module load paraview")
    assert settings.idle_seconds == 5.0
    assert settings.target.describe() == "on node08"
    assert Target("node08", "login").describe() == "on node08 through login"
    monkeypatch.delenv("PARAVIEW_ADDRESS")
    assert not config.from_environment().target.remote


def test_platform_tags() -> None:
    pip, uv = attach.platform_tags(
        {"platform": "linux-x86_64", "machine": "x86_64", "libc": ["glibc", "2.39"]}
    )
    assert "manylinux_2_28_x86_64" in pip and "manylinux2014_x86_64" in pip
    assert uv == "x86_64-manylinux_2_28"
    pip, uv = attach.platform_tags(
        {"platform": "linux-aarch64", "machine": "aarch64", "libc": ["glibc", "2.17"]}
    )
    assert uv == "aarch64-manylinux_2_17" and "manylinux_2_28_aarch64" not in pip
    pip, uv = attach.platform_tags(
        {"platform": "macosx-11.0-arm64", "machine": "arm64"}
    )
    assert pip == ["macosx_11_0_arm64"] and uv == "aarch64-apple-darwin"


def test_build_script_pins_packages_and_targets_pvpython_python() -> None:
    script = attach.build_script(
        "/home/u/.clio/paraview/3.0.0",
        {"python": "3.12", "platform": "linux-x86_64", "machine": "x86_64"},
        "/opt/pv/bin/pvpython",
    )
    for package in config.NODE_PACKAGES:
        assert package in script
    assert "--python-version 3.12" in script
    assert "--only-binary=:all:" in script
    assert "CLIO_BUILD=ok" in script


def test_code_bundle_carries_the_node_half() -> None:
    bundle, digest = attach.code_bundle()
    _, same = attach.code_bundle()
    assert digest == same and len(digest) == 16
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as archive:
        names = set(archive.getnames())
    assert {
        "run_node.py",
        "HASH",
        "paraview_node/main.py",
        "paraview_node/engine.py",
    } <= names


def test_parse_markers() -> None:
    marks = attach.parse_markers(
        "noise\nCLIO_DIR=/h/.clio\nCLIO_PVPYTHON=\nCLIO_X=a=b\n"
    )
    assert marks == {"DIR": "/h/.clio", "PVPYTHON": "", "X": "a=b"}


@pytest.mark.parametrize(
    ("stderr", "code"),
    [
        (
            "user@locked: Permission denied (password,keyboard-interactive).",
            "interactive_sign_in",
        ),
        ("user@host: Permission denied (publickey).", "key_rejected"),
        ("Host key verification failed.", "host_key_unknown"),
        (
            "ssh: Could not resolve hostname nowhere: Name or service not known",
            "unknown_host",
        ),
        ("ssh: connect to host 10.0.0.1 port 22: Connection timed out", "unreachable"),
        ("something else", "ssh_failed"),
    ],
)
def test_ssh_failures_are_typed(stderr: str, code: str) -> None:
    error = classify_ssh_failure("locked", stderr)
    assert error.code == code
    assert "locked" in error.message


def test_no_paraview_message_says_how_to_point_the_connector() -> None:
    error = no_paraview("on this machine")
    assert error.code == "no_paraview"
    assert "Where ParaView runs" in error.message
    assert "Command to run first" in error.message


def test_local_attach_without_pvpython_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # AQ35: a PATH with no pvpython on it.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    attacher = attach.Attacher(Settings(target=Target()))
    with pytest.raises(ParaViewError) as caught:
        attacher.ensure()
    assert caught.value.code == "no_paraview"
    assert "on this machine" in caught.value.message
    assert (tmp_path / ".clio" / "paraview" / config.SERVER_VERSION).is_dir()
    assert (
        oct(os.stat(tmp_path / ".clio" / "paraview" / config.SERVER_VERSION).st_mode)[
            -3:
        ]
        == "700"
    )


async def test_scene_tool_without_paraview_returns_the_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from paraview_mcp import server

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.delenv("PARAVIEW_ADDRESS", raising=False)
    monkeypatch.setattr(server, "_attacher", None)
    async with Client(mcp) as client:
        result = await client.call_tool(
            "open_view", {"path": "/data/x.vti"}, raise_on_error=False
        )
    assert result.is_error
    text = result.content[0].text
    assert "No ParaView was found on this machine" in text
    assert "[no_paraview]" in text


def test_prepare_line_runs_first(tmp_path: Path) -> None:
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "pvpython").write_text("#!/bin/sh\necho fake\n")
    (fake / "pvpython").chmod(0o755)
    transport = LocalTransport(Target(prepare=f"export PATH={fake}:$PATH"))
    done = transport.run("command -v pvpython")
    assert done.stdout.strip() == str(fake / "pvpython")


def test_present_scene_text_names_the_scene_and_keeps_no_secret() -> None:
    summary = {
        "scene": "wave-1",
        "dataset": {"path": "/d/wave.vti", "growing": True},
        "objects": [
            {
                "id": "o1",
                "label": "wave.vti",
                "kind": "reader",
                "type": "XMLImageDataReader",
                "display": {"visible": True, "representation": "Outline"},
                "data": {
                    "points": 8,
                    "cells": 1,
                    "arrays": [
                        {
                            "name": "p",
                            "association": "POINTS",
                            "components": 1,
                            "range": [0.0, 1.0],
                        }
                    ],
                },
            }
        ],
        "time": {
            "steps": 4,
            "index": 1,
            "time": 0.5,
            "follow_latest": False,
            "newer_steps": 2,
            "latest_time": 1.5,
        },
        "view": {"mode": "remote", "geometry_mb": 30.0, "threshold_mb": 20.0},
    }
    text = present.scene_text(summary, "on node08")
    assert "Scene wave-1 on node08" in text
    assert "step 2 of 4" in text and "2 newer step(s)" in text
    assert "remote rendering" in text and "30.00 MB" in text
    assert "scene='wave-1'" in text


def test_measure_text() -> None:
    text = present.measure_text(
        {
            "field": "p",
            "association": "POINTS",
            "component": "magnitude",
            "object": "o1",
            "label": "wave",
            "count": 3,
            "finite": 3,
            "min": 0.0,
            "max": 2.0,
            "mean": 1.0,
            "median": 1.0,
            "std": 0.8,
            "histogram": {"edges": [0.0, 1.0, 2.0], "counts": [1, 2]},
        }
    )
    assert "min 0, max 2, mean 1" in text and "[1, 2): 2" in text


def test_scene_ids_are_stable_per_path() -> None:
    from paraview_mcp.node import manifest

    a = manifest.scene_id_for("/data/run/wave.vti")
    assert a == manifest.scene_id_for("/data/run/wave.vti")
    assert a != manifest.scene_id_for("/data/other/wave.vti")
    assert a.startswith("wave-")


def test_manifest_round_trip(tmp_path: Path) -> None:
    from paraview_mcp.node import manifest

    scene = manifest.new_scene("s-1", {"path": "/x", "growing": False})
    scene["objects"] = [
        {"id": "o1", "kind": "reader"},
        {"id": "o2", "kind": "filter", "input": "o1", "label": "iso"},
        {"id": "o3", "kind": "filter", "input": "o2"},
        {"id": "o4", "kind": "filter", "input": "o1"},
    ]
    path = manifest.write(tmp_path, scene)
    assert json.loads(path.read_text())["updated"]
    assert manifest.read(tmp_path, "s-1")["objects"] == scene["objects"]
    assert manifest.known_scenes(tmp_path) == ["s-1"]
    assert manifest.descendants(scene, "o2") == ["o2", "o3"]
    assert manifest.find_object(scene, "iso")["id"] == "o2"
    with pytest.raises(ValueError):
        manifest.scene_dir(tmp_path, "../escape")


def test_reported_version_matches_the_node_folder_version() -> None:
    from paraview_mcp import server

    assert server.mcp.version == config.SERVER_VERSION
