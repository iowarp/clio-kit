"""Looking at a source on a shared file system: every disk operation first lists
the folders on the way to its path, so a path another machine made is seen at
once. No ParaView needed."""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from paraview_mcp.node import datafs, disk


@pytest.fixture
def listed(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every folder ``os.listdir`` is asked for, in order."""
    seen: list[str] = []
    real = os.listdir

    def record(path: Any = ".") -> list[str]:
        seen.append(os.path.abspath(path))
        return real(path)

    monkeypatch.setattr(disk.os, "listdir", record)
    return seen


def _run(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "shared" / "run"
    bp = run / "data.bp5"
    bp.mkdir(parents=True)
    (bp / "md.idx").write_bytes(b"")
    return run, bp


def test_refresh_lists_from_the_top_down(tmp_path: Path, listed: list[str]) -> None:
    run, bp = _run(tmp_path)
    assert disk.refresh(str(bp)) == [str(run.parent), str(run), str(bp)]
    assert listed == [str(run.parent), str(run), str(bp)]


def test_refresh_of_a_file_or_a_missing_path_lists_its_folders(
    tmp_path: Path, listed: list[str]
) -> None:
    run, bp = _run(tmp_path)
    assert disk.refresh(str(bp / "md.idx")) == [str(run), str(bp)]
    missing = run / "rerun" / "data.bp5"
    assert disk.refresh(str(missing)) == [str(run)]


def test_refresh_of_a_pattern_lists_the_folder_before_it(
    tmp_path: Path, listed: list[str]
) -> None:
    run, _ = _run(tmp_path)
    (run / "data").mkdir()
    pattern = str(run / "data" / "snapshot-*.xdmf")
    assert disk.refresh(pattern) == [str(run.parent), str(run), str(run / "data")]


@pytest.mark.parametrize(
    ("op", "args"),
    [
        ("disk_key", {}),
        ("files", {}),
        ("is_bp", {"reader": None}),
        ("check", {}),
        ("source_readiness", {"reader": "ADIOS2VTXReader", "pattern": False}),
    ],
)
def test_every_operation_lists_before_it_looks(
    tmp_path: Path, listed: list[str], op: str, args: dict[str, Any]
) -> None:
    run, bp = _run(tmp_path)
    disk.dispatch(op, {"path": str(bp), **args})
    assert listed[:3] == [str(run.parent), str(run), str(bp)]


def test_write_index_lists_the_source_first(tmp_path: Path, listed: list[str]) -> None:
    run, bp = _run(tmp_path)
    folder = tmp_path / "follow" / "data.bp5"
    disk.dispatch("write_index", {"path": str(bp), "folder": str(folder), "steps": 0})
    assert listed[:3] == [str(run.parent), str(run), str(bp)]


def test_expand_does_not_touch_the_disk(tmp_path: Path, listed: list[str]) -> None:
    disk.dispatch("expand", {"path": str(tmp_path / "x")})
    assert listed == []


def test_a_remade_output_is_seen_again(tmp_path: Path) -> None:
    """A rerun removes its output and writes it again; each look finds what is
    there now."""
    _, bp = _run(tmp_path)
    before = disk.dispatch("disk_key", {"path": str(bp)})
    for name in os.listdir(bp):
        os.remove(bp / name)
    bp.rmdir()
    assert disk.dispatch("files", {"path": str(bp)}) is None
    bp.mkdir()
    (bp / "md.idx").write_bytes(b"x" * 64)
    assert disk.dispatch("files", {"path": str(bp)}) == str(bp)
    assert disk.dispatch("disk_key", {"path": str(bp)}) != before


def test_sizes_and_times_are_read_through_an_open_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``os.stat`` on a shared file system can answer with what this machine saw
    minutes ago; a growing source would look unchanged. Every size and time is
    read through an open file instead."""
    _, bp = _run(tmp_path)
    (bp / "md.0").write_bytes(b"m" * 10)
    real = os.stat

    def stale(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        if os.path.basename(str(path)) in ("md.0", "md.idx"):
            return os.stat_result((0o100644, 0, 0, 1, 0, 0, 999, 0, 0, 0))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stale)
    key = dict(disk.dispatch("disk_key", {"path": str(bp)}))
    assert key["md.0"][0] == 10 and key["md.idx"][0] == 0
    assert disk._size(str(bp / "md.0")) == 10
    assert disk._size(str(bp / "missing")) == -1
    assert disk._stat(str(bp / "missing")) is None


def test_a_file_that_cannot_be_opened_falls_back_to_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bp = _run(tmp_path)
    (bp / "md.0").write_bytes(b"m" * 7)

    def denied(*args: Any, **kwargs: Any) -> int:
        raise PermissionError("denied")

    monkeypatch.setattr(disk.os, "open", denied)
    assert disk._size(str(bp / "md.0")) == 7
    assert disk._size(str(bp / "missing")) == -1


def _server_run(
    monkeypatch: pytest.MonkeyPatch, rank: int, op: str, args: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Run ``ServerDisk``'s script as one pvserver rank would; returns the folders
    that rank listed and the replies it wrote."""
    replies: list[str] = []

    class Column:
        def SetName(self, name: str) -> None:
            pass

        def InsertNextValue(self, value: str) -> None:
            replies.append(value)

    class Controller:
        def GetLocalProcessId(self) -> int:
            return rank

    core = types.ModuleType("vtkmodules.vtkCommonCore")
    core.vtkStringArray = Column  # type: ignore[attr-defined]
    parallel = types.ModuleType("vtkmodules.vtkParallelCore")
    parallel.vtkMultiProcessController = types.SimpleNamespace(  # type: ignore[attr-defined]
        GetGlobalController=lambda: Controller()
    )
    monkeypatch.setitem(sys.modules, "vtkmodules", types.ModuleType("vtkmodules"))
    monkeypatch.setitem(sys.modules, "vtkmodules.vtkCommonCore", core)
    monkeypatch.setitem(sys.modules, "vtkmodules.vtkParallelCore", parallel)
    name = f"clio_paraview_disk_test_{rank}"
    monkeypatch.delitem(sys.modules, name, raising=False)

    seen: list[str] = []
    real = os.listdir

    def record(path: Any = ".") -> list[str]:
        seen.append(os.path.abspath(path))
        return real(path)

    monkeypatch.setattr(os, "listdir", record)
    out = types.SimpleNamespace(AddColumn=lambda column: None)
    owner = types.SimpleNamespace(GetOutputDataObject=lambda port: out)
    scope = {
        "self": owner,
        "json": json,
        "NAME": name,
        "SOURCE": Path(disk.__file__).read_text("utf-8"),
        "OP": op,
        "ARGS": args,
    }
    try:
        exec(datafs._SERVER_SCRIPT, scope)  # noqa: S102 - the script the server runs
    finally:
        sys.modules.pop(name, None)
    return seen, replies


def test_every_pvserver_rank_lists_the_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, bp = _run(tmp_path)
    args = {"path": str(bp)}
    seen, replies = _server_run(monkeypatch, 1, "check", args)
    assert seen == [str(run.parent), str(run), str(bp)] and replies == []
    seen, replies = _server_run(monkeypatch, 0, "check", args)
    assert seen[:3] == [str(run.parent), str(run), str(bp)]
    assert "ok" in json.loads(replies[0])
