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
from paraview_mcp.node.errors import CommandError


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
    out = run / "output"
    out.mkdir(parents=True)
    (out / "part.0").write_bytes(b"")
    return run, out


def test_refresh_lists_from_the_top_down(tmp_path: Path, listed: list[str]) -> None:
    run, out = _run(tmp_path)
    assert disk.refresh(str(out)) == [str(run.parent), str(run), str(out)]
    assert listed == [str(run.parent), str(run), str(out)]


def test_refresh_of_a_file_or_a_missing_path_lists_its_folders(
    tmp_path: Path, listed: list[str]
) -> None:
    run, out = _run(tmp_path)
    assert disk.refresh(str(out / "part.0")) == [str(run), str(out)]
    missing = run / "rerun" / "output"
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
        ("series", {}),
        ("source_readiness", {"pattern": False}),
    ],
)
def test_every_operation_lists_before_it_looks(
    tmp_path: Path, listed: list[str], op: str, args: dict[str, Any]
) -> None:
    run, out = _run(tmp_path)
    disk.dispatch(op, {"path": str(out), **args})
    assert listed[:3] == [str(run.parent), str(run), str(out)]


def test_expand_does_not_touch_the_disk(tmp_path: Path, listed: list[str]) -> None:
    disk.dispatch("expand", {"path": str(tmp_path / "x")})
    assert listed == []


def test_a_remade_output_is_seen_again(tmp_path: Path) -> None:
    """A rerun removes its output and writes it again; each look finds what is
    there now."""
    _, out = _run(tmp_path)
    before = disk.dispatch("disk_key", {"path": str(out)})
    for name in os.listdir(out):
        os.remove(out / name)
    out.rmdir()
    assert disk.dispatch("files", {"path": str(out)}) is None
    out.mkdir()
    (out / "part.0").write_bytes(b"x" * 64)
    assert disk.dispatch("files", {"path": str(out)}) == str(out)
    assert disk.dispatch("disk_key", {"path": str(out)}) != before


@pytest.fixture
def cached(monkeypatch: pytest.MonkeyPatch) -> tuple[set[str], list[str]]:
    """A shared file system that cached some paths as missing: each is "not there"
    until the folder holding it is listed. Returns those paths and the folders
    listed."""
    stale: set[str] = set()
    listed: list[str] = []
    real_list, real_open = os.listdir, os.open
    real_exists, real_isdir = os.path.exists, os.path.isdir

    def missing(path: Any) -> bool:
        return os.path.abspath(str(path)) in stale

    def listdir(path: Any = ".") -> list[str]:
        folder = os.path.abspath(path)
        listed.append(folder)
        stale.difference_update({p for p in stale if os.path.dirname(p) == folder})
        return real_list(path)

    def open_(path: Any, *args: Any, **kwargs: Any) -> int:
        if missing(path):
            raise FileNotFoundError(2, "No such file or directory (cached)", str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "listdir", listdir)
    monkeypatch.setattr(os, "open", open_)
    monkeypatch.setattr(os.path, "exists", lambda p: not missing(p) and real_exists(p))
    monkeypatch.setattr(os.path, "isdir", lambda p: not missing(p) and real_isdir(p))
    return stale, listed


def test_a_cached_miss_is_looked_up_again_after_listing_once(
    tmp_path: Path, cached: tuple[set[str], list[str]]
) -> None:
    """Asked before the solver made it, a path stays "not there" on this machine;
    the first miss lists the folder holding it and looks again."""
    stale, listed = cached
    run, out = _run(tmp_path)
    (out / "part.0").write_bytes(b"m" * 5)

    stale.add(str(out))
    assert disk.files(str(out)) == str(out)
    assert listed == [str(run)]

    stale.add(str(out))
    listed.clear()
    reason, _ = disk.source_readiness(str(out), False)
    assert reason is None and str(run) in listed

    stale.add(str(out / "part.0"))
    listed.clear()
    stat = disk._stat(str(out / "part.0"))
    assert stat is not None and stat[0] == 5
    assert listed == [str(out)]
    assert disk._stat(str(out / "gone")) is None


def test_sizes_and_times_are_read_through_an_open_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``os.stat`` on a shared file system can answer with what this machine saw
    minutes ago; a growing source would look unchanged. Every size and time is
    read through an open file instead."""
    _, out = _run(tmp_path)
    (out / "part.0").write_bytes(b"m" * 10)
    (out / "part.1").write_bytes(b"")
    real = os.stat

    def stale(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        if os.path.basename(str(path)) in ("part.0", "part.1"):
            return os.stat_result((0o100644, 0, 0, 1, 0, 0, 999, 0, 0, 0))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stale)
    key = dict(disk.dispatch("disk_key", {"path": str(out)}))
    assert key["part.0"][0] == 10 and key["part.1"][0] == 0
    stat = disk._stat(str(out / "part.0"))
    assert stat is not None and stat[0] == 10
    assert disk._stat(str(out / "missing")) is None


def test_a_file_that_cannot_be_opened_falls_back_to_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, out = _run(tmp_path)
    (out / "part.0").write_bytes(b"m" * 7)

    def denied(*args: Any, **kwargs: Any) -> int:
        raise PermissionError("denied")

    monkeypatch.setattr(disk.os, "open", denied)
    stat = disk._stat(str(out / "part.0"))
    assert stat is not None and stat[0] == 7
    assert disk._stat(str(out / "missing")) is None


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
    run, out = _run(tmp_path)
    args = {"path": str(out)}
    seen, replies = _server_run(monkeypatch, 1, "disk_key", args)
    assert seen == [str(run.parent), str(run), str(out)] and replies == []
    seen, replies = _server_run(monkeypatch, 0, "disk_key", args)
    assert seen[:3] == [str(run.parent), str(run), str(out)]
    assert "ok" in json.loads(replies[0])


class _FakeServer:
    """ParaView's servermanager as ``ServerDisk`` uses it: a programmable source
    whose script runs, as rank 0 would run it, when the reduction filter updates."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        server = self

        class Column:
            def __init__(self) -> None:
                self.values: list[str] = []

            def SetName(self, name: str) -> None:
                pass

            def InsertNextValue(self, value: str) -> None:
                self.values.append(value)

            def GetNumberOfValues(self) -> int:
                return len(self.values)

            def GetValue(self, index: int) -> str:
                return self.values[index]

        class Table:
            def __init__(self) -> None:
                self.columns: list[Column] = []

            def AddColumn(self, column: Column) -> None:
                self.columns.append(column)

            def GetColumnByName(self, name: str) -> Column | None:
                return self.columns[0] if self.columns else None

        class Source:
            def __init__(self, **_: Any) -> None:
                self.Script = ""

        class Reducer:
            def __init__(self, Input: Source) -> None:
                self.input = Input
                self.table = Table()
                self.ran: str | None = None
                self.PassThrough: int | None = None
                server.reducers += 1

            def UpdatePipeline(self) -> None:
                if self.input.Script == self.ran:
                    return  # nothing changed upstream, as in a pipeline
                self.ran = self.input.Script
                self.table = Table()
                owner = types.SimpleNamespace(GetOutputDataObject=lambda p: self.table)
                server.scripts.append(self.input.Script)
                exec(self.input.Script, {"self": owner})  # noqa: S102

            def GetDataInformation(self, port: int) -> Any:
                return types.SimpleNamespace(
                    GetDataSetType=lambda: 19, GetExtent=lambda: (0, -1, 0, -1, 0, -1)
                )

        class Mover:
            def __init__(self, Input: Reducer) -> None:
                self.input = Input
                self.OutputDataType: int | None = None
                self.WholeExtent: Any = None
                server.movers += 1

            def UpdatePipeline(self) -> None:
                self.input.UpdatePipeline()

            def GetClientSideObject(self) -> Any:
                return types.SimpleNamespace(
                    GetOutputDataObject=lambda p: self.input.table,
                    Modified=lambda: None,
                )

        core = types.ModuleType("vtkmodules.vtkCommonCore")
        core.vtkStringArray = Column  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "vtkmodules", types.ModuleType("vtkmodules"))
        monkeypatch.setitem(sys.modules, "vtkmodules.vtkCommonCore", core)
        monkeypatch.setitem(sys.modules, "vtkmodules.vtkParallelCore", None)
        manager = types.ModuleType("paraview.servermanager")
        manager.sources = types.SimpleNamespace(ProgrammableSource=Source)  # type: ignore[attr-defined]
        manager.filters = types.SimpleNamespace(  # type: ignore[attr-defined]
            ReductionFilter=Reducer, ClientServerMoveData=Mover
        )
        package = types.ModuleType("paraview")
        package.servermanager = manager  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "paraview", package)
        monkeypatch.setitem(sys.modules, "paraview.servermanager", manager)
        self.reducers = 0
        self.movers = 0
        self.scripts: list[str] = []


def test_the_server_disk_sends_disk_py_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow link pays for every byte and proxy: disk.py goes over on the first
    call only, one reduction filter brings back every answer, and a server that
    lost the module (or never had it) asks for it again."""
    _, out = _run(tmp_path)
    fake = _FakeServer(monkeypatch)
    there = datafs.ServerDisk()
    monkeypatch.delitem(sys.modules, there._name, raising=False)
    try:
        first = there.disk_key(str(out))
        assert there.disk_key(str(out)) == first
        assert there.files(str(out)) == datafs.LocalDisk().files(str(out)) == str(out)
        assert fake.reducers == 1 and fake.movers == 1 and len(fake.scripts) == 3
        assert "SOURCE = None" not in fake.scripts[0]
        assert all("SOURCE = None" in s for s in fake.scripts[1:])
        assert all(len(s) < 3000 for s in fake.scripts[1:])

        del sys.modules[there._name]  # a server that no longer has it
        assert there.disk_key(str(out)) == first
        assert "SOURCE = None" in fake.scripts[3]
        assert "SOURCE = None" not in fake.scripts[4]
    finally:
        sys.modules.pop(there._name, None)


def test_a_server_disk_that_stops_answering_does_not_blame_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live run: a pvserver whose linger ends answers nothing. It answered
    before, so it has Python; say it stopped, not that it needs Python."""
    _, out = _run(tmp_path)
    _FakeServer(monkeypatch)
    there = datafs.ServerDisk()
    monkeypatch.delitem(sys.modules, there._name, raising=False)
    try:
        fresh = datafs.ServerDisk()
        fresh._mover = types.SimpleNamespace(
            UpdatePipeline=lambda: None,
            GetClientSideObject=lambda: types.SimpleNamespace(
                Modified=lambda: None, GetOutputDataObject=lambda p: None
            ),
        )
        fresh._proxy = types.SimpleNamespace(Script="")
        with pytest.raises(CommandError, match="need ParaView's Python"):
            fresh.disk_key(str(out))

        there.disk_key(str(out))
        assert there._mover is not None
        there._mover.GetClientSideObject = lambda: types.SimpleNamespace(
            Modified=lambda: None, GetOutputDataObject=lambda p: None
        )
        there._mover.UpdatePipeline = lambda: None
        with pytest.raises(CommandError, match="stopped answering") as gone:
            there.disk_key(str(out))
        assert "Python" not in str(gone.value)
    finally:
        sys.modules.pop(there._name, None)
