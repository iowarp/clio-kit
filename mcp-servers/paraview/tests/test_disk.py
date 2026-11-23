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


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_write_index_repoints_a_folder_an_earlier_run_left(tmp_path: Path) -> None:
    """The live 2a run: a scene's follow folder still linked into the last run's
    deleted output, and pvserver segfaulted reading it. Every link now points at
    this run's output, and what this run has not got goes."""
    _, old = _run(tmp_path / "old")
    _, bp = _run(tmp_path)
    for name in ("data.0", "md.0", "mmd.0"):
        (bp / name).write_bytes(b"x")
    folder = tmp_path / "follow" / "data.bp5"
    folder.mkdir(parents=True)
    for name in ("data.0", "md.0", "data.1"):
        os.symlink(old / name, folder / name)
    (folder / "mmd.0").write_bytes(b"stale")
    disk.write_index(str(bp), str(folder), 0)
    assert sorted(os.listdir(folder)) == ["data.0", "md.0", "md.idx", "mmd.0"]
    for name in ("data.0", "md.0", "mmd.0"):
        assert os.readlink(folder / name) == str(bp / name)


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
    run, bp = _run(tmp_path)
    (bp / "md.0").write_bytes(b"m" * 5)

    stale.add(str(bp))
    assert disk.files(str(bp)) == str(bp)
    assert listed == [str(run)]

    stale.update({str(bp), str(bp / "md.idx")})
    listed.clear()
    reason, _, _ = disk.source_readiness(str(bp), "ADIOS2VTXReader", False)
    assert reason == "its index header is not written whole yet"
    assert listed[:2] == [str(run), str(bp)]

    stale.add(str(bp / "md.0"))
    listed.clear()
    assert disk._size(str(bp / "md.0")) == 5
    assert listed == [str(bp)]
    assert disk._size(str(bp / "gone")) == -1


def test_a_growing_bp5_opens_once_one_step_is_surely_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A solver writes its next step most of the time: open_view must not wait for
    the folder to stand still, only for a step it can show whole."""
    _, bp = _run(tmp_path)
    for steps, clear, opens in [(3, False, True), (1, False, False), (1, True, True)]:
        state = disk.Readiness(None, clear, steps)
        monkeypatch.setattr(disk, "check", lambda path, state=state: state)
        reason, can_open, _ = disk.source_readiness(str(bp), "ADIOS2VTXReader", False)
        assert reason is None and can_open is opens
        assert disk.whole_steps(state) == (steps if clear else steps - 1)


def _bp_header(folder: Path, version: int, active: int) -> None:
    header = bytearray(disk.HEADER)
    header[disk.BP_VERSION_AT] = version
    header[disk.ACTIVE_AT[version]] = active
    (folder / "md.idx").write_bytes(bytes(header))


def test_a_writer_that_closes_its_bp_output_is_seen(tmp_path: Path) -> None:
    """A run that ends closes its ADIOS2 output and the index header says so:
    watch_steps then ends as finished rather than waiting for a stall."""
    bp = tmp_path / "data.bp5"
    bp.mkdir()
    assert disk.writer_active(str(bp)) is None
    (bp / "md.idx").write_bytes(b"short")
    assert disk.writer_active(str(bp)) is None
    for version in (4, 5):
        _bp_header(bp, version, 1)
        assert disk.writer_active(str(bp)) is True
        assert disk.writing(disk.disk_key(str(bp))) is True
        _bp_header(bp, version, 0)
        assert disk.writer_active(str(bp)) is False
        key = disk.disk_key(str(bp))
        assert disk.writing(key) is False
        # The flag is part of what tells the source changed.
        _bp_header(bp, version, 1)
        assert disk.disk_key(str(bp)) != key
    plain = tmp_path / "plain"
    plain.mkdir()
    assert disk.writing(disk.disk_key(str(plain))) is None


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
    _, bp = _run(tmp_path)
    fake = _FakeServer(monkeypatch)
    there = datafs.ServerDisk()
    monkeypatch.delitem(sys.modules, there._name, raising=False)
    try:
        first = there.disk_key(str(bp))
        assert there.disk_key(str(bp)) == first
        assert there.check(str(bp)) == datafs.LocalDisk().check(str(bp))
        assert fake.reducers == 1 and fake.movers == 1 and len(fake.scripts) == 3
        assert "SOURCE = None" not in fake.scripts[0]
        assert all("SOURCE = None" in s for s in fake.scripts[1:])
        assert all(len(s) < 3000 for s in fake.scripts[1:])

        del sys.modules[there._name]  # a server that no longer has it
        assert there.disk_key(str(bp)) == first
        assert "SOURCE = None" in fake.scripts[3]
        assert "SOURCE = None" not in fake.scripts[4]
    finally:
        sys.modules.pop(there._name, None)


def test_a_server_disk_that_stops_answering_does_not_blame_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live run: a pvserver whose linger ends answers nothing. It answered
    before, so it has Python; say it stopped, not that it needs Python."""
    _, bp = _run(tmp_path)
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
            fresh.disk_key(str(bp))

        there.disk_key(str(bp))
        assert there._mover is not None
        there._mover.GetClientSideObject = lambda: types.SimpleNamespace(
            Modified=lambda: None, GetOutputDataObject=lambda p: None
        )
        there._mover.UpdatePipeline = lambda: None
        with pytest.raises(CommandError, match="stopped answering") as gone:
            there.disk_key(str(bp))
        assert "Python" not in str(gone.value)
    finally:
        sys.modules.pop(there._name, None)
