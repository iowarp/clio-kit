"""Where the data is: this machine, or the pvserver the node half joined.

A built-in session reads files on this machine. A joined pvserver reads them
on its own machine, so following a growing source must look at that disk:
``ServerDisk`` runs ``disk``'s functions in the server's Python, on rank 0, and
brings back their result. It does so with a programmable source kept out of
the pipeline, so a server needs ParaView's Python and nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import disk, manifest
from .disk import Readiness
from .errors import CommandError

# Runs inside the server. SOURCE is disk.py; the module is registered so its
# dataclass resolves its own annotations.
_SERVER_SCRIPT = """
import json, sys, types
from vtkmodules.vtkCommonCore import vtkStringArray
try:
    from vtkmodules.vtkParallelCore import vtkMultiProcessController
    _c = vtkMultiProcessController.GetGlobalController()
    _rank = _c.GetLocalProcessId() if _c is not None else 0
except ImportError:
    _rank = 0
_out = self.GetOutputDataObject(0)
if _rank == 0:
    _mod = sys.modules.get(NAME)
    if _mod is None:
        _mod = types.ModuleType(NAME)
        sys.modules[NAME] = _mod
        try:
            exec(compile(SOURCE, NAME, "exec"), _mod.__dict__)
        except BaseException:
            del sys.modules[NAME]
            raise
    try:
        _text = json.dumps({"ok": _mod.dispatch(OP, ARGS)})
    except Exception as _exc:
        _text = json.dumps({"error": "%s: %s" % (type(_exc).__name__, _exc)})
    _column = vtkStringArray()
    _column.SetName("r")
    _column.InsertNextValue(_text)
    _out.AddColumn(_column)
"""


class LocalDisk:
    """The data is on this machine."""

    remote = False

    def call(self, op: str, **args: Any) -> Any:
        return disk.dispatch(op, args)

    def expand(self, path: str) -> str:
        return str(self.call("expand", path=path))

    def disk_key(self, path: str) -> Any:
        return self.call("disk_key", path=path)

    def files(self, path: str) -> list[str] | str | None:
        result: list[str] | str | None = self.call("files", path=path)
        return result

    def is_bp(self, path: str, reader: str | None = None) -> bool:
        return bool(self.call("is_bp", path=path, reader=reader))

    def check(self, path: str) -> Readiness:
        return Readiness(**self.call("check", path=path))

    def write_index(self, path: str, folder: str, steps: int) -> str:
        return str(self.call("write_index", path=path, folder=folder, steps=steps))

    def source_readiness(
        self, path: str, reader: str | None, pattern: bool
    ) -> tuple[str | None, bool, Any]:
        reason, clear, key = self.call(
            "source_readiness", path=path, reader=reader, pattern=pattern
        )
        return reason, bool(clear), key

    def follow_folder(self, root: Path, scene_id: str, item_id: str, name: str) -> str:
        """Where the copy of a growing BP5 folder goes, on the data's machine."""
        return str(manifest.scene_dir(root, scene_id) / "follow" / item_id / name)


class ServerDisk(LocalDisk):
    """The data is on the joined pvserver's machine."""

    remote = True

    def __init__(self) -> None:
        source = Path(disk.__file__).read_text("utf-8")
        self._source = source
        self._name = "clio_paraview_disk_" + str(abs(hash(source)) % 10**12)
        self._proxy: Any = None
        self._calls = 0

    def call(self, op: str, **args: Any) -> Any:
        from paraview import servermanager

        if self._proxy is None:
            self._proxy = servermanager.sources.ProgrammableSource(
                OutputDataSetType="vtkTable"
            )
        self._calls += 1
        # The call number makes every script new, so the source always runs again.
        self._proxy.Script = (
            f"# call {self._calls}\n"
            f"NAME = {self._name!r}\n"
            f"SOURCE = {self._source!r}\n"
            f"OP = {op!r}\n"
            f"ARGS = json.loads({json.dumps(args)!r})\n" + _SERVER_SCRIPT
        )
        self._proxy.UpdatePipeline()
        table = servermanager.Fetch(self._proxy, 0)
        column = table.GetColumnByName("r") if table is not None else None
        if column is None or column.GetNumberOfValues() < 1:
            raise CommandError(
                "server_disk",
                "The ParaView server did not answer a look at its disk. Its "
                "programmable sources need ParaView's Python on the server.",
            )
        reply = json.loads(column.GetValue(0))
        if "error" in reply:
            raise CommandError(
                "server_disk",
                f"The ParaView server could not look at its disk: {reply['error']}",
            )
        return reply["ok"]

    def follow_folder(self, root: Path, scene_id: str, item_id: str, name: str) -> str:
        # The server's home, which every rank of a cluster's pvserver shares.
        return f"~/.clio/paraview/follow/{scene_id}/{item_id}/{name}"


def for_session(session: str | None) -> LocalDisk:
    return ServerDisk() if session else LocalDisk()
