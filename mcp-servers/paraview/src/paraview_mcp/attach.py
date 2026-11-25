"""Attaching the front to a node half: prepare the folder, start or find it.

The node half lives in ``~/.clio/paraview/<server version>/`` on the machine
where ParaView runs:

- ``site/``: trame, trame-vtk and trame-vuetify, built for that machine's pvpython;
- ``code/``: the node half's code, copied over the same connection;
- ``node.json``: the running node half's pid, port and token (mode 0600);
  ``node-<namespace>.json`` when a server in another network namespace already
  runs one here;
- ``scenes/<id>/manifest.json``: each scene, written after every commit.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shlex
import tarfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import client, config, native
from .config import Settings
from .errors import PREFER_FORWARD, ParaViewError, no_paraview
from .link import LinkError, NodeClient
from .transport import Completed, Transport, Watch, transport_for

#: SSH failures that a forwarded pvserver endpoint would avoid.
_SSH_CODES = frozenset({"ssh_failed", "unreachable", "forward_failed"})

#: Installer output that means the folder refused the write, not the index.
_WRITE_DENIED = (
    "Access is denied",
    "os error 5",
    "Permission denied",
    "PermissionError",
)

NODE_SOURCE = Path(__file__).resolve().parent / "node"
RUN_NODE = """import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paraview_node.main import main
main()
"""


_JOIN_FAILED = "pvserver join failed: "

#: A live view reopens its ParaView at most this often within the window (seconds),
#: so a scene that crashes its pvserver each time it is drawn is not drawn forever.
REOPEN_LIMIT = 3
REOPEN_WINDOW = 600.0


def _whole_lines(text: str, limit: int = 1500) -> str:
    """The end of ``text`` within ``limit`` characters, cut only between lines."""
    kept: list[str] = []
    size = 0
    for line in reversed(text.strip().splitlines()):
        if kept and size + len(line) + 1 > limit:
            break
        kept.append(line if len(line) <= limit else "…" + line[-limit:])
        size += len(line) + 1
    return "\n".join(reversed(kept))


def start_failure(where: str, log: str) -> ParaViewError:
    """The typed error for a node half whose log shows it stopped while starting."""
    for line in log.splitlines():
        if _JOIN_FAILED in line:
            target, _, reason = line.split(_JOIN_FAILED, 1)[1].partition(": ")
            return ParaViewError(
                "pvserver_join_failed",
                f"The ParaView node half started {where} but could not join the "
                f"pvserver at {target}: {reason or 'no reason given'}. Check that "
                "the pvserver is still running and that its forwarded port is open.",
            )
    return ParaViewError(
        "node_failed",
        f"The ParaView node half failed to start {where}:\n{_whole_lines(log)}",
    )


_JOINING = "joining pvserver at "


@dataclass
class StartLook:
    """One look at a node half that is starting."""

    record: dict[str, Any] | None
    #: The log's tail when the log shows that the start failed.
    failed: str | None
    alive: bool
    size: int
    last: str
    tail: str = ""


def wait_for_start(
    look: Callable[[], StartLook],
    code_hash: str,
    where: str,
    log_path: str,
    stall: float,
    cancel: threading.Event,
    progress: Callable[[str], None] | None = None,
    stop: Callable[[], None] | None = None,
    every: float = 1.0,
) -> dict[str, Any]:
    """Wait for a starting node half for as long as it shows progress.

    There is no overall limit. The start fails at once when the process ends or
    its log shows a failure, and only after ``stall`` seconds with no new log
    output; ``cancel`` ends the wait whenever it is set. Each new log line is
    passed to ``progress``, so the wait is visible.
    """
    began = changed = told = time.monotonic()
    size = -1
    said = ""
    dead = 0
    while True:
        if cancel.is_set():
            if stop is not None:
                stop()
            raise ParaViewError(
                "cancelled", f"Stopped starting the ParaView node half {where}."
            )
        seen = look()
        if seen.record is not None and seen.record.get("code_hash") == code_hash:
            return seen.record
        if seen.failed:
            raise start_failure(where, seen.failed)
        now = time.monotonic()
        if seen.size != size:
            size, changed = seen.size, now
        # A launcher may hand over to a child; take the process for gone only
        # when it has also stopped writing.
        dead = 0 if seen.alive else dead + 1
        if dead >= 3 and now - changed >= 2 * every:
            raise ParaViewError(
                "node_failed",
                f"The ParaView node half stopped while starting {where}. Its log "
                f"({log_path}) ends with:\n{_whole_lines(seen.tail or seen.last)}",
            )
        quiet = now - changed
        if progress is not None:
            line = seen.last.strip()
            if line != said or now - told >= 5.0:
                said, told = line, now
                progress(
                    f"Starting ParaView {where}, {now - began:.0f} s"
                    + (f": {line}" if line else "")
                )
        if quiet >= stall:
            if stop is not None:
                stop()
            last = seen.last.strip()
            if last.startswith(_JOINING):
                target = last[len(_JOINING) :]
                raise ParaViewError(
                    "pvserver_join_stalled",
                    f"The ParaView node half {where} waited {quiet:.0f} s to join the "
                    f"pvserver at {target} with no answer. A pvserver serves one "
                    "client at a time, so it may be serving another one, or its "
                    "forwarded port leads nowhere. Check the pvserver and its "
                    "forward, then open the view again.",
                )
            raise ParaViewError(
                "node_start_stalled",
                f"The ParaView node half {where} showed no progress for {quiet:.0f} "
                f"s, so it was stopped. Its log ({log_path}) ends with:\n"
                f"{_whole_lines(seen.tail or last) or '(nothing)'}",
            )
        cancel.wait(every)


def gone_error(
    note: str, tail: str, where: str, scene: str | None = None
) -> ParaViewError:
    """The error for a linked node half that has gone, from what it left behind.

    ``note`` is the ``<slot>.lost.json`` it writes when its pvserver closes the
    connection; ``tail`` is the end of its log, for any other end.
    """
    try:
        lost = json.loads(note) if note.startswith("{") else None
    except ValueError:
        lost = None
    if isinstance(lost, dict) and not scene and isinstance(lost.get("scene"), str):
        scene = lost["scene"]
    again = (
        f" The scene {scene} is saved: its live view opens it again on its own once "
        f"a pvserver runs, and open_view(scene={scene!r}) does too."
        if scene
        else " open_view starts a new node half."
    )
    if isinstance(lost, dict):
        return ParaViewError(
            "pvserver_lost",
            f"The ParaView view {where} has stopped: {lost.get('reason')}. "
            "Nothing more can be drawn or read through it until then." + again,
        )
    return ParaViewError(
        "node_lost",
        f"The ParaView node half {where} has stopped, so its live view shows "
        "nothing more."
        + (f" Its log ends with:\n{_whole_lines(tail)}\n" if tail else "")
        + again,
    )


def code_bundle() -> tuple[bytes, str]:
    """The node half's code as a gzip tarball, and its content hash."""
    files = sorted(p for p in NODE_SOURCE.glob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update(RUN_NODE.encode())
    code_hash = digest.hexdigest()[:16]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:

        def add(name: str, data: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(data))

        for path in files:
            add(f"paraview_node/{path.name}", path.read_bytes())
        add("run_node.py", RUN_NODE.encode())
        add("HASH", code_hash.encode())
    return buffer.getvalue(), code_hash


def parse_markers(text: str) -> dict[str, str]:
    """Read ``CLIO_<KEY>=value`` lines a probe script printed."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("CLIO_") and "=" in line:
            key, _, value = line.partition("=")
            out[key[5:]] = value.strip()
    return out


def local_folder(version: str) -> str:
    """The node half's folder on this machine: clio-kit's own data folder.

    On another machine it is ``~/.clio/paraview/<version>/``. Here it sits under
    ``CLIO_KIT_CACHE_DIR`` when the host sets it, as the ParaView client does,
    else in clio-kit's data folder. A host that confines the servers it launches
    grants and keeps that cache folder; inside a Windows AppContainer the data
    folder resolves to a per-launch profile the fence does not let them write.
    """
    configured = os.environ.get("CLIO_KIT_CACHE_DIR", "").strip()
    if configured:
        return str(Path(configured).expanduser() / "paraview-node" / version)
    import platformdirs

    base = platformdirs.user_data_dir("clio-kit", appauthor=False)
    return str(Path(base) / "paraview" / version)


def node_base(pvserver: str | None) -> str:
    """The record name of the node half for ``pvserver``.

    A node half joins one pvserver for its life, so each pvserver has its own
    record, lock and log: a view of one never stops the node half of another.
    """
    if not pvserver:
        return "node"
    return "node-pv" + hashlib.sha1(pvserver.encode()).hexdigest()[:8]


def probe_script(version: str, folder: str | None = None, base: str = "node") -> str:
    where = shlex.quote(folder) if folder else f'"$HOME/.clio/paraview/{version}"'
    return f"""
D={where}
mkdir -p "$D" && chmod 700 "$D"
# A new server version prepares a new folder; its scenes come from the newest older one.
if [ ! -d "$D/scenes" ]; then
  for old in $(ls -d "$D"/../*/scenes 2>/dev/null | sort -V -r); do
    cp -R "$old" "$D/scenes" && break
  done
fi
echo "CLIO_DIR=$D"
echo "CLIO_PVPYTHON=$(command -v pvpython 2>/dev/null || true)"
[ -f "$D/env.json" ] && echo "CLIO_ENV=$(cat "$D/env.json")"
[ -f "$D/code/HASH" ] && echo "CLIO_CODE=$(cat "$D/code/HASH")"
[ -f "$D/{base}.json" ] && echo "CLIO_NODE=$(cat "$D/{base}.json")"
# A node half answers only inside its own network namespace, as sandboxed servers have.
NS=$(readlink /proc/self/ns/net 2>/dev/null | tr -cd 0-9)
echo "CLIO_NS=$NS"
if [ -n "$NS" ]; then
  [ -f "$D/{base}-$NS.json" ] && echo "CLIO_OWNNODE=$(cat "$D/{base}-$NS.json")"
  if [ -f "$D/{base}.lock" ] && command -v flock >/dev/null 2>&1; then
    if flock -n "$D/{base}.lock" true; then echo "CLIO_LOCKED=0"; else echo "CLIO_LOCKED=1"; fi
  fi
fi
exit 0
"""


def node_slot(
    marks: dict[str, str], base: str = "node"
) -> tuple[str, dict[str, Any] | None]:
    """The record name this front uses, and the node half it names, if any.

    A node half answers only inside its own network namespace, and its pid
    means nothing outside its pid namespace. A host that confines each server
    it launches in fresh namespaces can leave a node half another server
    started holding the folder. That one is never linked to or stopped: this
    front runs its own beside it, under ``<base>-<namespace>``.
    """
    node = json.loads(marks["NODE"]) if marks.get("NODE") else None
    ns = marks.get("NS", "")
    if not ns:
        return base, node
    if marks.get("OWNNODE"):
        return f"{base}-{ns}", json.loads(marks["OWNNODE"])
    held = marks.get("LOCKED") == "1"
    theirs = node is not None and str(node.get("net") or ns) != ns
    if theirs and not held:
        return base, None
    if theirs or (node is None and held):
        return f"{base}-{ns}", None
    return base, node


PYINFO = (
    "import json,platform,sys,sysconfig;"
    "print('CLIO_PYINFO='+json.dumps({'python':'%d.%d'%sys.version_info[:2],"
    "'platform':sysconfig.get_platform(),'machine':platform.machine(),"
    "'libc':list(platform.libc_ver())}))"
)

GL_PROBE = (
    "from paraview.simple import *\n"
    "v=CreateRenderView();Show(Sphere(),v);Render(v)\n"
    "print('CLIO_GL='+('ok' if v.GetRenderWindow().SupportsOpenGL() else 'no'))"
)


def gl_probe_script(pvpython: str) -> str:
    """Shell lines that report whether offscreen rendering works natively.

    Login and compute nodes often have no X server, EGL or system OSMesa. The
    Kitware binaries ship their own OSMesa behind the launcher's ``--mesa``
    switch, so the probe tries that when native rendering fails.
    """
    pv = shlex.quote(pvpython)
    probe = shlex.quote(GL_PROBE)
    return (
        "echo 'Trying offscreen rendering' >&2\n"
        f"if {pv} --force-offscreen-rendering -c {probe} "
        "| grep -q '^CLIO_GL=ok'; then echo CLIO_GLMODE=native\n"
        f"elif echo 'Trying the Mesa that ParaView ships' >&2; {pv} --mesa "
        f"--force-offscreen-rendering -c {probe} "
        "| grep -q '^CLIO_GL=ok'; then echo CLIO_GLMODE=mesa\n"
        "else echo CLIO_GLMODE=none; fi"
    )


def platform_tags(info: dict[str, Any]) -> tuple[list[str], str]:
    """pip ``--platform`` values and the uv ``--python-platform`` for pvpython."""
    machine = info.get("machine") or "x86_64"
    machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine.lower(), machine)
    plat = str(info.get("platform", ""))
    if plat.startswith("win"):
        return (["win_amd64"], "x86_64-pc-windows-msvc")
    if plat.startswith("macosx"):
        tag_machine = "arm64" if machine == "aarch64" else machine
        return ([f"macosx_11_0_{tag_machine}"], f"{machine}-apple-darwin")
    libc = info.get("libc") or ["glibc", "2.17"]
    try:
        major, minor = (int(p) for p in str(libc[1]).split(".")[:2])
    except (ValueError, IndexError):
        major, minor = 2, 17
    pip = [
        f"manylinux_{major}_{m}_{machine}"
        for m in sorted({minor, 28, 17})
        if m <= minor
    ]
    pip.append(f"manylinux2014_{machine}")
    uv_minor = 28 if (major, minor) >= (2, 28) else 17
    return (sorted(set(pip)), f"{machine}-manylinux_2_{uv_minor}")


def build_script(folder: str, pyinfo: dict[str, Any], pvpython: str | None) -> str:
    """Install the trame packages into ``folder/site`` for pvpython's Python."""
    pip_plats, uv_plat = platform_tags(pyinfo)
    packages = " ".join(config.NODE_PACKAGES)
    plats = " ".join(f"--platform {p}" for p in pip_plats)
    pyver = pyinfo["python"]
    # Not --quiet: each line an install prints is the sign it has not stalled.
    pip_args = (
        f"install --disable-pip-version-check --no-input --progress-bar off "
        f'--timeout 20 --retries 1 --target "$T" --python-version {pyver} {plats} '
        f"--only-binary=:all: {packages}"
    )
    pv = shlex.quote(pvpython) if pvpython else '""'
    return f"""
D={shlex.quote(folder)}
T="$D/site.building.$$"
rm -rf "$T"; mkdir -p "$T"
ok=1
if command -v uv >/dev/null 2>&1; then
  echo "Installing with uv" >&2
  UV_HTTP_TIMEOUT=20 uv pip install --target "$T" --python-version {pyver} \\
    --python-platform {uv_plat} {packages} && ok=0
fi
if [ $ok -ne 0 ] && python3 -m pip --version >/dev/null 2>&1; then
  echo "Installing with pip" >&2
  python3 -m pip {pip_args} && ok=0
fi
if [ $ok -ne 0 ] && [ -n {pv} ] && {pv} -m pip --version >/dev/null 2>&1; then
  echo "Installing with pvpython's pip" >&2
  {pv} -m pip {pip_args} && ok=0
fi
if [ $ok -ne 0 ]; then rm -rf "$T"; echo "CLIO_BUILD=failed"; exit 3; fi
rm -rf "$D/site"; mv "$T" "$D/site"
echo "CLIO_BUILD=ok"
"""


@dataclass
class Link:
    """A live attachment: the transport, the forwarded port and the client."""

    transport: Transport
    folder: str
    local_port: int
    client: NodeClient
    node: dict[str, Any]
    pvpython: str
    paraview: str
    built_from: str | None = None
    built_now: bool = False
    uploaded_now: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    slot: str = "node"

    @property
    def token(self) -> str:
        return str(self.node["token"])

    def port_for(self, remote: Any) -> int:
        """The local port of one of the node half's view ports: each open scene has
        its own. The node's own port is the link's."""
        try:
            remote = int(remote)
        except (TypeError, ValueError):
            return self.local_port
        if remote == int(self.node["port"]):
            return self.local_port
        return self.transport.forward(remote, keep=True)


class Attacher:
    """Keeps at most one link to a node half and re-attaches when it drops.

    Once linked, a node half that has gone is reported (``pvserver_lost`` or
    ``node_lost``, with the reason it left) instead of being replaced by an empty
    one; only opening a view starts a new node half then.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.link: Link | None = None
        self.local_port: int | None = None
        self.lock = threading.RLock()
        self._stop = threading.Event()
        self._heart: threading.Thread | None = None
        #: ParaView runs on this Windows machine: steps run in Python, not a shell.
        self._native = False
        #: Set to end a node half's start that is being waited for.
        self.cancel = threading.Event()
        #: Told each new line a starting node half logs.
        self.progress: Callable[[str], None] | None = None
        #: The running call's stop and progress, while ``ensure`` serves it.
        self._call: tuple[threading.Event | None, Callable[[str], None] | None] = (
            None,
            None,
        )
        #: Whether this front has had a node half; one that goes is then reported.
        self.had_link = False
        #: The last scene opened through this attacher, for the lost-node advice.
        self.scene: str | None = None
        self.base = node_base(settings.pvserver)
        #: The linked node half's port and token, and each open scene's view port
        #: there and here: what a reopen after its pvserver went serves again.
        self._main: tuple[int, str] | None = None
        self.views: dict[str, tuple[int, int]] = {}
        #: Set while a reopen starts a node half: the port and token it keeps.
        self._reuse: tuple[int, str] | None = None
        self._reopens: list[float] = []

    # -- public ----------------------------------------------------------------
    def ensure(
        self,
        opening: bool = False,
        stop: threading.Event | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> Link:
        """A healthy link, attaching first if there is none.

        Unless ``opening``, a node half that was linked and has gone raises the
        reason it went, rather than starting an empty one in its place. A node half
        that is busy is waited for, with each wait told to ``progress``, until it
        answers or ``stop`` is set.
        """
        with self.lock:
            self._call = (stop, progress)
            try:
                link = self.link
                if link is not None:
                    if link.transport.alive() and self._answers(link.client):
                        return link
                    self._drop()
                self.link = self._attach(opening)
                self.had_link = True
                self._linked(self.link)
                self._start_heartbeat()
                return self.link
            finally:
                self._call = (None, None)

    def _linked(self, link: Link) -> None:
        """A new link: views of another node half are not this one's to reopen."""
        main = (int(link.node["port"]), link.token)
        if main != self._main:
            self.views.clear()
            self._main = main

    def remember(self, scene: str, remote: Any, local: int) -> None:
        """An opened scene's view port on the node half and here, for a reopen."""
        if self._main is None:
            return
        try:
            port = int(remote)
        except (TypeError, ValueError):
            port = self._main[0]
        self.views.pop(scene, None)
        self.views[scene] = (port, local)

    def reopen(self) -> dict[str, Any]:
        """For a live view whose ParaView went: start the node half again on its
        old port and token, and open each scene it showed on its old view port, so
        the views the host shows draw again where they are. Each scene opens from
        its saved manifest. The status after it, with ``reopen``: ``reopened``,
        ``waiting`` (the old node half has not ended yet), ``failed`` (the start did
        not work, so a later try may), or ``given_up`` (too many in a short time).
        """
        with self.lock:
            state = self.status()
            if state["state"] in ("running", "busy") or not self.views:
                return state
            if state.get("ending"):
                return {**state, "reopen": "waiting"}
            if self._main is None:
                return state
            now = time.monotonic()
            self._reopens = [t for t in self._reopens if now - t < REOPEN_WINDOW]
            if len(self._reopens) >= REOPEN_LIMIT:
                return {
                    **state,
                    "reopen": "given_up",
                    "text": f"{state['text']} The view was reopened {REOPEN_LIMIT} "
                    "times in ten minutes, so it is not reopened again; "
                    "open_view(scene=..., new=true) starts the scene afresh.",
                }
            self._reopens.append(now)
            main_port, token = self._main
            views = sorted(self.views.items(), key=lambda v: v[1][0] != main_port)
            main_local = next(
                (local for _, (remote, local) in views if remote == main_port),
                self.local_port,
            )
            current = self.scene
            self._drop()
            self.local_port = main_local
            self._reuse = (main_port, token)
            try:
                link = self._attach(opening=True)
            except ParaViewError as exc:
                return {
                    **state,
                    "reopen": "failed",
                    "text": f"{state['text']} Reopening it did not work yet: "
                    f"{exc.message}",
                }
            finally:
                self._reuse = None
            self.link = link
            self.had_link = True
            self._main = (int(link.node["port"]), link.token)
            self._start_heartbeat()
            opened, missed = self._open_again(link, views, main_port)
            if current in opened and len(opened) > 1:
                try:
                    link.client.command("open_view", {"scene": current})
                except (LinkError, ParaViewError):
                    pass
            self.views = {s: opened[s] for s, _ in views if s in opened}
            text = (
                f"ParaView went ({state['text']}) and was started again: "
                f"{', '.join(opened) or 'no scene'} reopened from the saved scene."
            )
            if missed:
                text += " Not reopened: " + "; ".join(missed) + "."
            return {"state": "running", "reopen": "reopened", "text": text}

    def _open_again(
        self, link: Link, views: list[tuple[str, tuple[int, int]]], main_port: int
    ) -> tuple[dict[str, tuple[int, int]], list[str]]:
        """Open each scene again on its old view port, forwarded to its old local
        port; the scenes opened, with their ports, and why any did not."""
        opened: dict[str, tuple[int, int]] = {}
        missed: list[str] = []
        for scene, (remote, local) in views:
            args: dict[str, Any] = {"scene": scene}
            if remote != main_port:
                args["view_port"] = remote
            try:
                summary = link.client.command("open_view", args)
                now = int((summary.get("service") or {}).get("port") or main_port)
                if now == int(link.node["port"]):
                    here = link.local_port
                else:
                    here = link.transport.forward(now, local, keep=True)
            except (LinkError, ParaViewError, TypeError, ValueError) as exc:
                missed.append(f"{scene} ({exc})")
                continue
            opened[scene] = (now, here)
            if here != local:
                missed.append(
                    f"{scene} opened on another port, so its view must be opened "
                    "again with open_view"
                )
        return opened, missed

    def close(self) -> None:
        self._stop.set()
        with self.lock:
            self._drop()

    def status(self) -> dict[str, Any]:
        """Whether the linked node half runs, without waiting on it.

        For a live view that cannot reach ParaView: ``running`` (it answers),
        ``busy`` (it holds its loop, or a start or check is under way), ``lost``
        (it went, with why) or ``closed`` (no node half was linked).
        """
        if not self.lock.acquire(blocking=False):
            return {"state": "busy", "text": "ParaView is starting or busy."}
        try:
            link = self.link
            if link is None:
                return {"state": "closed", "text": "No ParaView node half is open."}
            try:
                health = link.client.health()
                if health.get("lost"):
                    gone = gone_error(
                        json.dumps({"reason": health["lost"]}),
                        "",
                        link.transport.where,
                        self.scene,
                    )
                    # The node half still runs: it saves its scenes and then ends.
                    return {
                        "state": "lost",
                        "code": gone.code,
                        "text": gone.message,
                        "ending": True,
                    }
                return {"state": "running", "text": "ParaView answers."}
            except LinkError as exc:
                if exc.busy:
                    return {
                        "state": "busy",
                        "text": "ParaView is busy with a long step, such as a "
                        "render or a measurement.",
                    }
            if link.transport.alive():
                try:
                    gone = self._gone(link.transport, link.folder, link.slot)
                except ParaViewError as exc:
                    gone = exc
            else:
                gone = gone_error("", "", link.transport.where, self.scene)
            return {"state": "lost", "code": gone.code, "text": gone.message}
        finally:
            self.lock.release()

    def describe(self) -> dict[str, Any]:
        target = self.settings.target
        link = self.link
        out: dict[str, Any] = {
            "where": (
                f"on the ParaView server at {self.settings.pvserver}, through "
                f"a client {target.describe()}"
                if self.settings.pvserver
                else target.describe()
            ),
            "address": target.address,
            "via": target.via,
            "prepare": target.prepare,
            "pvserver": self.settings.pvserver,
            "attached": link is not None,
        }
        if self.settings.version:
            out["paraview_version"] = self.settings.version
            out["paraview_client"] = client.installed(self.settings.version)
        if link is not None:
            out.update(
                folder=link.folder,
                pvpython=link.pvpython,
                paraview=link.paraview,
                node_pid=link.node.get("pid"),
                pvserver_session=link.node.get("session"),
                environment_built_from=link.built_from,
                built_this_attach=link.built_now,
                code_copied_this_attach=link.uploaded_now,
                attach_seconds=link.timings,
            )
        return out

    # -- attaching ---------------------------------------------------------------
    def _drop(self) -> None:
        if self.link is not None:
            self.link.transport.close()
            self.link = None

    def _attach(self, opening: bool = True) -> Link:
        started = time.monotonic()
        settings = self.settings
        target = settings.target
        transport = transport_for(target, settings.connect_timeout)
        transport.watch = self._watch()
        try:
            transport.open()
            return self._attach_over(transport, started, opening)
        except ParaViewError as exc:
            transport.close()
            if transport.remote and settings.pvserver and exc.code in _SSH_CODES:
                raise ParaViewError(exc.code, exc.message + PREFER_FORWARD) from exc
            raise
        except BaseException:
            transport.close()
            raise

    def _attach_over(
        self, transport: Transport, started: float, opening: bool = True
    ) -> Link:
        settings = self.settings
        base = self.base
        timings: dict[str, float] = {"connect": time.monotonic() - started}
        folder_here = None if transport.remote else local_folder(config.SERVER_VERSION)
        self._native = not transport.remote and native.wanted()
        probe_err = ""
        if self._native:
            marks = native.probe(str(folder_here), base)
        else:
            probe = transport.run(
                probe_script(config.SERVER_VERSION, folder_here, base),
                label="Checking the ParaView folder",
            )
            marks = parse_markers(probe.stdout)
            probe_err = probe.stderr.strip() or probe.stdout.strip()
        folder = marks.get("DIR")
        pvpython = marks.get("PVPYTHON") or ""
        if not folder:
            raise ParaViewError(
                "probe_failed",
                f"Could not prepare the ParaView folder {transport.where}: {probe_err}",
            )
        env = json.loads(marks["ENV"]) if marks.get("ENV") else None
        if not transport.remote and settings.version:
            pvpython = self._local_pvpython(pvpython, env)
        if not pvpython:
            detail = ""
            if settings.target.prepare and probe_err:
                detail = f"The prepare line said: {probe_err[-300:]}"
            fetchable = sorted(v for v, m in client.BUILDS if m == client.machine())
            if not transport.remote and not settings.version and fetchable:
                detail += (
                    f"{' ' if detail else ''}To fetch Kitware's ParaView to this "
                    f"machine instead, set `version` ('ParaView version') to "
                    f"{' or '.join(fetchable)}; it must match any pvserver it joins."
                )
            raise no_paraview(transport.where, detail)
        if env is None or env.get("pvpython") != pvpython:
            env = self._interrogate(transport, pvpython)
        if settings.version and env.get("paraview") != settings.version:
            raise ParaViewError(
                "paraview_version_mismatch",
                f"The ParaView {transport.where} is {env.get('paraview')}, and "
                f"{settings.version} is wanted: a ParaView joins only a server of its "
                "own version.",
            )
        # A prepare line can bring its own GL libraries, so the probe result is
        # kept per prepare line.
        gl_for = settings.target.prepare or ""
        if "gl_args" not in env or env.get("gl_for") != gl_for:
            env["gl_args"] = self._gl_args(transport, pvpython)
            env["gl_for"] = gl_for
            if env.get("built"):
                self._write_env(transport, folder, env)
        gl_args = [str(a) for a in env.get("gl_args") or []]
        built_from = None
        if not env.get("built"):
            built_from = self._build(transport, folder, env, pvpython)
            env["built"] = built_from
            self._write_env(transport, folder, env)
        timings["prepare"] = time.monotonic() - started
        bundle, code_hash = code_bundle()
        uploaded = marks.get("CODE") != code_hash
        if uploaded:
            self._upload(transport, folder, bundle)
        slot, node = node_slot(marks, base)
        # The rendering switch rides on the node's hash, so a node half started
        # without it is replaced rather than reused.
        node_hash = "+".join([code_hash, *(a.lstrip("-") for a in gl_args)])
        if settings.pvserver:
            # A node half joins one pvserver for its life; another one needs a new node.
            node_hash += "+pv=" + settings.pvserver
        link = None
        if node is not None and node.get("code_hash") == node_hash:
            # Another front with other knobs (one a connector change superseded,
            # say) may still show views through it: share it rather than stop it.
            # Its knobs hold until it ends; the next node half takes these.
            link = self._try_link(transport, folder, node, pvpython, env)
        if link is None and not opening and self.had_link:
            if node is not None and self._matches(node, node_hash):
                waiting = (
                    f" It joined the pvserver at {settings.pvserver}: if that has "
                    "ended (its job finished, say), the node half may be waiting on "
                    "it, and open_view with a running pvserver starts a new one."
                    if settings.pvserver
                    else ""
                )
                raise ParaViewError(
                    "node_unreachable",
                    f"The ParaView node half {transport.where} is still running but "
                    "did not answer, so nothing was done. It may be busy with a "
                    "long render; retry, or call open_view to open the view again."
                    + waiting,
                )
            # The node half this front was linked to has gone: say why, rather
            # than show an empty scene in its place.
            raise self._gone(transport, folder, slot)
        if link is None:
            if node is not None:
                freed = self._stop_node(transport, folder, node, slot)
                if not freed and slot == base:
                    # Another sandbox's node half would not let go: run beside it.
                    slot = native.own_slot(base)
            node = self._start_node(
                transport, folder, pvpython, node_hash, gl_args, slot
            )
            link = self._try_link(transport, folder, node, pvpython, env)
        if link is None:
            raise ParaViewError(
                "node_unhealthy",
                f"The ParaView node half started {transport.where} but does not "
                f"answer. Its log is {folder}/{slot}.log.",
            )
        link.slot = slot
        link.built_from = built_from or env.get("built")
        link.built_now = built_from is not None
        link.uploaded_now = uploaded
        timings["total"] = time.monotonic() - started
        link.timings = {k: round(v, 2) for k, v in timings.items()}
        return link

    def _matches(self, node: dict[str, Any], code_hash: str) -> bool:
        """Whether a running node half was started with today's code and knobs."""
        settings = self.settings
        if node.get("code_hash") != code_hash:
            return False
        if node.get("idle_seconds") not in (None, settings.idle_seconds):
            return False
        threshold = settings.threshold_mb
        return threshold is None or node.get("threshold_mb") == threshold

    def _local_pvpython(self, found: str, env: dict[str, Any] | None) -> str:
        """The pvpython of the wanted version here: one already fetched, the one on
        PATH when it is that version, else one fetched now."""
        want = str(self.settings.version)
        fetched = client.installed(want)
        if fetched:
            return fetched
        if found:
            known = (
                env.get("paraview") if env and env.get("pvpython") == found else None
            )
            if (known or native.version_of(found, self._watch())) == want:
                return found
        return client.ensure(want)

    def _interrogate(self, transport: Transport, pvpython: str) -> dict[str, Any]:
        pv = shlex.quote(pvpython)
        if self._native:
            done = native.interrogate(pvpython, PYINFO, self._watch())
        else:
            done = transport.run(
                f"{pv} --version 2>/dev/null | head -n1 | sed 's/^/CLIO_VERSION=/'\n"
                f"{pv} --dr -c {shlex.quote(PYINFO)} 2>/dev/null | grep '^CLIO_'",
                label="Asking pvpython its version",
            )
        marks = parse_markers(done.stdout)
        if "PYINFO" not in marks:
            raise ParaViewError(
                "pvpython_failed",
                f"pvpython at {pvpython} {transport.where} did not run: "
                f"{done.stderr.strip()[-400:]}",
            )
        version_text = marks.get("VERSION", "")
        number = version_text.rsplit(" ", 1)[-1]
        try:
            major, minor = (int(p) for p in (number.split(".") + ["0"])[:2])
        except ValueError:
            major, minor = 0, 0
        if (major, minor) < (5, 13):
            raise ParaViewError(
                "paraview_too_old",
                f"The ParaView {transport.where} is {version_text or 'unknown'}; "
                "ParaView 5.13 or later is required.",
            )
        info = json.loads(marks["PYINFO"])
        return {"pvpython": pvpython, "paraview": number, "pyinfo": info}

    def _gl_args(self, transport: Transport, pvpython: str) -> list[str]:
        """Launcher switches pvpython needs to render offscreen on this host."""
        if self._native:
            # Windows builds render offscreen through the desktop's own OpenGL.
            return []
        done = transport.run(
            gl_probe_script(pvpython), label="Checking offscreen rendering"
        )
        mode = parse_markers(done.stdout).get("GLMODE")
        return ["--mesa"] if mode == "mesa" else []

    def _build(
        self, transport: Transport, folder: str, env: dict[str, Any], pvpython: str
    ) -> str:
        script = build_script(folder, env["pyinfo"], pvpython)
        label = "Installing trame for the node half"
        try:
            if self._native:
                done = native.build(folder, env["pyinfo"], pvpython, transport.watch)
            else:
                done = transport.run(script, label=label)
        except ParaViewError as exc:
            # A stalled install may still be done from the jump host; a cancel ends it.
            if exc.code != "setup_stalled":
                raise
            done = Completed(1, "", exc.message)
        if parse_markers(done.stdout).get("BUILD") == "ok":
            return "node"
        failures = [f"{transport.where}: {_tail(done)}"]
        jump = getattr(transport, "jump_host", lambda: None)()
        if jump:
            from .config import Target
            from .transport import SshTransport

            helper = SshTransport(Target(address=jump), self.settings.connect_timeout)
            helper.watch = transport.watch
            try:
                helper.open()
                done = helper.run(
                    build_script(folder, env["pyinfo"], None), label=label
                )
                if parse_markers(done.stdout).get("BUILD") == "ok":
                    check = transport.run(
                        f"test -d {shlex.quote(folder)}/site/trame",
                        label="Checking the install is shared",
                    )
                    if check.code == 0:
                        return f"jump host {jump}"
                    failures.append(
                        f"{jump} built it, but {folder} is not shared with "
                        f"{transport.where}"
                    )
                else:
                    failures.append(f"{jump}: {_tail(done)}")
            except ParaViewError as exc:
                if exc.code == "cancelled":
                    raise
                failures.append(f"{jump}: {exc.message}")
            finally:
                helper.close()
        pip_plats, _ = platform_tags(env["pyinfo"])
        packages = " ".join(config.NODE_PACKAGES)
        joined = "; ".join(failures)
        if any(mark in joined for mark in _WRITE_DENIED):
            raise ParaViewError(
                "node_folder_unwritable",
                "Could not install trame, trame-vtk and trame-vuetify for the ParaView "
                f"{transport.where}: writing into {folder} was denied. A host that "
                "confines this server must let it write there; set CLIO_KIT_CACHE_DIR "
                f"to a folder it may write and keep. ({joined})",
            )
        raise ParaViewError(
            "no_package_index",
            "Could not install trame, trame-vtk and trame-vuetify for the ParaView "
            f"{transport.where}: "
            + (
                f"neither it nor the jump host {jump} could reach"
                if jump
                else "it could not reach"
            )
            + " the Python package index. ("
            + joined
            + ") Install them from a machine that can, into the same folder: pip "
            f"install --target {folder}/site --python-version "
            f"{env['pyinfo']['python']} "
            + " ".join(f"--platform {p}" for p in pip_plats)
            + f" --only-binary=:all: {packages}",
        )

    def _write_env(
        self, transport: Transport, folder: str, env: dict[str, Any]
    ) -> None:
        payload = json.dumps(env).encode()
        if self._native:
            native.write_env(folder, env)
            return
        transport.run(
            f"cat > {shlex.quote(folder)}/env.json",
            stdin=payload,
            label="Saving the ParaView facts",
        )

    def _upload(self, transport: Transport, folder: str, bundle: bytes) -> None:
        f = shlex.quote(folder)
        if self._native:
            done = native.upload(folder, bundle)
        else:
            done = transport.run(
                f"rm -rf {f}/code.new && mkdir -p {f}/code.new && tar -xzf - -C "
                f"{f}/code.new && rm -rf {f}/code && mv {f}/code.new {f}/code",
                stdin=bundle,
                label="Copying the node half",
            )
        if done.code != 0:
            raise ParaViewError(
                "upload_failed",
                f"Could not copy the node half {transport.where}: {_tail(done)}",
            )

    def _stop_node(
        self,
        transport: Transport,
        folder: str,
        node: dict[str, Any],
        slot: str = "node",
    ) -> bool:
        """Stop a node half; whether its slot is free for a new one."""
        pid = int(node.get("pid") or 0)
        if self._native:
            return native.stop(folder, pid, slot)
        f = shlex.quote(folder)
        script = f"rm -f {f}/{slot}.json\n"
        if pid > 0:
            script += f"kill {pid} 2>/dev/null; sleep 1; kill -9 {pid} 2>/dev/null\n"
        transport.run(script + "exit 0", label="Stopping the old node half")
        return True

    def _start_node(
        self,
        transport: Transport,
        folder: str,
        pvpython: str,
        code_hash: str,
        gl_args: list[str] | None = None,
        slot: str = "node",
    ) -> dict[str, Any]:
        settings = self.settings
        f = shlex.quote(folder)
        args = [
            *(gl_args or []),
            "--dr",
            "--force-offscreen-rendering",
            f"{folder}/code/run_node.py",
            "--root",
            folder,
            "--name",
            slot,
            "--idle-seconds",
            str(settings.idle_seconds),
            "--pvserver-port",
            str(settings.pvserver_port),
            "--code-hash",
            code_hash,
        ]
        if settings.threshold_mb is not None:
            args += ["--threshold-mb", str(settings.threshold_mb)]
        if settings.pvserver:
            args += ["--pvserver", settings.pvserver]
        reuse = self._reuse
        token_write = ""
        if reuse is not None:
            # A reopened node half serves on its old port with its old token, so the
            # views the host shows reach it again without being opened anew.
            token_path = f"{folder}/{slot}.token"
            args += ["--port", str(reuse[0]), "--token-file", token_path]
            if self._native:
                written = Path(token_path)
                written.write_text(reuse[1], "utf-8")
                if os.name != "nt":
                    written.chmod(0o600)
            else:
                token_write = (
                    f"(umask 077; printf '%s' {shlex.quote(reuse[1])} > "
                    f"{shlex.quote(token_path)})\n"
                )
        if self._native:
            return self._start_native(folder, [pvpython, *args], code_hash, slot)
        command = " ".join(shlex.quote(a) for a in [pvpython, *args])
        log = f"{f}/{slot}.log"
        begun = transport.run(
            f"rm -f {f}/{slot}.json {f}/{slot}.lost.json\ncd {f}\n{token_write}"
            f"[ -f {log} ] && mv -f {log} {f}/{slot}.prev.log\n"
            f"if command -v setsid >/dev/null 2>&1; then S=setsid; else S=; fi\n"
            f"nohup $S {command} > {log} 2>&1 < /dev/null &\n"
            'echo "CLIO_PID=$!"\n'
            "exit 0",
            label="Starting the node half",
        )
        pid = parse_markers(begun.stdout).get("PID", "")
        pid = pid if pid.isdigit() else "0"
        # A launcher that hands over to a child keeps the node half's arguments;
        # the bracket keeps the pattern from matching this script's own shell.
        alive = (
            f'if kill -0 {pid} 2>/dev/null || pgrep -u "$(id -u)" -f -- '
            f"{shlex.quote('[-]-name ' + slot + ' --idle-seconds')} >/dev/null 2>&1; "
            "then echo CLIO_ALIVE=1; else echo CLIO_ALIVE=0; fi\n"
        )
        failure = "Traceback\\|another node half\\|pvserver join failed"

        def look() -> StartLook:
            done = transport.run(
                f"cat {f}/{slot}.json 2>/dev/null; echo\n{alive}"
                f'echo "CLIO_SIZE=$(wc -c < {log} 2>/dev/null || echo 0)"\n'
                f"echo \"CLIO_FAILED=$(grep -c '{failure}' {log} 2>/dev/null)\"\n"
                f"echo \"CLIO_LAST=$(tail -n 1 {log} 2>/dev/null | tr -d '\\r' "
                '| cut -c1-400)"\n'
                "exit 0",
                label="Looking at the starting node half",
            )
            first = (done.stdout.splitlines() or [""])[0].strip()
            record = json.loads(first) if first.startswith("{") else None
            marks = parse_markers(done.stdout)
            size = marks.get("SIZE", "0").strip()
            failed = marks.get("FAILED", "0").strip()
            is_alive = marks.get("ALIVE") != "0"
            tail = ""
            if (failed.isdigit() and int(failed) > 0) or not is_alive:
                tail = transport.run(
                    f"tail -n 25 {log}", label="Reading its log"
                ).stdout
            return StartLook(
                record=record,
                failed=tail if failed.isdigit() and int(failed) > 0 else None,
                alive=is_alive,
                size=int(size) if size.isdigit() else 0,
                last=marks.get("LAST", ""),
                tail=tail,
            )

        def stop() -> None:
            if pid != "0":
                transport.run(f"kill {pid} 2>/dev/null; exit 0", label="Stopping it")

        return wait_for_start(
            look,
            code_hash,
            transport.where,
            f"{folder}/{slot}.log",
            settings.start_stall,
            self.cancel,
            self.progress,
            stop,
        )

    def _start_native(
        self, folder: str, command: list[str], code_hash: str, slot: str
    ) -> dict[str, Any]:
        process = native.start(folder, command, slot)
        log_path = Path(folder) / f"{slot}.log"

        def look() -> StartLook:
            node, failed = native.started(folder, slot)
            alive = process.poll() is None
            try:
                size = log_path.stat().st_size
            except OSError:
                size = 0
            tail = native.log_tail(folder, slot)
            return StartLook(
                record=node,
                failed=failed,
                alive=alive,
                size=size,
                last=(tail.splitlines() or [""])[-1],
                tail=tail,
            )

        def stop() -> None:
            if process.poll() is None:
                process.terminate()

        return wait_for_start(
            look,
            code_hash,
            "on this machine",
            str(log_path),
            self.settings.start_stall,
            self.cancel,
            self.progress,
            stop,
        )

    def _gone(self, transport: Transport, folder: str, slot: str) -> ParaViewError:
        """Why the node half this front was linked to has gone, as an error."""
        if self._native:
            note, tail = native.lost_note(folder, slot), native.log_tail(folder, slot)
        else:
            f = shlex.quote(folder)
            done = transport.run(
                f"cat {f}/{slot}.lost.json 2>/dev/null; echo; echo CLIO_TAIL\n"
                f"tail -n 25 {f}/{slot}.log 2>/dev/null; exit 0",
                label="Reading why the node half went",
            )
            note, _, tail = done.stdout.partition("CLIO_TAIL")
        return gone_error(note.strip(), tail.strip(), transport.where, self.scene)

    def _try_link(
        self,
        transport: Transport,
        folder: str,
        node: dict[str, Any],
        pvpython: str,
        env: dict[str, Any],
    ) -> Link | None:
        try:
            port = transport.forward(int(node["port"]), self.local_port)
        except ParaViewError:
            return None
        client = NodeClient(port, str(node["token"]))
        if not self._answers(client):
            return None
        self.local_port = port
        return Link(
            transport=transport,
            folder=folder,
            local_port=port,
            client=client,
            node=node,
            pvpython=pvpython,
            paraview=str(env.get("paraview", "")),
        )

    def _answers(self, client: NodeClient) -> bool:
        """Whether the node half answers, waiting while it is only busy.

        A software volume render holds the node's loop for tens of seconds. A health
        check that times out then means "busy", not "gone": restarting it would lose
        the open scene. So a busy node half is waited for as long as it stays busy,
        and the wait is said; only a refused link fails. The person ends the wait by
        cancelling the call.
        """
        stop, said = self._call
        began = time.monotonic()
        while True:
            try:
                client.health()
                return True
            except LinkError as exc:
                if not exc.busy:
                    return False
            if self.cancel.is_set() or (stop is not None and stop.is_set()):
                raise ParaViewError(
                    "cancelled",
                    "The call was cancelled while ParaView was busy; nothing was done.",
                )
            tell = said or self.progress
            if tell is not None:
                tell(
                    f"ParaView has not answered for {time.monotonic() - began:.0f} s: "
                    "it is busy (a long render holds it), not gone, so this waits"
                )

    def _watch(self) -> Watch:
        """How setup steps are watched: the running call's cancel and progress."""

        def cancelled() -> bool:
            stop = self._call[0]
            return self.cancel.is_set() or (stop is not None and stop.is_set())

        def tell(text: str) -> None:
            said = self._call[1] or self.progress
            if said is not None:
                said(text)

        return Watch(self.settings.setup_stall, cancelled, tell)

    # -- keeping the link warm ----------------------------------------------------
    def _start_heartbeat(self) -> None:
        if self._heart is not None and self._heart.is_alive():
            return
        self._heart = threading.Thread(target=self._beat, daemon=True)
        self._heart.start()

    def _beat(self) -> None:
        while not self._stop.wait(60.0):
            link = self.link
            if link is None:
                continue
            try:
                link.client.ping()
            except LinkError:
                pass


def _tail(done: Completed) -> str:
    text = (done.stderr.strip() or done.stdout.strip()).splitlines()
    return " ".join(text[-3:]) if text else f"exit status {done.code}"
