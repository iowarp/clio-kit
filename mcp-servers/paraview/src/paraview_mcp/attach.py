"""Attaching the front to a node half: prepare the folder, start or find it.

The node half lives in ``~/.clio/paraview/<server version>/`` on the machine
where ParaView runs:

- ``site/``: trame and trame-vtk, built for that machine's pvpython;
- ``code/``: the node half's code, copied over the same connection;
- ``node.json``: the running node half's pid, port and token (mode 0600);
- ``scenes/<id>/manifest.json``: each scene, written after every commit.
"""

from __future__ import annotations

import hashlib
import io
import json
import shlex
import tarfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config
from .config import Settings
from .errors import ParaViewError, no_paraview
from .link import LinkError, NodeClient
from .transport import Completed, Transport, transport_for

NODE_SOURCE = Path(__file__).resolve().parent / "node"
RUN_NODE = """import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paraview_node.main import main
main()
"""


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

    On another machine it is ``~/.clio/paraview/<version>/``. Here it sits in
    clio-kit's data folder, which a host that confines the servers it launches
    (CLIO's write fence) lets them write.
    """
    import platformdirs

    base = platformdirs.user_data_dir("clio-kit", appauthor=False)
    return str(Path(base) / "paraview" / version)


def probe_script(version: str, folder: str | None = None) -> str:
    where = shlex.quote(folder) if folder else f'"$HOME/.clio/paraview/{version}"'
    return f"""
D={where}
mkdir -p "$D" && chmod 700 "$D"
echo "CLIO_DIR=$D"
echo "CLIO_PVPYTHON=$(command -v pvpython 2>/dev/null || true)"
[ -f "$D/env.json" ] && echo "CLIO_ENV=$(cat "$D/env.json")"
[ -f "$D/code/HASH" ] && echo "CLIO_CODE=$(cat "$D/code/HASH")"
[ -f "$D/node.json" ] && echo "CLIO_NODE=$(cat "$D/node.json")"
exit 0
"""


PYINFO = (
    "import json,platform,sys,sysconfig;"
    "print('CLIO_PYINFO='+json.dumps({'python':'%d.%d'%sys.version_info[:2],"
    "'platform':sysconfig.get_platform(),'machine':platform.machine(),"
    "'libc':list(platform.libc_ver())}))"
)


def platform_tags(info: dict[str, Any]) -> tuple[list[str], str]:
    """pip ``--platform`` values and the uv ``--python-platform`` for pvpython."""
    machine = info.get("machine") or "x86_64"
    machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine.lower(), machine)
    plat = str(info.get("platform", ""))
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
    """Install trame and trame-vtk into ``folder/site`` for pvpython's Python."""
    pip_plats, uv_plat = platform_tags(pyinfo)
    packages = " ".join(config.NODE_PACKAGES)
    plats = " ".join(f"--platform {p}" for p in pip_plats)
    pyver = pyinfo["python"]
    pip_args = (
        f"install --quiet --disable-pip-version-check --no-input --timeout 20 "
        f'--retries 1 --target "$T" --python-version {pyver} {plats} '
        f"--only-binary=:all: {packages}"
    )
    pv = shlex.quote(pvpython) if pvpython else '""'
    return f"""
D={shlex.quote(folder)}
T="$D/site.building.$$"
rm -rf "$T"; mkdir -p "$T"
ok=1
if command -v uv >/dev/null 2>&1; then
  UV_HTTP_TIMEOUT=20 uv pip install --quiet --target "$T" --python-version {pyver} \\
    --python-platform {uv_plat} {packages} && ok=0
fi
if [ $ok -ne 0 ] && python3 -m pip --version >/dev/null 2>&1; then
  python3 -m pip {pip_args} && ok=0
fi
if [ $ok -ne 0 ] && [ -n {pv} ] && {pv} -m pip --version >/dev/null 2>&1; then
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

    @property
    def token(self) -> str:
        return str(self.node["token"])


class Attacher:
    """Keeps at most one link to a node half and re-attaches when it drops."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.link: Link | None = None
        self.local_port: int | None = None
        self.lock = threading.RLock()
        self._stop = threading.Event()
        self._heart: threading.Thread | None = None

    # -- public ----------------------------------------------------------------
    def ensure(self) -> Link:
        """A healthy link, attaching first if there is none."""
        with self.lock:
            link = self.link
            if link is not None:
                try:
                    if link.transport.alive():
                        link.client.health()
                        return link
                except LinkError:
                    pass
                self._drop()
            self.link = self._attach()
            self._start_heartbeat()
            return self.link

    def close(self) -> None:
        self._stop.set()
        with self.lock:
            self._drop()

    def describe(self) -> dict[str, Any]:
        target = self.settings.target
        link = self.link
        out: dict[str, Any] = {
            "where": target.describe(),
            "address": target.address,
            "via": target.via,
            "prepare": target.prepare,
            "attached": link is not None,
        }
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

    def _attach(self) -> Link:
        started = time.monotonic()
        settings = self.settings
        target = settings.target
        transport = transport_for(target, settings.connect_timeout)
        try:
            transport.open()
            return self._attach_over(transport, started)
        except BaseException:
            transport.close()
            raise

    def _attach_over(self, transport: Transport, started: float) -> Link:
        settings = self.settings
        timings: dict[str, float] = {"connect": time.monotonic() - started}
        folder_here = None if transport.remote else local_folder(config.SERVER_VERSION)
        probe = transport.run(
            probe_script(config.SERVER_VERSION, folder_here), timeout=120
        )
        marks = parse_markers(probe.stdout)
        folder = marks.get("DIR")
        pvpython = marks.get("PVPYTHON") or ""
        if not folder:
            raise ParaViewError(
                "probe_failed",
                f"Could not prepare the ParaView folder {transport.where}: "
                f"{probe.stderr.strip() or probe.stdout.strip()}",
            )
        if not pvpython:
            detail = ""
            if settings.target.prepare and probe.stderr.strip():
                detail = f"The prepare line said: {probe.stderr.strip()[-300:]}"
            raise no_paraview(transport.where, detail)
        env = json.loads(marks["ENV"]) if marks.get("ENV") else None
        if env is None or env.get("pvpython") != pvpython:
            env = self._interrogate(transport, pvpython)
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
        node = json.loads(marks["NODE"]) if marks.get("NODE") else None
        link = None
        if node is not None and self._matches(node, code_hash):
            link = self._try_link(transport, folder, node, pvpython, env)
        if link is None:
            if node is not None:
                self._stop_node(transport, folder, node)
            node = self._start_node(transport, folder, pvpython, code_hash)
            link = self._try_link(transport, folder, node, pvpython, env)
        if link is None:
            raise ParaViewError(
                "node_unhealthy",
                f"The ParaView node half started {transport.where} but does not "
                f"answer. Its log is {folder}/node.log.",
            )
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

    def _interrogate(self, transport: Transport, pvpython: str) -> dict[str, Any]:
        pv = shlex.quote(pvpython)
        done = transport.run(
            f"{pv} --version 2>/dev/null | head -n1 | sed 's/^/CLIO_VERSION=/'\n"
            f"{pv} --dr -c {shlex.quote(PYINFO)} 2>/dev/null | grep '^CLIO_'",
            timeout=180,
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
            major = int(number.split(".")[0])
        except ValueError:
            major = 0
        if major < 6:
            raise ParaViewError(
                "paraview_too_old",
                f"The ParaView {transport.where} is {version_text or 'unknown'}; "
                "ParaView 6.0 or later is required.",
            )
        info = json.loads(marks["PYINFO"])
        return {"pvpython": pvpython, "paraview": number, "pyinfo": info}

    def _build(
        self, transport: Transport, folder: str, env: dict[str, Any], pvpython: str
    ) -> str:
        script = build_script(folder, env["pyinfo"], pvpython)
        done = transport.run(script, timeout=900)
        if parse_markers(done.stdout).get("BUILD") == "ok":
            return "node"
        failures = [f"{transport.where}: {_tail(done)}"]
        jump = getattr(transport, "jump_host", lambda: None)()
        if jump:
            from .config import Target
            from .transport import SshTransport

            helper = SshTransport(Target(address=jump), self.settings.connect_timeout)
            try:
                helper.open()
                done = helper.run(
                    build_script(folder, env["pyinfo"], None), timeout=900
                )
                if parse_markers(done.stdout).get("BUILD") == "ok":
                    check = transport.run(f"test -d {shlex.quote(folder)}/site/trame")
                    if check.code == 0:
                        return f"jump host {jump}"
                    failures.append(
                        f"{jump} built it, but {folder} is not shared with "
                        f"{transport.where}"
                    )
                else:
                    failures.append(f"{jump}: {_tail(done)}")
            except ParaViewError as exc:
                failures.append(f"{jump}: {exc.message}")
            finally:
                helper.close()
        pip_plats, _ = platform_tags(env["pyinfo"])
        packages = " ".join(config.NODE_PACKAGES)
        raise ParaViewError(
            "no_package_index",
            "Could not install trame and trame-vtk for the ParaView "
            f"{transport.where}: neither it"
            + (f" nor the jump host {jump}" if jump else "")
            + " could reach the Python package index. ("
            + "; ".join(failures)
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
        transport.run(
            f"cat > {shlex.quote(folder)}/env.json", stdin=payload, timeout=60
        )

    def _upload(self, transport: Transport, folder: str, bundle: bytes) -> None:
        f = shlex.quote(folder)
        done = transport.run(
            f"rm -rf {f}/code.new && mkdir -p {f}/code.new && "
            f"tar -xzf - -C {f}/code.new && rm -rf {f}/code && mv {f}/code.new {f}/code",
            stdin=bundle,
            timeout=120,
        )
        if done.code != 0:
            raise ParaViewError(
                "upload_failed",
                f"Could not copy the node half {transport.where}: {_tail(done)}",
            )

    def _stop_node(
        self, transport: Transport, folder: str, node: dict[str, Any]
    ) -> None:
        pid = int(node.get("pid") or 0)
        f = shlex.quote(folder)
        script = f"rm -f {f}/node.json\n"
        if pid > 0:
            script += f"kill {pid} 2>/dev/null; sleep 1; kill -9 {pid} 2>/dev/null\n"
        transport.run(script + "exit 0", timeout=60)

    def _start_node(
        self, transport: Transport, folder: str, pvpython: str, code_hash: str
    ) -> dict[str, Any]:
        settings = self.settings
        f = shlex.quote(folder)
        args = [
            "--dr",
            "--force-offscreen-rendering",
            f"{folder}/code/run_node.py",
            "--root",
            folder,
            "--idle-seconds",
            str(settings.idle_seconds),
            "--pvserver-port",
            str(settings.pvserver_port),
            "--code-hash",
            code_hash,
        ]
        if settings.threshold_mb is not None:
            args += ["--threshold-mb", str(settings.threshold_mb)]
        command = " ".join(shlex.quote(a) for a in [pvpython, *args])
        transport.run(
            f"rm -f {f}/node.json\ncd {f}\n"
            f"if command -v setsid >/dev/null 2>&1; then S=setsid; else S=; fi\n"
            f"nohup $S {command} > {f}/node.log 2>&1 < /dev/null &\n"
            "exit 0",
            timeout=60,
        )
        deadline = time.monotonic() + settings.start_timeout
        while time.monotonic() < deadline:
            time.sleep(1.0)
            done = transport.run(
                f"cat {f}/node.json 2>/dev/null; echo; "
                f"grep -c 'Traceback\\|another node half' {f}/node.log 2>/dev/null || true",
                timeout=60,
            )
            lines = [line for line in done.stdout.splitlines() if line.strip()]
            if lines and lines[0].startswith("{"):
                node: dict[str, Any] = json.loads(lines[0])
                if node.get("code_hash") == code_hash:
                    return node
            if lines and lines[-1].isdigit() and int(lines[-1]) > 0:
                log = transport.run(f"tail -n 25 {f}/node.log", timeout=60)
                raise ParaViewError(
                    "node_failed",
                    f"The ParaView node half failed to start {transport.where}: "
                    f"{log.stdout.strip()[-1500:]}",
                )
        raise ParaViewError(
            "node_start_timeout",
            f"The ParaView node half did not start {transport.where} within "
            f"{settings.start_timeout:g} s. Its log is {folder}/node.log.",
        )

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
        try:
            client.health()
        except LinkError:
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
