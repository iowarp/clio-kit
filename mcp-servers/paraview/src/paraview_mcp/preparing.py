"""Preparing and starting a node half on the machine where ParaView runs: find its
pvpython, ask it its version, check offscreen rendering, install trame beside it,
copy the code over, and start, stop or read why it went."""

from __future__ import annotations

import json
import shlex
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import client, config, native
from .config import Settings
from .errors import ParaViewError
from .node_setup import (
    PYINFO,
    build_script,
    gl_probe_script,
    parse_markers,
    platform_tags,
)
from .starting import StartLook, gone_error, wait_for_start
from .transport import Completed, Transport, Watch

#: Installer output that means the folder refused the write, not the index.
_WRITE_DENIED = (
    "Access is denied",
    "os error 5",
    "Permission denied",
    "PermissionError",
)

#: How a front hands a node half started in place of a gone one the token it had
#: (paraview_mcp.node.main reads it), so views a host serves keep their access.
TOKEN_ENV = "CLIO_PARAVIEW_NODE_TOKEN"


class NodePreparing:
    """Mixed into the attacher: the steps that prepare and start a node half."""

    settings: Settings
    #: ParaView runs on this Windows machine: steps run in Python, not a shell.
    _native: bool
    _reuse: dict[str, Any] | None
    cancel: threading.Event
    progress: Callable[[str], None] | None
    scene: str | None

    # Provided by the attacher.
    _watch: Callable[[], Watch]

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
        reuse = self._reuse or {}
        token = str(reuse.get("token") or "")
        if reuse.get("port"):
            args += ["--port", str(reuse["port"])]
        if self._native:
            return self._start_native(
                folder,
                [pvpython, *args],
                code_hash,
                slot,
                {TOKEN_ENV: token} if token else None,
            )
        command = " ".join(shlex.quote(a) for a in [pvpython, *args])
        log = f"{f}/{slot}.log"
        # The token goes over the connection's input, never on a command line.
        handover = (
            f"IFS= read -r {TOKEN_ENV} || true\nexport {TOKEN_ENV}\n" if token else ""
        )
        begun = transport.run(
            f"rm -f {f}/{slot}.json {f}/{slot}.lost.json\ncd {f}\n"
            f"[ -f {log} ] && mv -f {log} {f}/{slot}.prev.log\n"
            f"{handover}"
            f"if command -v setsid >/dev/null 2>&1; then S=setsid; else S=; fi\n"
            f"nohup $S {command} > {log} 2>&1 < /dev/null &\n"
            'echo "CLIO_PID=$!"\n'
            "exit 0",
            stdin=f"{token}\n".encode() if token else None,
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
        self,
        folder: str,
        command: list[str],
        code_hash: str,
        slot: str,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        process = native.start(folder, command, slot, env)
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


def _tail(done: Completed) -> str:
    text = (done.stderr.strip() or done.stdout.strip()).splitlines()
    return " ".join(text[-3:]) if text else f"exit status {done.code}"
