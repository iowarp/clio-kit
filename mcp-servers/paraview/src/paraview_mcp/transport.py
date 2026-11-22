"""How the front reaches the machine where ParaView runs.

``LocalTransport`` runs shell scripts on this machine. ``SshTransport`` holds
one system ``ssh`` connection (a control master, with the person's own keys,
agent and ``~/.ssh/config``) and runs every script and the one port forward
over it. Where ssh has no control master (OpenSSH for Windows), each script is
its own ``ssh`` and the forward is a dedicated ``ssh -N -L``. It never asks for
a password: ``BatchMode=yes`` makes a host that wants interactive sign-in fail
at once.
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from .config import Target
from .errors import ParaViewError


@dataclass
class Completed:
    """The outcome of one script."""

    code: int
    stdout: str
    stderr: str


@dataclass
class Watch:
    """How a step is watched: there is no limit on its total time.

    A step is stopped only when ``cancelled`` says the person cancelled, or when it
    has printed nothing for ``stall`` seconds. While it runs, ``progress`` is told
    how long it has run and the last line it printed.
    """

    stall: float = 600.0
    cancelled: Callable[[], bool] = field(default=lambda: False)
    progress: Callable[[str], None] | None = None


def _end(process: subprocess.Popen[bytes]) -> None:
    """Stop ``process`` and whatever it started."""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
                check=False,
            )
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        process.kill()
    try:
        process.wait(5)
    except subprocess.TimeoutExpired:
        process.kill()


def watched(
    command: list[str],
    watch: Watch,
    label: str,
    stdin: bytes | None = None,
    cwd: str | Path | None = None,
    every: float = 0.5,
) -> Completed:
    """Run ``command`` for as long as it keeps printing, as ``watch`` says.

    Raises ``cancelled`` when the person cancels and ``setup_stalled`` after
    ``watch.stall`` seconds with no output; either way the process is stopped.
    """
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        start_new_session=os.name != "nt",
        creationflags=0x08000000 if os.name == "nt" else 0,
    )
    out: list[bytes] = []
    err: list[bytes] = []
    heard = [time.monotonic()]
    last = [""]

    def pump(stream: IO[bytes] | None, into: list[bytes]) -> None:
        if stream is None:
            return
        for raw in iter(stream.readline, b""):
            into.append(raw)
            heard[0] = time.monotonic()
            text = raw.decode("utf-8", "replace").strip()
            if text:
                last[0] = text[:300]
        stream.close()

    def feed() -> None:
        try:
            assert process.stdin is not None
            process.stdin.write(stdin or b"")
            process.stdin.close()
        except OSError:
            pass

    threads = [
        threading.Thread(target=pump, args=(process.stdout, out), daemon=True),
        threading.Thread(target=pump, args=(process.stderr, err), daemon=True),
    ]
    if stdin is not None:
        threads.append(threading.Thread(target=feed, daemon=True))
    for thread in threads:
        thread.start()
    began = told = time.monotonic()
    said = ""
    while True:
        try:
            process.wait(every)
            break
        except subprocess.TimeoutExpired:
            pass
        now = time.monotonic()
        if watch.cancelled():
            _end(process)
            raise ParaViewError(
                "cancelled", f"{label} was cancelled after {now - began:.0f} s."
            )
        quiet = now - heard[0]
        if quiet >= watch.stall:
            _end(process)
            raise ParaViewError(
                "setup_stalled",
                f"{label} printed nothing for {quiet:.0f} s, so it was stopped. "
                f"It last said: {last[0] or '(nothing)'}",
            )
        if watch.progress is not None and (last[0] != said or now - told >= 5.0):
            said, told = last[0], now
            text = f"{label}, {now - began:.0f} s"
            if last[0]:
                text += f": {last[0]}"
            if quiet >= 30:
                text += f" (nothing new for {quiet:.0f} s)"
            watch.progress(text)
    for thread in threads:
        # A child that keeps the pipes open after the step ends is not waited on.
        thread.join(5)
    return Completed(
        process.returncode or 0,
        b"".join(out).decode("utf-8", "replace"),
        b"".join(err).decode("utf-8", "replace"),
    )


def free_port() -> int:
    """A loopback port nothing listens on right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def with_prepare(prepare: str | None, script: str) -> str:
    """Prefix ``script`` with the connector's prepare line."""
    if not prepare:
        return script
    return f"{prepare}\n{script}"


class LocalTransport:
    """ParaView on this machine: scripts run here and no port is forwarded."""

    remote = False

    def __init__(self, target: Target) -> None:
        self.target = target
        self.watch = Watch()

    @property
    def where(self) -> str:
        return "on this machine"

    def open(self) -> None:
        """Nothing to open locally."""

    def run(
        self, script: str, stdin: bytes | None = None, label: str = "A setup step"
    ) -> Completed:
        full = with_prepare(self.target.prepare, script)
        return watched(["bash", "-c", full], self.watch, f"{label} {self.where}", stdin)

    def forward(self, remote_port: int, local_port: int | None = None) -> int:
        return remote_port

    def alive(self) -> bool:
        return True

    def close(self) -> None:
        """Nothing to close locally."""

    def processes(self) -> list[int]:
        return []


def classify_ssh_failure(host: str, stderr: str) -> ParaViewError:
    """Turn ssh's stderr into a message that says what to do."""
    text = stderr.strip()
    lowered = text.lower()
    tail = text.splitlines()[-1] if text else "ssh exited without a message"
    interactive = "password" in lowered or "keyboard-interactive" in lowered
    if "permission denied" in lowered and not interactive:
        return ParaViewError(
            "key_rejected",
            f"{host} did not accept your SSH keys. Add your public key to "
            f"~/.ssh/authorized_keys on {host}, or load the right key into your "
            f"agent. ssh said: {tail}",
        )
    if interactive:
        return ParaViewError(
            "interactive_sign_in",
            f"{host} asks for interactive sign-in (a password or a one-time code), and "
            "the ParaView connector signs in only with your SSH keys or agent. Set up "
            f"key-based sign-in for {host}, or use a host that accepts your key. "
            f"ssh said: {tail}",
        )
    if "host key verification failed" in lowered:
        return ParaViewError(
            "host_key_unknown",
            f"{host}'s host key is not in your known_hosts. Connect once with ssh "
            f"yourself to check and accept it, then try again. ssh said: {tail}",
        )
    if "could not resolve hostname" in lowered:
        return ParaViewError(
            "unknown_host",
            f"The SSH host name {host!r} does not resolve. Check the connector's "
            f"'Where ParaView runs' value or your ~/.ssh/config. ssh said: {tail}",
        )
    if any(
        word in lowered
        for word in ("timed out", "connection refused", "no route", "unreachable")
    ):
        return ParaViewError(
            "unreachable", f"{host} cannot be reached over SSH. ssh said: {tail}"
        )
    return ParaViewError("ssh_failed", f"ssh to {host} failed: {tail}")


#: What ssh prints when this platform cannot hold a control master socket.
_NO_MULTIPLEX = ("getsockname failed", "not a socket", "controlpath too long")


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


class SshTransport:
    """One control-master ``ssh`` to the ParaView host; scripts reuse it.

    Without control master support each script opens its own connection and the
    port forward is its own ``ssh -N -L`` process.
    """

    remote = True

    def __init__(self, target: Target, connect_timeout: int = 15) -> None:
        if target.address is None:
            raise ValueError("an SSH transport needs an address")
        self.target = target
        self.host: str = target.address
        self.connect_timeout = connect_timeout
        self.watch = Watch()
        self.ssh = shutil.which("ssh")
        self.folder = Path(tempfile.mkdtemp(prefix="clio-pv-"))
        os.chmod(self.folder, 0o700)
        self.socket = self.folder / "cm"
        self.log = self.folder / "master.log"
        self.master: subprocess.Popen[bytes] | None = None
        self.forwards: list[tuple[int, int]] = []
        #: OpenSSH for Windows has no control master (no Unix sockets).
        self.multiplex = os.name != "nt"
        self.tunnels: list[subprocess.Popen[bytes]] = []
        self._jump: list[str] | None = None

    @property
    def where(self) -> str:
        return self.target.describe()

    def _base(self) -> list[str]:
        if self.ssh is None:
            raise ParaViewError(
                "no_ssh", "The ssh command is not installed on this machine."
            )
        base = [
            self.ssh,
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={self.connect_timeout}",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
        ]
        if self.multiplex:
            # Relative to the folder every ssh here runs in: a host's temporary
            # folder can be too deep for a Unix socket path (about 104 bytes).
            base += ["-o", f"ControlPath={self.socket.name}"]
        return base

    def _jump_args(self) -> list[str]:
        """``-J via`` unless ``~/.ssh/config`` already gives this host a jump."""
        if self._jump is None:
            self._jump = (
                []
                if self.config_jump() or not self.target.via
                else ["-J", self.target.via]
            )
        return self._jump

    def config_jump(self) -> str | None:
        """The ProxyJump ``~/.ssh/config`` gives this host, if any."""
        done = subprocess.run(
            [*self._base(), "-G", self.host],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            cwd=self.folder,
        )
        for line in done.stdout.splitlines():
            key, _, value = line.partition(" ")
            if key.lower() == "proxyjump" and value.strip().lower() != "none":
                return value.strip()
        return None

    def jump_host(self) -> str | None:
        """The jump host in use: the config's, else the connector's ``via``."""
        return self.config_jump() or self.target.via

    def open(self) -> None:
        if not self.multiplex:
            self._open_plain()
            return
        command = [
            *self._base(),
            "-o",
            "ControlMaster=yes",
            "-o",
            "ControlPersist=no",
            "-o",
            "ExitOnForwardFailure=yes",
            "-N",
            *self._jump_args(),
            self.host,
        ]
        with open(self.log, "wb") as log:
            self.master = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=log,
                start_new_session=True,
                cwd=self.folder,
            )
        deadline = time.monotonic() + 2 * self.connect_timeout + 10
        while time.monotonic() < deadline:
            if self.master.poll() is not None:
                stderr = self.log.read_text("utf-8", "replace")
                if any(mark in stderr.lower() for mark in _NO_MULTIPLEX):
                    self.master = None
                    self.multiplex = False
                    self._open_plain()
                    return
                self.close()
                raise classify_ssh_failure(self.host, stderr)
            if self.socket.exists() and self._check():
                return
            time.sleep(0.2)
        self.close()
        raise ParaViewError(
            "unreachable",
            f"ssh to {self.host} did not finish connecting in "
            f"{2 * self.connect_timeout + 10} s.",
        )

    def _open_plain(self) -> None:
        """Check the host signs in, since there is no master to hold open."""
        try:
            done = subprocess.run(
                [*self._base(), *self._jump_args(), self.host, "true"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=2 * self.connect_timeout + 10,
                check=False,
                cwd=self.folder,
            )
        except subprocess.TimeoutExpired as exc:
            self.close()
            raise ParaViewError(
                "unreachable",
                f"ssh to {self.host} did not finish connecting in "
                f"{2 * self.connect_timeout + 10} s.",
            ) from exc
        if done.returncode != 0:
            self.close()
            raise classify_ssh_failure(
                self.host, done.stderr.decode("utf-8", "replace")
            )

    def _check(self) -> bool:
        done = subprocess.run(
            [*self._base(), "-O", "check", self.host],
            capture_output=True,
            timeout=15,
            check=False,
            cwd=self.folder,
        )
        return done.returncode == 0

    def run(
        self, script: str, stdin: bytes | None = None, label: str = "A setup step"
    ) -> Completed:
        full = with_prepare(self.target.prepare, script)
        reuse = ["-o", "ControlMaster=no"] if self.multiplex else self._jump_args()
        command = [
            *self._base(),
            *reuse,
            self.host,
            "bash -lc " + shlex.quote(full),
        ]
        return watched(
            command, self.watch, f"{label} on {self.host}", stdin, self.folder
        )

    def forward(self, remote_port: int, local_port: int | None = None) -> int:
        """Forward one loopback port here to the node half's loopback port."""
        if not self.multiplex:
            return self._forward_plain(remote_port, local_port)
        for local, remote in list(self.forwards):
            subprocess.run(
                [
                    *self._base(),
                    "-O",
                    "cancel",
                    "-L",
                    f"127.0.0.1:{local}:127.0.0.1:{remote}",
                    self.host,
                ],
                capture_output=True,
                timeout=15,
                check=False,
                cwd=self.folder,
            )
            self.forwards.remove((local, remote))
        port = local_port or free_port()
        done = subprocess.run(
            [
                *self._base(),
                "-O",
                "forward",
                "-L",
                f"127.0.0.1:{port}:127.0.0.1:{remote_port}",
                self.host,
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            cwd=self.folder,
        )
        if done.returncode != 0:
            raise ParaViewError(
                "forward_failed",
                f"Could not forward a local port to {self.host}: {done.stderr.strip()}",
            )
        self.forwards.append((port, remote_port))
        return port

    def _stop_tunnels(self) -> None:
        for tunnel in self.tunnels:
            if tunnel.poll() is None:
                tunnel.kill()
                try:
                    tunnel.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        self.tunnels.clear()
        self.forwards.clear()

    def _forward_plain(self, remote_port: int, local_port: int | None) -> int:
        """A dedicated ``ssh -N -L`` that lives as long as this transport."""
        self._stop_tunnels()
        port = local_port or free_port()
        log_path = self.folder / f"forward-{port}.log"
        command = [
            *self._base(),
            "-o",
            "ExitOnForwardFailure=yes",
            "-N",
            "-L",
            f"127.0.0.1:{port}:127.0.0.1:{remote_port}",
            *self._jump_args(),
            self.host,
        ]
        with open(log_path, "wb") as log:
            tunnel = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=log,
                start_new_session=True,
                cwd=self.folder,
            )
        self.tunnels.append(tunnel)
        deadline = time.monotonic() + 2 * self.connect_timeout + 10
        while time.monotonic() < deadline:
            if tunnel.poll() is not None:
                stderr = log_path.read_text("utf-8", "replace").strip()
                self._stop_tunnels()
                raise ParaViewError(
                    "forward_failed",
                    f"Could not forward a local port to {self.host}: {stderr}",
                )
            if _port_open(port):
                self.forwards.append((port, remote_port))
                return port
            time.sleep(0.2)
        self._stop_tunnels()
        raise ParaViewError(
            "forward_failed",
            f"The port forward to {self.host} did not open in "
            f"{2 * self.connect_timeout + 10} s.",
        )

    def alive(self) -> bool:
        if not self.multiplex:
            return all(tunnel.poll() is None for tunnel in self.tunnels)
        return self.master is not None and self.master.poll() is None and self._check()

    def processes(self) -> list[int]:
        pids = [tunnel.pid for tunnel in self.tunnels]
        return ([self.master.pid] if self.master is not None else []) + pids

    def close(self) -> None:
        self._stop_tunnels()
        if self.master is not None and self.master.poll() is None:
            subprocess.run(
                [*self._base(), "-O", "exit", self.host],
                capture_output=True,
                timeout=15,
                check=False,
                cwd=self.folder,
            )
            try:
                self.master.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.master.kill()
        self.forwards.clear()
        shutil.rmtree(self.folder, ignore_errors=True)


Transport = LocalTransport | SshTransport


def transport_for(target: Target, connect_timeout: int = 15) -> Transport:
    """The transport for ``target``: SSH when it names an address."""
    if target.address is None:
        return LocalTransport(target)
    return SshTransport(target, connect_timeout)
