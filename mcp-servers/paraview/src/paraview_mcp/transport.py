"""How the front reaches the machine where ParaView runs.

``LocalTransport`` runs shell scripts on this machine. ``SshTransport`` holds
one system ``ssh`` connection (a control master, with the person's own keys,
agent and ``~/.ssh/config``) and runs every script and the one port forward
over it. It never asks for a password: ``BatchMode=yes`` makes a host that
wants interactive sign-in fail at once.
"""

from __future__ import annotations

import os
import shlex
import shutil
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Target
from .errors import ParaViewError


@dataclass
class Completed:
    """The outcome of one script."""

    code: int
    stdout: str
    stderr: str


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

    @property
    def where(self) -> str:
        return "on this machine"

    def open(self) -> None:
        """Nothing to open locally."""

    def run(
        self, script: str, stdin: bytes | None = None, timeout: float = 600.0
    ) -> Completed:
        full = with_prepare(self.target.prepare, script)
        try:
            done = subprocess.run(
                ["bash", "-c", full],
                input=stdin,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ParaViewError(
                "timed_out", f"A step on this machine took longer than {timeout:g} s."
            ) from exc
        return Completed(
            done.returncode,
            done.stdout.decode("utf-8", "replace"),
            done.stderr.decode("utf-8", "replace"),
        )

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


class SshTransport:
    """One control-master ``ssh`` to the ParaView host; scripts reuse it."""

    remote = True

    def __init__(self, target: Target, connect_timeout: int = 15) -> None:
        if target.address is None:
            raise ValueError("an SSH transport needs an address")
        self.target = target
        self.host: str = target.address
        self.connect_timeout = connect_timeout
        self.ssh = shutil.which("ssh")
        self.folder = Path(tempfile.mkdtemp(prefix="clio-pv-"))
        os.chmod(self.folder, 0o700)
        self.socket = self.folder / "cm"
        self.log = self.folder / "master.log"
        self.master: subprocess.Popen[bytes] | None = None
        self.forwards: list[tuple[int, int]] = []

    @property
    def where(self) -> str:
        return self.target.describe()

    def _base(self) -> list[str]:
        if self.ssh is None:
            raise ParaViewError(
                "no_ssh", "The ssh command is not installed on this machine."
            )
        return [
            self.ssh,
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={self.connect_timeout}",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            f"ControlPath={self.socket}",
        ]

    def config_jump(self) -> str | None:
        """The ProxyJump ``~/.ssh/config`` gives this host, if any."""
        done = subprocess.run(
            [*self._base(), "-G", self.host],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
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
        jump = (
            [] if self.config_jump() or not self.target.via else ["-J", self.target.via]
        )
        command = [
            *self._base(),
            "-o",
            "ControlMaster=yes",
            "-o",
            "ControlPersist=no",
            "-o",
            "ExitOnForwardFailure=yes",
            "-N",
            *jump,
            self.host,
        ]
        with open(self.log, "wb") as log:
            self.master = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=log,
                start_new_session=True,
            )
        deadline = time.monotonic() + 2 * self.connect_timeout + 10
        while time.monotonic() < deadline:
            if self.master.poll() is not None:
                stderr = self.log.read_text("utf-8", "replace")
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

    def _check(self) -> bool:
        done = subprocess.run(
            [*self._base(), "-O", "check", self.host],
            capture_output=True,
            timeout=15,
            check=False,
        )
        return done.returncode == 0

    def run(
        self, script: str, stdin: bytes | None = None, timeout: float = 600.0
    ) -> Completed:
        full = with_prepare(self.target.prepare, script)
        command = [
            *self._base(),
            "-o",
            "ControlMaster=no",
            self.host,
            "bash -lc " + shlex.quote(full),
        ]
        try:
            done = subprocess.run(
                command,
                input=stdin,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ParaViewError(
                "timed_out", f"A step on {self.host} took longer than {timeout:g} s."
            ) from exc
        return Completed(
            done.returncode,
            done.stdout.decode("utf-8", "replace"),
            done.stderr.decode("utf-8", "replace"),
        )

    def forward(self, remote_port: int, local_port: int | None = None) -> int:
        """Forward one loopback port here to the node half's loopback port."""
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
        )
        if done.returncode != 0:
            raise ParaViewError(
                "forward_failed",
                f"Could not forward a local port to {self.host}: {done.stderr.strip()}",
            )
        self.forwards.append((port, remote_port))
        return port

    def alive(self) -> bool:
        return self.master is not None and self.master.poll() is None and self._check()

    def processes(self) -> list[int]:
        return [self.master.pid] if self.master is not None else []

    def close(self) -> None:
        if self.master is not None and self.master.poll() is None:
            subprocess.run(
                [*self._base(), "-O", "exit", self.host],
                capture_output=True,
                timeout=15,
                check=False,
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
