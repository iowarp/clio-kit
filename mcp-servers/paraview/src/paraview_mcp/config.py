"""Where ParaView runs, as the connector's environment gives it."""

from __future__ import annotations

import os
from dataclasses import dataclass

SERVER_VERSION = "3.3.0"
DEFAULT_IDLE_SECONDS = 1800.0
DEFAULT_PVSERVER_PORT = 11111
# Pinned to the versions this server was qualified with.
NODE_PACKAGES = ("trame==4.0.0", "trame-vtk==2.11.17", "trame-vuetify==3.3.0")


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    # An unfilled connector placeholder means the need was left empty.
    if not value or (value.startswith("${") and value.endswith("}")):
        return None
    return value


@dataclass(frozen=True)
class Target:
    """One place ParaView can run: this machine, or an SSH host."""

    address: str | None = None
    via: str | None = None
    prepare: str | None = None

    @property
    def remote(self) -> bool:
        return self.address is not None

    def describe(self) -> str:
        if self.address is None:
            return "on this machine"
        hop = f" through {self.via}" if self.via else ""
        return f"on {self.address}{hop}"


@dataclass(frozen=True)
class Settings:
    """The connector values plus the tunables tests shorten."""

    target: Target
    idle_seconds: float = DEFAULT_IDLE_SECONDS
    threshold_mb: float | None = None
    pvserver_port: int = DEFAULT_PVSERVER_PORT
    #: host:port of a running pvserver to join; the data is read on its machine.
    pvserver: str | None = None
    #: The ParaView version to fetch for this machine when none of it is installed.
    version: str | None = None
    connect_timeout: int = 15
    #: How long a starting node half may show no sign of progress (no new log
    #: output, no record) before it is taken for stuck. There is no overall limit:
    #: a start that keeps progressing is waited for, and the wait can be cancelled.
    start_stall: float = 120.0
    #: How long a setup step (a probe, a rendering check, an install) may print
    #: nothing before it is taken for stuck. There is no overall limit either.
    setup_stall: float = 600.0


def _float(name: str, default: float | None) -> float | None:
    raw = _clean(os.environ.get(name))
    return float(raw) if raw is not None else default


def from_environment() -> Settings:
    """Read the connector's values from the environment CLIO starts us with."""
    target = Target(
        address=_clean(os.environ.get("PARAVIEW_ADDRESS")),
        via=_clean(os.environ.get("PARAVIEW_VIA")),
        prepare=_clean(os.environ.get("PARAVIEW_PREPARE")),
    )
    idle = _float("PARAVIEW_IDLE_SECONDS", DEFAULT_IDLE_SECONDS)
    port = _float("PARAVIEW_PVSERVER_PORT", DEFAULT_PVSERVER_PORT)
    return Settings(
        target=target,
        idle_seconds=float(idle if idle is not None else DEFAULT_IDLE_SECONDS),
        threshold_mb=_float("PARAVIEW_REMOTE_RENDER_THRESHOLD_MB", None),
        pvserver_port=int(port if port is not None else DEFAULT_PVSERVER_PORT),
        pvserver=_clean(os.environ.get("PARAVIEW_PVSERVER")),
        version=_clean(os.environ.get("PARAVIEW_VERSION")),
        start_stall=float(
            _float("PARAVIEW_START_STALL", _float("PARAVIEW_START_TIMEOUT", 120.0))
            or 120.0
        ),
        setup_stall=float(_float("PARAVIEW_SETUP_STALL", 600.0) or 600.0),
    )
