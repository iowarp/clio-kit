"""Run the ParaView remote-attach scenarios against the cluster fixture.

Each scenario starts the front as an MCP host would (stdio, a fresh process)
with the connector's environment, calls its tools through a real MCP client,
and checks what happened on the cluster with ``cluster.sh``. Run from
``mcp-servers/paraview`` with the cluster up::

    cluster/cluster.sh up
    uv run python tests/remote/aq_cluster.py --out DIR [43 38 39 44 41 46 47 42 50 40]

Scenario ids are the plan's AQ numbers. Results go to DIR/results.json and are
printed as they finish. Exit status is the number of failed scenarios.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

HERE = Path(__file__).resolve().parent
SERVER = HERE.parents[1]
CLUSTER = SERVER / "cluster" / "cluster.sh"
FIX: dict[str, str] = {}


def cluster(*args: str, timeout: float = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(CLUSTER), *args], capture_output=True, text=True, timeout=timeout, check=False
    )


def load_fixture_env() -> None:
    out = cluster("env").stdout
    for line in out.splitlines():
        if line.startswith("export "):
            key, _, value = line[7:].partition("=")
            FIX[key] = value.strip('"')


def front_env(address: str | None, via: str | None, **extra: str) -> dict[str, str]:
    env = dict(os.environ)
    for key in ("PARAVIEW_ADDRESS", "PARAVIEW_VIA", "PARAVIEW_PREPARE"):
        env.pop(key, None)
    if address:
        env["PARAVIEW_ADDRESS"] = address
    if via:
        env["PARAVIEW_VIA"] = via
    env["PARAVIEW_PREPARE"] = FIX["CLIO_PV_CLUSTER_PREPARE"]
    env.update(extra)
    return env


def text(result: Any) -> str:
    return "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")


class Front:
    """One front process and its MCP client."""

    def __init__(self, env: dict[str, str]) -> None:
        transport = StdioTransport(
            command="uv", args=["run", "--directory", str(SERVER), "paraview-mcp"], env=env
        )
        self.client = Client(transport, timeout=1200)

    async def __aenter__(self) -> Front:
        await self.client.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.client.__aexit__(*exc)

    async def call(self, name: str, args: dict[str, Any] | None = None) -> tuple[Any, float]:
        started = time.monotonic()
        result = await self.client.call_tool(name, args or {}, raise_on_error=False)
        return result, round(time.monotonic() - started, 2)


def node_halves() -> str:
    return cluster("exec", "node08", "pgrep", "-af", "run_node.py").stdout.strip()


def stop_node_halves() -> None:
    cluster("exec", "node08", "pkill", "-f", "run_node.py")
    deadline = time.monotonic() + 30
    while node_halves() and time.monotonic() < deadline:
        time.sleep(1)


def host_ssh() -> str:
    return subprocess.run(
        ["ps", "-C", "ssh", "-o", "pid,ppid,args", "--no-headers"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


ALT = ("CLIO_PV_CLUSTER_ALT_ADDRESS", "CLIO_PV_CLUSTER_ALT_VIA")


def remote_env(**extra: str) -> dict[str, str]:
    return front_env(FIX[ALT[0]], FIX[ALT[1]], **extra)


async def aq43() -> dict[str, Any]:
    async with Front(front_env(FIX["CLIO_PV_CLUSTER_LOCKED"], None)) as front:
        result, seconds = await front.call("open_view", {"path": "/data/x.vti"})
    leftover = [line for line in host_ssh().splitlines() if "locked" in line]
    ok = result.is_error and "interactive sign-in" in text(result) and seconds < 30
    return {"pass": ok and not leftover, "seconds": seconds, "text": text(result),
            "ssh_left": leftover}


async def aq38(state: dict[str, Any]) -> dict[str, Any]:
    """First attach to node08 through login; prepares the folder from login."""
    stop_node_halves()
    cluster("exec", "node08", "rm", "-rf", f"{FIX['CLIO_PV_CLUSTER_HOME']}/.clio")
    async with Front(remote_env()) as front:
        opened, open_s = await front.call(
            "open_view", {"path": FIX["CLIO_PV_CLUSTER_DATASET"], "new": True}
        )
        added, _ = await front.call(
            "create_filter",
            {
                "filter": "Contour",
                "properties": {
                    "ContourBy": ["POINTS", FIX["CLIO_PV_CLUSTER_DATASET_FIELD"]],
                    "Isosurfaces": [157.0],
                },
            },
        )
        state_result, _ = await front.call("scene_state")
        structured = state_result.structured_content or {}
        attachment = structured.get("attachment", {})
        processes = cluster("scenarios", "39")
        port_check = cluster("scenarios", "50")
        ssh_now = host_ssh()
    state["scene"] = (opened.structured_content or {}).get("scene")
    service = (opened.meta or {}).get("ai.iowarp.clio/service", {})
    ok = (
        not opened.is_error
        and not added.is_error
        and attachment.get("built_this_attach")
        and "jump host" in str(attachment.get("environment_built_from"))
        and service.get("port")
    )
    return {
        "pass": bool(ok),
        "open_view_seconds": open_s,
        "attachment": attachment,
        "open_view_text": text(opened),
        "token_in_text": bool(service.get("token")) and service["token"] in text(opened),
        "aq39": {"pass": processes.returncode == 0, "output": processes.stdout[-3000:],
                 "host_ssh": ssh_now},
        "aq50": {"pass": port_check.returncode == 0, "output": port_check.stdout[-2000:]},
    }


async def aq44(state: dict[str, Any]) -> dict[str, Any]:
    async with Front(remote_env()) as front:
        result, seconds = await front.call("scene_state")
    attachment = (result.structured_content or {}).get("attachment", {})
    ok = not result.is_error and not attachment.get("built_this_attach") and seconds < 60
    return {"pass": ok, "seconds": seconds, "attachment": attachment}


def _master_pid() -> int | None:
    for line in host_ssh().splitlines():
        pid, _ppid, args = line.split(None, 2)
        if "ControlMaster=yes" in args and "clio-pv-" in args:
            return int(pid)
    return None


async def aq41(state: dict[str, Any]) -> dict[str, Any]:
    async with Front(remote_env()) as front:
        await front.call("open_view", {"scene": state["scene"]})
        pid = _master_pid()
        if pid:
            os.kill(pid, 9)
        time.sleep(1)
        added, seconds = await front.call(
            "create_filter", {"filter": "Slice", "input": "o1", "label": "slice-after-drop"}
        )
        after, _ = await front.call("scene_state")
    labels = [o.get("label") for o in (after.structured_content or {}).get("scene", {}).get(
        "objects", [])]
    ok = pid is not None and not added.is_error and "slice-after-drop" in labels and len(labels) >= 3
    return {"pass": ok, "killed_ssh": pid, "create_filter_seconds": seconds, "labels": labels}


async def aq46(state: dict[str, Any]) -> dict[str, Any]:
    cluster("grow", "stop")
    cluster("exec", "node08", "rm", "-rf", f"{FIX['CLIO_PV_CLUSTER_HOME']}/data/growing")
    cluster("grow", "start", "--interval", "6", "--steps", "40")
    time.sleep(14)
    record: dict[str, Any] = {}
    async with Front(remote_env()) as front:
        opened, _ = await front.call(
            "open_view", {"path": FIX["CLIO_PV_CLUSTER_GROWING"], "growing": True, "new": True}
        )
        first = (opened.structured_content or {}).get("summary", {}).get("time", {})
        time.sleep(15)
        later, _ = await front.call("scene_state")
        followed = later.structured_content["scene"]["time"]
        await front.call("set_timestep", {"index": 0})
        time.sleep(15)
        parked, _ = await front.call("scene_state")
        parked_time = parked.structured_content["scene"]["time"]
        record["parked_text"] = text(parked).splitlines()[-4:]
        await front.call("set_timestep", {"latest": True})
    cluster("grow", "stop")
    ok = (
        followed["steps"] > first.get("steps", 0)
        and followed["follow_latest"]
        and followed["index"] == followed["steps"] - 1
        and parked_time["index"] == 0
        and parked_time["newer_steps"] > 0
    )
    record.update(pass_=ok, first=first, followed=followed, parked=parked_time)
    record["pass"] = record.pop("pass_")
    return record


async def aq47(state: dict[str, Any]) -> dict[str, Any]:
    """The saved call restores the scene: once live, once after the node half restarts."""
    async with Front(remote_env()) as front:
        await front.call("open_view", {"scene": state["scene"]})
        await front.call("set_representation", {"object": "o2", "representation": "Wireframe"})
    async with Front(remote_env()) as front:
        live, _ = await front.call("open_view", {"path": FIX["CLIO_PV_CLUSTER_DATASET"]})
    stop_node_halves()
    async with Front(remote_env()) as front:
        restored, seconds = await front.call(
            "open_view", {"path": FIX["CLIO_PV_CLUSTER_DATASET"]}
        )
    summary = (restored.structured_content or {}).get("summary", {})
    displays = {o["id"]: o.get("display", {}).get("representation") for o in summary.get(
        "objects", [])}
    ok = summary.get("restored") and displays.get("o2") == "Wireframe" and len(displays) >= 3
    return {"pass": bool(ok), "live_text": text(live).splitlines()[:6],
            "restored_seconds": seconds, "displays": displays}


async def aq42(state: dict[str, Any]) -> dict[str, Any]:
    stop_node_halves()
    async with Front(remote_env(PARAVIEW_IDLE_SECONDS="20")) as front:
        await front.call("open_view", {"scene": state["scene"]})
    started = time.monotonic()
    while node_halves() and time.monotonic() - started < 180:
        time.sleep(3)
    gone_after = round(time.monotonic() - started, 1)
    home = FIX["CLIO_PV_CLUSTER_HOME"]
    listing = cluster(
        "exec", "node08", "ls", "-l", f"{home}/.clio/paraview/3.0.0/scenes/{state['scene']}"
    ).stdout
    log = cluster("exec", "node08", "tail", "-n", "3", f"{home}/.clio/paraview/3.0.0/node.log")
    ok = not node_halves() and "manifest.json" in listing
    return {"pass": ok, "gone_after_s": gone_after, "scene_folder": listing,
            "log": log.stdout}


async def aq40(state: dict[str, Any]) -> dict[str, Any]:
    stop_node_halves()
    cluster("variant", "pvserver", timeout=600)
    async with Front(remote_env()) as front:
        opened, seconds = await front.call("open_view")
    cluster("variant", "plain", timeout=600)
    ok = not opened.is_error and "session" in text(opened)
    return {"pass": ok, "seconds": seconds, "text": text(opened)}


SCENARIOS: dict[str, Callable[..., Any]] = {
    "43": aq43, "38": aq38, "44": aq44, "41": aq41, "46": aq46, "47": aq47,
    "42": aq42, "40": aq40,
}


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("ids", nargs="*", default=["43", "38", "44", "41", "46", "47", "42"])
    opts = parser.parse_args()
    opts.out.mkdir(parents=True, exist_ok=True)
    load_fixture_env()
    state_file = opts.out / "state.json"
    state: dict[str, Any] = json.loads(state_file.read_text()) if state_file.exists() else {}
    results_file = opts.out / "results.json"
    results: dict[str, Any] = (
        json.loads(results_file.read_text()) if results_file.exists() else {}
    )
    failed = 0
    for scenario in opts.ids:
        function = SCENARIOS[scenario]
        started = time.monotonic()
        try:
            record = await (function() if scenario == "43" else function(state))
        except Exception as exc:  # noqa: BLE001 - record and go on
            record = {"pass": False, "error": f"{type(exc).__name__}: {exc}"}
        record["elapsed_s"] = round(time.monotonic() - started, 1)
        results[f"AQ{scenario}"] = record
        failed += 0 if record.get("pass") else 1
        print(json.dumps({f"AQ{scenario}": record}, indent=2)[:4000], flush=True)
        results_file.write_text(json.dumps(results, indent=2))
        state_file.write_text(json.dumps(state))
    return failed


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
