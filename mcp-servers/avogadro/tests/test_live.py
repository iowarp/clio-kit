"""The real stack: Xvfb, Avogadro, x11vnc and the loopback service. Skipped without them."""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from fastmcp import Client

from avogadro_mcp import server
from avogadro_mcp.server import mcp

from .conftest import REAL
from .test_offline import CJSON, COMMAND

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not REAL, reason="needs avogadro2, Xvfb and x11vnc"),
]


@pytest.mark.timeout(240) if hasattr(pytest.mark, "timeout") else (lambda f: f)
async def test_open_stream_snapshot_and_read_back(water, out):
    async with Client(mcp) as client:
        opened = await client.call_tool("open_in_avogadro", {"path": str(water)})
        meta = opened.meta or {}
        s = server.session()
        try:
            assert s.running() and s.opened == [str(water)]
            service = (opened.meta or {}).get("ai.iowarp.clio/service") or {
                "port": s.service_port,
                "token": s.token,
            }
            base = f"http://127.0.0.1:{service['port']}"
            assert httpx.get(base + "/").status_code == 401
            auth = {"Authorization": f"Bearer {service['token']}"}
            page = httpx.get(base + "/index.html", headers=auth)
            assert page.status_code == 200 and "novnc/core/rfb.js" in page.text
            assert httpx.get(base + "/novnc/core/rfb.js", headers=auth).status_code == 200
            assert httpx.get(base + "/novnc/../server.py", headers=auth).status_code == 404

            import websockets

            async with websockets.connect(
                f"ws://127.0.0.1:{service['port']}/websockify",
                additional_headers=auth,
                subprotocols=["binary"],
            ) as ws:
                hello = await asyncio.wait_for(ws.recv(), 10)
            assert bytes(hello).startswith(b"RFB 003.")  # the VNC server, through the bridge

            shot = await client.call_tool("avogadro_snapshot", {})
            png = base64.b64decode(shot.content[1].data)
            assert png[:8] == b"\x89PNG\r\n\x1a\n"

            # The person's "Send to CLIO", run as Avogadro runs the command.
            env = {"CLIO_AVOGADRO_HANDOFF": str(s.handoff), "PATH": "/usr/bin:/bin"}
            subprocess.run(
                [sys.executable, str(COMMAND), "--run-command"],
                input=json.dumps({"cjson": CJSON}),
                text=True,
                env=env,
                check=True,
            )
            assert s.state()["sent"]["selected"] == [1, 2]
            got = await client.call_tool("read_avogadro", {})
            data = got.structured_content
            assert data["formula"] == "H2O" and data["selected"] == [1, 2]
            assert Path(data["structure"]["path"]).parent == out / "chemistry/avogadro"
        finally:
            await client.call_tool("close_avogadro", {})
    assert not any(p.poll() is None for p in s.procs.values())
    assert not s.root.exists()
    _ = meta
