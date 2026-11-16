"""Install lookup, checksum refusal, the handoff command and CJSON reading, offline."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from avogadro_mcp import cjson, install
from avogadro_mcp.server import mcp

COMMAND = Path(__file__).resolve().parents[1] / "src/avogadro_mcp/handoff_command.py"
CJSON = {
    "chemicalJson": 1,
    "atoms": {
        "elements": {"number": [8, 1, 1]},
        "coords": {"3d": [0, 0, 0.117, 0, 0.757, -0.469, 0, -0.757, -0.469]},
        "selected": [False, True, True],
    },
}


def test_cjson_gives_formula_selection_and_xyz(tmp_path):
    mol = cjson.read(CJSON)
    assert mol["formula"] == "H2O" and mol["selected"] == [1, 2]
    cjson.write_xyz(tmp_path / "w.xyz", mol, "c")
    from ase.io import read

    assert read(str(tmp_path / "w.xyz")).get_chemical_formula() == "H2O"


def test_the_handoff_command_writes_and_echoes_the_molecule(tmp_path):
    env = {**os.environ, "CLIO_AVOGADRO_HANDOFF": str(tmp_path)}
    menu = subprocess.run(
        [sys.executable, str(COMMAND), "--menu-path"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert menu.stdout.strip() == "&Extensions|CLIO"
    run = subprocess.run(
        [sys.executable, str(COMMAND), "--run-command", "--lang", "en"],
        input=json.dumps({"cjson": CJSON}),
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert json.loads(run.stdout)["cjson"] == CJSON  # else Avogadro empties the molecule
    (sent,) = tmp_path.glob("sent-*.cjson")
    assert json.loads(sent.read_text()) == CJSON


def test_lookup_order(monkeypatch, tmp_path, out):
    exe = tmp_path / "avo"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setenv("AVOGADRO_COMMAND", str(exe))
    assert install.find()["source"] == "AVOGADRO_COMMAND"
    monkeypatch.setenv("AVOGADRO_COMMAND", str(tmp_path / "missing"))
    got = install.find()
    assert not got["found"] and "Where Avogadro is" in got["problem"]
    monkeypatch.delenv("AVOGADRO_COMMAND")
    monkeypatch.setenv("PATH", str(tmp_path))
    assert not install.find()["found"]


async def test_tools_say_what_is_missing(monkeypatch, out):
    monkeypatch.setattr(
        install,
        "find",
        lambda: {"found": False, "source": None, "problem": "Avogadro 2 is not installed"},
    )
    monkeypatch.setattr(install, "platform_supported", lambda: (True, ""))
    async with Client(mcp) as client:
        status = await client.call_tool("avogadro_status", {})
        assert "install-avogadro" in status.content[0].text
        with pytest.raises(ToolError, match=r"install-avogadro.*\[not_installed\]"):
            await client.call_tool("open_in_avogadro", {})
        with pytest.raises(ToolError, match=r"\[nothing_sent\]"):
            await client.call_tool("read_avogadro", {})
        tools = {t.name: t for t in await client.list_tools()}
    assert set(tools) == {
        "avogadro_status",
        "open_in_avogadro",
        "read_avogadro",
        "avogadro_snapshot",
        "close_avogadro",
    }
