"""Exercise the real Bash boundary used to invoke site Lmod installations."""

import asyncio
import os

import pytest
from lmod_mcp.capabilities import module_runtime, lmod_handler


@pytest.mark.asyncio
async def test_environment_survives_calls_and_arguments_are_not_shell_code(
    tmp_path, monkeypatch
):
    backend = tmp_path / "lmod"
    backend.write_text("""#!/bin/bash
if [[ "$2" == "restore" ]]; then
    printf 'export CLIO_MODULE_VALUE=%q\\n' "$3"
else
    printf '%s\\n' "$CLIO_MODULE_VALUE" >&2
fi
""")
    backend.chmod(0o755)
    monkeypatch.setenv("LMOD_CMD", str(backend))
    monkeypatch.setattr(module_runtime, "_changes", {})
    monkeypatch.setattr(module_runtime, "_removed", set())
    monkeypatch.setattr(module_runtime, "_lock", asyncio.Lock())
    marker = tmp_path / "unexpected"
    value = f"$(touch {marker})"
    assert (await lmod_handler._run_module_command(["restore", value], True))[2] == 0
    _, stderr, status = await lmod_handler._run_module_command(["list"], True)
    assert status == 0 and stderr.strip() == value
    assert not marker.exists()
    assert "CLIO_MODULE_VALUE" not in os.environ


@pytest.mark.asyncio
async def test_failed_avail_is_not_a_successful_module_list(monkeypatch):
    async def failure(*args, **kwargs):
        return "", "Lmod error: missing installation", 1

    monkeypatch.setattr(lmod_handler, "_run_module_command", failure)
    assert (await lmod_handler.search_available_modules())["success"] is False
