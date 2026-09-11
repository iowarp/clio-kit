"""Invoke real Lmod shell integration and retain its process-local environment."""

import asyncio
import os
from pathlib import Path
import shutil

_lock = asyncio.Lock()
_changes: dict[str, str] = {}
_removed: set[str] = set()
_MARKER = b"\0CLIO_LMOD_ENVIRONMENT\0"
_SCRIPT = r"""clio_module_code=$("$LMOD_CMD" bash "$@") || exit $?
eval "$clio_module_code" || exit $?
printf '\0CLIO_LMOD_ENVIRONMENT\0'
env -0
"""


def environment() -> dict[str, str]:
    result = dict(os.environ)
    for name in _removed:
        result.pop(name, None)
    result.update(_changes)
    return result


def lmod_command() -> str | None:
    configured = os.environ.get("LMOD_CMD")
    if configured:
        return configured
    found = shutil.which("lmod")
    if found:
        return found
    for candidate in (
        "/usr/share/lmod/lmod/libexec/lmod",
        "/usr/local/lmod/lmod/libexec/lmod",
    ):
        if Path(candidate).is_file():
            return candidate
    return None


async def run_lmod(
    command: str, args: list[str], capture_stderr: bool
) -> tuple[str, str, int]:
    """Evaluate trusted site modulefiles in Bash; keep arguments separate from code."""
    # Lmod parses global options before its subcommand.
    if "-t" in args or args[0] == "savelist":
        args = ["-t", *[arg for arg in args if arg != "-t"]]
    async with _lock:
        env = environment()
        env.update(
            LMOD_CMD=command, LMOD_QUIET="1", LMOD_COLORIZE="no", LMOD_PAGER="none"
        )
        # Startup files must not contaminate either the protocol or captured state.
        env.pop("BASH_ENV", None)
        process = await asyncio.create_subprocess_exec(
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            _SCRIPT,
            "clio-lmod",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await process.communicate()
        code = process.returncode or 0
        if code == 0:
            output, marker, state = stdout.rpartition(_MARKER)
            if not marker:
                return "", "Lmod did not return its environment", 1
            updated = dict(
                item.decode().split("=", 1)
                for item in state.split(b"\0")
                if b"=" in item
            )
            # Shell bookkeeping and invocation settings are not module state.
            ignored = {
                "_",
                "SHLVL",
                "PWD",
                "OLDPWD",
                "LMOD_CMD",
                "LMOD_QUIET",
                "LMOD_COLORIZE",
                "LMOD_PAGER",
            }
            for name in (set(env) | set(updated)) - ignored:
                if name not in updated:
                    _changes.pop(name, None)
                    _removed.add(name)
                elif updated[name] != env.get(name):
                    _changes[name] = updated[name]
                    _removed.discard(name)
            stdout = output
        if not capture_stderr:
            stdout += stderr
            stderr = b""
        return stdout.decode(), stderr.decode(), code
