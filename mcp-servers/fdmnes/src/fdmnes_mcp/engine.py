"""Find the FDMNES executable, run it in a run folder, and read what it wrote.

FDMNES is a separate program (Institut Néel, CNRS; free, its own licence) that this
server never downloads. It is found as ``FDMNES_COMMAND`` or ``FDMNES_EXE`` (the
variable ChemGraph uses), else ``fdmnes`` / ``fdmnes_linux`` on PATH. The fdmnes
plugin's install skill builds it with Spack (``spack install fdmnes``).
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .errors import FdmnesError

ENV_VARS = ("FDMNES_COMMAND", "FDMNES_EXE")
PATH_NAMES = ("fdmnes", "fdmnes_linux")
Report = Callable[[float, float | None, str], Awaitable[None]]


def find_executable() -> dict[str, Any]:
    """Where FDMNES is, and how it was found; ``found`` false says what to set."""
    for var in ENV_VARS:
        value = os.environ.get(var)
        if value:
            path = shutil.which(value) or (value if Path(value).is_file() else None)
            if path and os.access(path, os.X_OK):
                return {"found": True, "command": str(Path(path).resolve()), "source": var}
            return {
                "found": False,
                "source": var,
                "problem": f"{var}={value!r} is not an executable file",
            }
    for name in PATH_NAMES:
        path = shutil.which(name)
        if path:
            return {"found": True, "command": str(Path(path).resolve()), "source": "PATH"}
    return {"found": False, "source": None, "problem": "no FDMNES executable found"}


def require_executable() -> str:
    found = find_executable()
    if not found["found"]:
        raise FdmnesError(
            "binary_missing",
            f"FDMNES is not available here ({found['problem']}). Install it (the fdmnes "
            "plugin's install skill builds it with Spack: spack install fdmnes) and set "
            "FDMNES_COMMAND to the executable, or put fdmnes on PATH.",
        )
    return str(found["command"])


async def run(
    command: str, run_dir: Path, timeout_s: float | None, report: Report
) -> dict[str, Any]:
    """Run FDMNES in ``run_dir``; stdout and stderr go to files there."""
    out_path, err_path = run_dir / "fdmnes_stdout.txt", run_dir / "fdmnes_stderr.txt"
    started = time.monotonic()
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    with out_path.open("wb") as out, err_path.open("wb") as err:
        proc = await asyncio.create_subprocess_exec(
            command, cwd=str(run_dir), stdout=asyncio.subprocess.PIPE, stderr=err, env=env
        )
        assert proc.stdout is not None
        cycles = 0

        async def pump() -> None:
            nonlocal cycles
            assert proc.stdout is not None
            async for raw in proc.stdout:
                out.write(raw)
                line = raw.decode(errors="replace")
                m = re.match(r"\s*Cycle\s+(\d+)", line)
                if m and int(m.group(1)) != cycles:
                    cycles = int(m.group(1))
                    await report(cycles, None, f"SCF cycle {cycles}")
                elif "Number of Energies" in line:
                    await report(cycles + 1, None, "XANES calculated, convoluting")

        try:
            await asyncio.wait_for(asyncio.gather(pump(), proc.wait()), timeout_s)
        except TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise FdmnesError(
                "timeout",
                f"FDMNES did not finish within {timeout_s:g} s and was stopped; the run "
                f"folder {run_dir} keeps its output. Use a smaller radius, method "
                "'green', or a longer timeout.",
                run_dir=str(run_dir),
            ) from exc
    return {
        "returncode": proc.returncode,
        "wall_time_s": round(time.monotonic() - started, 2),
        "stdout": str(out_path),
        "stderr": str(err_path),
    }


def tail(path: Path, lines: int = 8) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def read_log(run_dir: Path, stem: str) -> dict[str, Any]:
    """Facts FDMNES reports: edge energy, absorbing sites computed, point group, SCF."""
    text = "\n".join(
        (run_dir / name).read_text(errors="replace")
        for name in ("fdmnes_stdout.txt", f"{stem}_bav.txt")
        if (run_dir / name).is_file()
    )
    info: dict[str, Any] = {}
    if m := re.search(r"E_edge\s*=\s*([-\d.]+)\s*eV", text):
        info["edge_energy_eV"] = float(m.group(1))
    if m := re.search(r"Threshold:\s*(.+)", text):
        info["threshold"] = m.group(1).strip()
    if m := re.search(r"non equivalent absorbing atom\s*=\s*(\d+)", text):
        info["inequivalent_absorbers"] = int(m.group(1))
    if m := re.search(r"Cluster radius =\s*([\d.]+) A, nb\. of atom =\s*(\d+)", text):
        info["cluster"] = {"radius_A": float(m.group(1)), "atoms": int(m.group(2))}
    if m := re.search(r"Point group :\s*(\S+)\s*\(\s*(\S+)\s*\)", text):
        info["point_group"] = f"{m.group(1)} ({m.group(2)})"
    cycles = [int(c) for c in re.findall(r"Cycle\s+(\d+)", text)]
    if cycles:
        info["scf_cycles"] = max(cycles)
    info["scf_not_converged"] = "has not converged" in text
    return info


def conv_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("*conv.txt"))
