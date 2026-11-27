"""Find the gRASPA executable and run it in a run folder.

gRASPA (snurr-group/gRASPA, MIT) needs an NVIDIA (CUDA, NVHPC) or AMD (HIP) GPU; the
SYCL port runs on Intel GPUs. This server never downloads or builds it. It is found as
``GRASPA_COMMAND`` or ``CHEMGRAPH_GRASPA_EXECUTABLE`` (ChemGraph's variable), else
``nvc_main.x``, ``hip_main.x``, ``graspa`` or ``sycl.out`` on PATH. The graspa plugin's install skill
builds it with Spack on a GPU node (``spack install graspa``).
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

from .errors import GraspaError

ENV_VARS = ("GRASPA_COMMAND", "CHEMGRAPH_GRASPA_EXECUTABLE")
PATH_NAMES = ("nvc_main.x", "hip_main.x", "graspa", "sycl.out")
Report = Callable[[float, float | None, str], Awaitable[None]]
CYCLE = re.compile(r"^(INITIALIZATION|EQUILIBRATION|PRODUCTION) Cycle:\s*(\d+),\s*(\d+) Adsorbate")


def find_executable() -> dict[str, Any]:
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
    return {"found": False, "source": None, "problem": "no gRASPA executable found"}


def missing_text(found: dict[str, Any]) -> str:
    """Where gRASPA was looked for, the setting that points at it, and the GPU-node route."""
    where = (
        f"gRASPA was not found: {found['problem']}"
        if found.get("source")
        else f"gRASPA was not found in {', '.join(ENV_VARS)} or PATH ({', '.join(PATH_NAMES)})"
    )
    return (
        f"{where}. If it is installed elsewhere, set GRASPA_COMMAND to the executable; "
        "otherwise see the graspa plugin's install skill (it builds gRASPA on a GPU node). "
        "Without gRASPA here, prepare_graspa writes the run folder to submit to a GPU node "
        "and read_graspa reads the result."
    )


def require_executable() -> str:
    found = find_executable()
    if not found["found"]:
        raise GraspaError("binary_missing", missing_text(found))
    return str(found["command"])


async def run(
    command: str, run_dir: Path, timeout_s: float | None, report: Report
) -> dict[str, Any]:
    out_path, err_path = run_dir / "graspa_stdout.txt", run_dir / "graspa_stderr.txt"
    started = time.monotonic()
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("ZE_FLAT_DEVICE_HIERARCHY", "FLAT")
    with out_path.open("wb") as out, err_path.open("wb") as err:
        proc = await asyncio.create_subprocess_exec(
            command, cwd=str(run_dir), stdout=asyncio.subprocess.PIPE, stderr=err, env=env
        )

        async def pump() -> None:
            assert proc.stdout is not None
            reports = 0
            async for raw in proc.stdout:
                out.write(raw)
                m = CYCLE.match(raw.decode(errors="replace"))
                if m:
                    phase, cycle, molecules = m.group(1), int(m.group(2)), m.group(3)
                    reports += 1
                    await report(
                        float(reports),
                        None,
                        f"{phase.lower()} cycle {cycle}: {molecules} molecules",
                    )

        try:
            await asyncio.wait_for(asyncio.gather(pump(), proc.wait()), timeout_s)
        except TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise GraspaError(
                "timeout",
                f"gRASPA did not finish within {timeout_s:g} s and was stopped; {run_dir} "
                "keeps its output.",
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
