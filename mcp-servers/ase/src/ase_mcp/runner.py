"""Run a calculation: validate, gate the calculator, build its environment, start the
worker, relay its progress, and read its result or typed error."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

from . import calculators, envs
from .errors import AseError
from .records import new_run_dir
from .schemas import RunInput

ProgressFn = Callable[[float, float | None, str], Awaitable[None]]


async def _noop(progress: float, total: float | None, message: str) -> None:
    return None


async def prepare_env(extra: str | None, report: ProgressFn) -> dict[str, Any] | None:
    """Make sure the environment of ``extra`` exists; ``None`` for the server's own."""
    if extra is None:
        return None
    st = envs.status(extra)
    if not st["prepared"]:
        envs.uv_executable()
        await report(
            0,
            None,
            f"Building the {extra} calculator environment (first use, "
            f"downloads {envs.APPROX_SIZE.get(extra)}; can take minutes)",
        )
    task = asyncio.create_task(asyncio.to_thread(envs.ensure, extra))
    beat = 0
    while not task.done():
        await asyncio.sleep(2.0)
        beat += 2
        if not task.done() and not st["prepared"]:
            await report(beat, None, f"Building the {extra} environment ({beat} s)")
    return task.result()


async def run(job: RunInput, output_dir: str | None, report: ProgressFn = _noop) -> dict:
    resolved = job.calculator.resolved()
    calculators.gate(resolved, job.driver)
    extra = calculators.EXTRA[resolved["type"]]
    env_info = await prepare_env(extra, report)

    run_dir = new_run_dir(output_dir, job.run_name)
    job = job.model_copy(update={"run_dir": str(run_dir), "run_name": run_dir.name})
    job_file = run_dir / "input.json"
    job_file.write_text(job.model_dump_json(indent=2), encoding="utf-8")

    cmd, env = envs.worker_command(extra, job_file)
    log_path = run_dir / "worker.log"
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=log, stderr=asyncio.subprocess.STDOUT, env=env, cwd=str(run_dir)
        )
        progress_file = run_dir / "progress.json"
        last = None
        try:
            while True:
                try:
                    await asyncio.wait_for(proc.wait(), timeout=0.5)
                    break
                except TimeoutError:
                    pass
                try:
                    p = json.loads(progress_file.read_text())
                except (OSError, ValueError):
                    continue
                if p != last:
                    last = p
                    await report(
                        float(p.get("done") or 0), p.get("total"), str(p.get("message", ""))
                    )
        except asyncio.CancelledError:
            proc.kill()
            await proc.wait()
            raise
    progress_file.unlink(missing_ok=True)

    error_file = run_dir / "error.json"
    result_file = run_dir / "result.json"
    if proc.returncode == 0 and result_file.is_file():
        result = json.loads(result_file.read_text(encoding="utf-8"))
        if env_info is not None:
            result["worker_environment"] = {
                k: env_info.get(k)
                for k in ("extra", "path", "built_now", "build_seconds", "size_bytes")
            }
        result["elapsed_s"] = round(time.time() - t0, 2)
        return result
    if error_file.is_file():
        err = json.loads(error_file.read_text(encoding="utf-8"))
        raise AseError(
            err.get("reason", "calculation_failed"),
            f"{err.get('message')} (run folder {run_dir}; log worker.log)",
            run_dir=str(run_dir),
        )
    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
    raise AseError(
        "calculation_failed",
        f"The calculation process ended with exit code {proc.returncode} without a result "
        f"(crash or out of memory). Last lines of {log_path}:\n" + "\n".join(tail),
        run_dir=str(run_dir),
    )
