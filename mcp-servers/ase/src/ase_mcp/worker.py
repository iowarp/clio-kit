"""Worker entry point: ``python -m ase_mcp.worker <run_dir>/input.json``.

The server runs every calculation in this separate process, inside the environment the
calculator needs (:mod:`.envs`), so a long or crashing calculation never blocks or takes
down the server. The worker writes ``result.json`` on success, or ``error.json`` with a
typed reason on failure, and ``progress.json`` while it runs.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m ase_mcp.worker <input.json>", file=sys.stderr)
        return 64
    job_file = Path(args[0])
    run_dir = job_file.parent
    error_file = run_dir / "error.json"
    try:
        from .compute import run
        from .schemas import RunInput

        job = RunInput.model_validate_json(job_file.read_text(encoding="utf-8"))
        run(job)
        return 0
    except Exception as exc:  # noqa: BLE001 - every failure leaves a typed error file
        typed = type(exc).__name__ == "AseError"
        if typed:
            reason = exc.reason  # type: ignore[attr-defined]
        elif isinstance(exc, ImportError):
            reason = "calculator_unavailable"
        else:
            reason = "calculation_failed"
        payload = {
            "reason": reason,
            "message": str(exc) if typed else f"{type(exc).__name__}: {exc}",
            "details": getattr(exc, "details", {}),
            "traceback": traceback.format_exc(),
        }
        error_file.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
