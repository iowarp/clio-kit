"""
Slurm job output retrieval capabilities.
Handles job output and log file access.
"""

import os
import stat
from typing import Optional

from .utils import check_slurm_available
from .job_details import get_job_details


def get_job_output(
    job_id: str,
    output_type: str = "stdout",
    *,
    max_chars: Optional[int] = None,
) -> dict:
    """
    Get job output files (stdout/stderr).

    Args:
        job_id: The Slurm job ID
        output_type: Type of output ("stdout" or "stderr")
        max_chars: Optional maximum trailing characters to read. The legacy
            default reads the complete file.

    Returns:
        Dictionary with job output content
    """
    if not check_slurm_available():
        raise RuntimeError(
            "Slurm is not available on this system. Please install Slurm."
        )

    try:
        # Scheduler detail names the output file while the job is known. Once
        # it ages out of the controller on a cluster without accounting, the
        # file this server asked sbatch to write is still found by job ID.
        job_details = get_job_details(job_id).get("details") or {}
        extension = {"stdout": "out", "stderr": "err"}.get(output_type)
        possible_files = [job_details.get(output_type)] if extension else []
        if extension and os.path.basename(job_id) == job_id:
            possible_files += [
                f"logs/slurm_output/slurm_{job_id}.{extension}",
                f"slurm_{job_id}.{extension}",  # fallback for old files
            ]
        output_file = next(
            (path for path in possible_files if path and os.path.exists(path)), None
        )
        if output_file is None:
            return {
                "job_id": job_id,
                "output_type": output_type,
                "error": (
                    f"No {output_type} file found for job {job_id}: nothing at a "
                    "scheduler-reported path, and logs/slurm_output/ under "
                    f"{os.getcwd()} has no file for this job ID"
                ),
                "real_slurm": True,
            }
        content, truncated = _read_output(output_file, max_chars=max_chars)
        return {
            "job_id": job_id,
            "output_type": output_type,
            "file_path": output_file,
            "content": content,
            "truncated": truncated,
            "real_slurm": True,
        }

    except Exception as e:
        return {
            "job_id": job_id,
            "output_type": output_type,
            "error": str(e),
            "real_slurm": True,
        }


def _read_output(path: str, *, max_chars: Optional[int]) -> tuple[str, bool]:
    """Read an output file completely or as a bounded UTF-8-safe tail."""
    if max_chars is None:
        with open(path, "r", encoding="utf-8", errors="replace") as stream:
            return stream.read(), False
    if max_chars < 1:
        raise ValueError("max_chars must be positive")

    byte_window = max_chars * 4
    with open(path, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"job output is not a regular file: {path}")
        if metadata.st_size > byte_window:
            stream.seek(metadata.st_size - byte_window, os.SEEK_SET)
        content = stream.read(byte_window).decode("utf-8", errors="replace")
    truncated = metadata.st_size > byte_window or len(content) > max_chars
    return content[-max_chars:], truncated
