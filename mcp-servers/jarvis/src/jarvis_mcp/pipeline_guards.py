"""Day-one and conflict guards shared by pipeline handlers and package discovery.

Kept out of ``capabilities/jarvis_handler.py`` so that module stays within its
file-size ratchet; nothing here imports another ``jarvis_mcp`` module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import HTTPException

UNINITIALIZED_HINT = (
    "JARVIS has no configuration in this JARVIS_ROOT yet. One-time admin step: "
    "start the server with '--profile all' (clio-kit mcp-server jarvis -- "
    "--profile all) and call jm_create_config with config_dir, private_dir and "
    "shared_dir; the default profile works afterwards."
)


def failure(operation: str, error: Exception) -> HTTPException:
    """Return the 500 for a failed operation, naming the admin step when it applies.

    An ``HTTPException`` raised on purpose (such as the 409 below) passes through.
    """
    if isinstance(error, HTTPException):
        return error
    hint = f". {UNINITIALIZED_HINT}" if "not initialized" in str(error) else ""
    return HTTPException(status_code=500, detail=f"{operation} failed: {error}{hint}")


def refuse_existing(pipeline: Any, pipeline_id: str) -> Any:
    """Return ``pipeline`` unless ``pipeline_id`` is already stored on disk.

    ``Pipeline.create()`` resets packages and environment in place, so an
    existing id must be refused rather than silently emptied.
    """
    jarvis: Any = getattr(pipeline, "jarvis", None)
    directory = (
        jarvis.get_pipeline_dir(pipeline_id)
        if hasattr(jarvis, "get_pipeline_dir")
        else None
    )
    if (
        isinstance(directory, (str, os.PathLike))
        and (Path(directory) / "pipeline.yaml").exists()
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Create refused: pipeline '{pipeline_id}' already exists; "
                "choose another pipeline_id, or keep this one and change "
                "its steps with jarvis_add_step / jarvis_edit_step"
            ),
        )
    return pipeline
