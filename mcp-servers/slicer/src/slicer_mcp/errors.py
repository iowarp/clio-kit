"""The one error type of the slicer server: a stable reason code and words for a person.

Every failure the agent can meet carries a ``reason`` (``slicer_missing``,
``profile_not_found``, ``model_not_found``, ``slice_failed`` …) and a message that says
what happened and what to do, so the agent can pass it on as it is. A tool turns it into
an MCP tool error whose text ends with ``[reason]``.
"""

from __future__ import annotations

from typing import Any


class SlicerError(Exception):
    """A typed failure; ``reason`` is stable, ``details`` is extra structured context."""

    def __init__(self, reason: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.reason = reason
        self.details = details

    def text(self) -> str:
        return f"{self} [{self.reason}]"


__all__ = ["SlicerError"]
