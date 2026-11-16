"""The one error type of the graspa server: a stable reason code and words for a person.

Reasons: ``not_found``, ``invalid_input``, ``unreadable_structure``,
``binary_missing`` (no gRASPA executable), ``graspa_failed`` (gRASPA exited non-zero or
printed no loading), ``timeout``, ``unreadable_output``.
A tool turns the error into an MCP tool error whose text ends with ``[reason]``.
"""

from __future__ import annotations

from typing import Any


class GraspaError(Exception):
    """A typed failure; ``reason`` is stable, ``details`` is extra structured context."""

    def __init__(self, reason: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.reason = reason
        self.details = details

    def text(self) -> str:
        return f"{self} [{self.reason}]"


__all__ = ["GraspaError"]
