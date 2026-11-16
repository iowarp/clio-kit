"""The one error type of the fdmnes server: a stable reason code and words for a person.

Reasons: ``not_found`` (a file that does not exist), ``invalid_input``,
``unreadable_structure``, ``binary_missing`` (no FDMNES executable), ``fdmnes_failed``
(FDMNES exited non-zero or wrote no spectrum), ``timeout``, ``unreadable_spectrum``, and for the Materials Project ``mp_key_missing``,
``mp_key_rejected``, ``mp_unavailable``.
A tool turns the error into an MCP tool error whose text ends with ``[reason]``.
"""

from __future__ import annotations

from typing import Any


class FdmnesError(Exception):
    """A typed failure; ``reason`` is stable, ``details`` is extra structured context."""

    def __init__(self, reason: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.reason = reason
        self.details = details

    def text(self) -> str:
        return f"{self} [{self.reason}]"


__all__ = ["FdmnesError"]
