"""The one error type of the xcompact3d server: a stable reason code and words for a person.

A tool turns it into an MCP tool error whose text ends with ``[reason]``.
"""

from __future__ import annotations


class CaseError(Exception):
    """A typed failure; ``reason`` is stable."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason

    def text(self) -> str:
        return f"{self} [{self.reason}]"


__all__ = ["CaseError"]
