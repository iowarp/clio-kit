"""The one error type of the decimer server: a stable reason code and words for a person.

Reasons: ``image_not_found``, ``unreadable_image``, ``image_too_large``, ``bad_box``,
``model_download_failed``, ``model_failed``, ``invalid_smiles``. A tool turns the error
into an MCP tool error whose text ends with ``[reason]``.
"""

from __future__ import annotations

from typing import Any


class DecimerError(Exception):
    """A typed failure; ``reason`` is stable, ``details`` is extra structured context."""

    def __init__(self, reason: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.reason = reason
        self.details = details

    def text(self) -> str:
        return f"{self} [{self.reason}]"


__all__ = ["DecimerError"]
