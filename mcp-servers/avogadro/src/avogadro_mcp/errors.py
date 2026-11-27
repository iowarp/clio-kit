"""The one error type of the avogadro server: a stable reason code and words for a person.

Reasons: ``not_found``, ``invalid_input``, ``not_installed`` (no Avogadro yet),
``unsupported_platform``, ``display_missing`` (no Xvfb or x11vnc), ``download_failed``,
``checksum_mismatch``, ``start_failed``, ``not_running``, ``nothing_sent``, ``rpc_failed``.
A tool turns the error into an MCP tool error whose text ends with ``[reason]``.
"""

from __future__ import annotations

from typing import Any


class AvogadroError(Exception):
    """A typed failure; ``reason`` is stable, ``details`` is extra structured context."""

    def __init__(self, reason: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.reason = reason
        self.details = details

    def text(self) -> str:
        return f"{self} [{self.reason}]"


__all__ = ["AvogadroError"]
