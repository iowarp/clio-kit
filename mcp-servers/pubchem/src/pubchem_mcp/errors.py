"""The one error type of the pubchem server: a stable reason code and words for a person.

Reasons: ``not_found`` (no compound for that name or CID), ``pubchem_unavailable``
(network down, PubChem busy after retries), ``bad_request``. A tool turns the error
into an MCP tool error whose text ends with ``[reason]``.
"""

from __future__ import annotations

from typing import Any


class PubChemError(Exception):
    """A typed failure; ``reason`` is stable, ``details`` is extra structured context."""

    def __init__(self, reason: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.reason = reason
        self.details = details

    def text(self) -> str:
        return f"{self} [{self.reason}]"


__all__ = ["PubChemError"]
