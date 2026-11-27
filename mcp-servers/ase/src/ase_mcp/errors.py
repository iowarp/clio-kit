"""The one error type of the ase server: a stable reason code and words for a person.

Every failure the agent can meet carries a ``reason`` (``not_found``, ``invalid_input``,
``binary_missing``, ``hf_token_missing``, ``uv_missing``, ``env_build_failed``,
``calculation_failed`` ...) and a message that says what happened and what to do. A
tool turns it into an MCP tool error whose text ends with ``[reason]``.
"""

from __future__ import annotations

from typing import Any


class AseError(Exception):
    """A typed failure; ``reason`` is stable, ``details`` is extra structured context."""

    def __init__(self, reason: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.reason = reason
        self.details = details

    def text(self) -> str:
        return f"{self} [{self.reason}]"

    def to_dict(self) -> dict[str, Any]:
        return {"reason": self.reason, "message": str(self), "details": self.details}


__all__ = ["AseError"]
