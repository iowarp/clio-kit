"""Structured-file provider protocol.

Provisional copy of section 3 of the draft RFC "CLIO Structured File
Integration". CLIO core owns this contract and has not published it yet; when
it does, import it from there and delete this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

NodeKind = Literal["container", "array", "table", "scalar", "text", "blob", "link"]

ErrorCode = Literal["not_found", "unsupported", "too_large", "io", "locked"]

ERROR_CODES: frozenset[str] = frozenset(
    {"not_found", "unsupported", "too_large", "io", "locked"}
)


class Node(TypedDict):
    locator: str
    name: str
    kind: NodeKind
    shape: tuple[int, ...] | None
    dtype: str | None
    has_children: bool


class Description(TypedDict):
    node: Node
    attributes: dict[str, Any]
    preview: Any
    storage: dict[str, Any]
    render_hint: str | None
    convention: str | None


class Payload(TypedDict):
    dtype: str
    shape: tuple[int, ...]
    data: Any
    truncated: bool


class ProviderError(Exception):
    """Error raised by a provider, carrying one of the fixed `ERROR_CODES`."""

    def __init__(self, code: ErrorCode, message: str, locator: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.locator = locator


@runtime_checkable
class StructuredFileProvider(Protocol):
    id: str
    media_types: frozenset[str]

    def detect(self, path: Path) -> bool: ...

    def children(self, path: Path, locator: str) -> list[Node]: ...

    def describe(self, path: Path, locator: str) -> Description: ...

    def read(self, path: Path, locator: str, selection: str | None) -> Payload: ...

    def canonicalize(self, locator: str) -> str: ...

    def complete(self, path: Path, partial: str) -> list[str]: ...
