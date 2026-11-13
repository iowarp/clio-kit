"""The node half's typed refusal and path helper."""

from __future__ import annotations

import os


class CommandError(Exception):
    """A command the engine refused; the scene is unchanged."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _expand(path: str) -> str:
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
