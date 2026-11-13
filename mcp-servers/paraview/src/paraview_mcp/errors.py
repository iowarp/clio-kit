"""Typed failures the front reports to the agent in plain words."""

from __future__ import annotations


class ParaViewError(Exception):
    """A failure with a stable code and a message that says what to do."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def __str__(self) -> str:
        return f"{self.message} [{self.code}]"


HOW_TO_POINT = (
    "To use a ParaView elsewhere, set the ParaView connector's 'Where ParaView runs' "
    "to the SSH host name of that machine (and 'Jump host' if you reach it through "
    "another host). To use one on this machine, set 'Command to run first' to a line "
    "that puts pvpython on the PATH, for example 'module load paraview' or "
    "'export PATH=/opt/paraview/bin:$PATH'."
)


def no_paraview(where: str, detail: str = "") -> ParaViewError:
    """No pvpython was found where the connector looked."""
    extra = f" {detail}" if detail else ""
    return ParaViewError(
        "no_paraview",
        f"No ParaView was found {where}: pvpython is not on the PATH.{extra} "
        + HOW_TO_POINT,
    )
