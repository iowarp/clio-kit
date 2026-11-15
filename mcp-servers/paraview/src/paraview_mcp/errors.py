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
    "Where ParaView is found is set by the ParaView connector's values, which the "
    "person or the agent can set: to use a ParaView on another machine, set `address` "
    "('Where ParaView runs') to its SSH host name, and `via` ('Jump host') if it is "
    "reached through another host; to use one on this machine, set `prepare` "
    "('Command to run first') to a line that puts pvpython on the PATH, for example "
    "'module load paraview' or 'export PATH=/opt/paraview/bin:$PATH'."
)


def no_paraview(where: str, detail: str = "") -> ParaViewError:
    """No pvpython was found where the connector looked."""
    extra = f" {detail}" if detail else ""
    return ParaViewError(
        "no_paraview",
        f"No ParaView was found {where}: pvpython is not on the PATH.{extra} "
        + HOW_TO_POINT,
    )
