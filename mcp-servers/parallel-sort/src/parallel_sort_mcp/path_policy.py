"""Which files this server's tools may read and write.

Tools take file paths from the model, and the model's input can come from any
document it was shown. Without a limit, a read-only tool returns the contents of
any file the server can open, such as ``~/.ssh/id_rsa`` or a ``.env``. Every
path argument is therefore checked before the tool runs:

- it must resolve (symbolic links followed) under an allowed root: the working
  directory, ``CLIO_KIT_ARTIFACTS`` when set, and the folders listed in
  ``PARALLEL_SORT_MCP_ALLOWED_ROOTS`` (separated by ``os.pathsep``);
- below that root, no file or folder on the way may be hidden (start with
  ``.``), so ``.ssh``, ``.aws`` and ``.env`` stay closed even when the working
  directory is the home folder.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware, MiddlewareContext

ROOTS_ENV = "PARALLEL_SORT_MCP_ALLOWED_ROOTS"

#: Tool arguments that name a file or folder.
PATH_ARGUMENTS = frozenset({"log_file", "output_file", "file_path"})


def allowed_roots() -> list[Path]:
    """The resolved folders tools may use, read from the environment each call."""
    roots = [os.getcwd()]
    if os.environ.get("CLIO_KIT_ARTIFACTS"):
        roots.append(os.environ["CLIO_KIT_ARTIFACTS"])
    roots.extend(
        p for p in os.environ.get(ROOTS_ENV, "").split(os.pathsep) if p.strip()
    )
    return [Path(root).expanduser().resolve() for root in roots]


def path_refusal(path: str) -> str | None:
    """Why ``path`` may not be used, or ``None`` when it may."""
    resolved = Path(path).expanduser().resolve()
    for root in allowed_roots():
        if resolved == root or resolved.is_relative_to(root):
            hidden = [
                part
                for part in resolved.relative_to(root).parts
                if part.startswith(".")
            ]
            if hidden:
                return f"{path}: hidden files and folders ({hidden[0]}) are not opened"
            return None
    roots = ", ".join(str(root) for root in allowed_roots())
    return (
        f"{path} is outside the folders this server may use ({roots}); "
        f"add its folder to {ROOTS_ENV} to allow it"
    )


class AllowedRootsMiddleware(Middleware):
    """Refuse a tool call whose path arguments fall outside :func:`allowed_roots`."""

    async def on_call_tool(self, context: MiddlewareContext, call_next: Any) -> Any:
        arguments = getattr(context.message, "arguments", None) or {}
        for name in PATH_ARGUMENTS & arguments.keys():
            value = arguments[name]
            for item in value if isinstance(value, list) else [value]:
                if isinstance(item, str) and item.strip():
                    reason = path_refusal(item)
                    if reason is not None:
                        raise ToolError(reason)
        return await call_next(context)
