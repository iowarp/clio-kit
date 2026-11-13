"""Package settings that the step tools forward to the handler as keywords.

``jarvis_add_step``, ``jarvis_edit_step`` and the legacy ``append_pkg`` /
``configure_pkg`` tools splat the caller's config dict into
:func:`append_pkg` / :func:`configure_pkg` next to those functions' own
parameters. A config key named like one of them (``target``, ``pkg_id``, ...)
is passed twice, and Python raises a raw ``TypeError`` before the handler runs
(issue #385). No JARVIS-CD package declares a setting with any of these names
(checked against jarvis-cd 1.8.1's package menus), so such a key is always an
argument in the wrong place: refuse it and name the parameter it belongs in.
"""

from collections.abc import Mapping
from typing import Any

from fastmcp.exceptions import ToolError

# Handler parameter -> the tool parameter that carries it, or None when the
# caller cannot set it. One map per tool, because their names differ.
ADD_STEP: Mapping[str, str | None] = {
    "pipeline_id": "pipeline_id",
    "pkg_type": "package_name",
    "pkg_id": "step_id",
    "target": "target",
    "do_configure": None,
    "agent_visible_only": None,
}
EDIT_STEP: Mapping[str, str | None] = {
    "pipeline_id": "pipeline_id",
    "pkg_id": "step_id",
    "agent_visible_only": None,
}
APPEND_PKG: Mapping[str, str | None] = {
    "pipeline_id": "pipeline_id",
    "pkg_type": "pkg_type",
    "pkg_id": "pkg_id",
    "do_configure": "do_configure",
    "target": "target",
    "agent_visible_only": None,
}
CONFIGURE_PKG: Mapping[str, str | None] = {
    "pipeline_id": "pipeline_id",
    "pkg_id": "pkg_id",
    "agent_visible_only": None,
}


def package_settings(
    settings: Mapping[str, Any] | None, reserved: Mapping[str, str | None]
) -> dict[str, Any]:
    """Return ``settings`` as keywords, refusing any key the handler call owns."""
    settings = dict(settings or {})
    clashes = sorted(key for key in settings if key in reserved)
    if not clashes:
        return settings
    reasons = []
    for key in clashes:
        parameter = reserved[key]
        if parameter is None:
            reasons.append(f"'{key}' is not a package setting and cannot be set here")
        else:
            reasons.append(
                f"'{key}' is not a package setting; pass it as the top-level "
                f"'{parameter}' parameter"
            )
    raise ToolError("config rejected: " + "; ".join(reasons))


def edit_step_settings(
    config: Mapping[str, Any] | None, operation: str
) -> dict[str, Any]:
    """``jarvis_edit_step``'s config: required to edit, refused on remove."""
    if operation == "edit":
        if config is None:
            raise ToolError("config is required when operation='edit'")
        return package_settings(config, EDIT_STEP)
    if config not in (None, {}):
        raise ToolError("config is not accepted when operation='remove'")
    return {}
