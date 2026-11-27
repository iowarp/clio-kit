"""Registry distribution coordinates, independent of a server's implementation language."""

from __future__ import annotations

from typing import Any

PACKAGE_TYPES = {"pypi", "npm", "oci", "nuget", "mcpb"}


def registry_package(configuration: dict[str, Any]) -> dict[str, Any]:
    """Validate explicit distribution metadata before emitting a registry package."""
    kind = configuration.get("registryType")
    if kind not in PACKAGE_TYPES:
        raise ValueError(f"registryType must be one of {sorted(PACKAGE_TYPES)}")
    for field in ("identifier", "version"):
        if not isinstance(configuration.get(field), str) or not configuration[field]:
            raise ValueError(f"Registry package needs {field}")
    allowed = {
        "registryType",
        "identifier",
        "version",
        "runtimeHint",
        "registryBaseUrl",
        "fileSha256",
        "runtimeArguments",
        "packageArguments",
        "environmentVariables",
        "transport",
    }
    unknown = set(configuration) - allowed
    if unknown:
        raise ValueError(f"Unknown registry package fields: {sorted(unknown)}")
    package = dict(configuration)
    package.setdefault("transport", {"type": "stdio"})
    return package
