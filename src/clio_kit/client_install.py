"""Install shared plugin components using each client's project configuration."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
import tempfile
import uuid
from typing import Any

import tomli_w

from clio_kit.component_cache import guarded, register_project
from clio_kit.hooks import hook_components
from clio_kit.skill_cli import stage_skills
from clio_kit.install_transaction import InstallTransaction
from clio_kit.skills import read_skill_frontmatter

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]


# These are project-local paths; installation never changes the global profile.
CLIENTS = {
    "codex": (".agents/skills", ".codex/config.toml", "mcp_servers"),
    "opencode": (".opencode/skills", "opencode.json", "mcp"),
    "cursor": (".cursor/skills", ".cursor/mcp.json", "mcpServers"),
    "antigravity": (".agents/skills", ".agents/mcp_config.json", "mcpServers"),
    "claude-code": (".claude/skills", ".mcp.json", "mcpServers"),
    "vscode": (".github/skills", ".vscode/mcp.json", "servers"),
}


def collect_components(
    root: Path | None,
    name: str,
    *,
    stack: ExitStack | None = None,
    project: Path | None = None,
    update: bool = False,
) -> dict[str, Any]:
    """Resolve local packages and dependencies without running their code."""
    from clio_kit.local_plugins import discover_local_plugins
    from clio_kit.plugins import validate_plugin

    from clio_kit.catalogue import read_catalogue
    from clio_kit.external_plugins import fetch_source
    from clio_kit.client_adapters import native_components

    release = None
    artifacts: set[str] = set()
    if root is None:
        from clio_kit.component_store import catalogue, artifact_path
        from clio_kit.release_components import validate_index

        release = catalogue()
        validate_index(release)
        entries = list(release.get("external_entries", []))
        entries += [{"name": n, "source": None} for n in release["packages"]]
    else:
        entries = read_catalogue(root)["packages"]
        entries = entries + discover_local_plugins(root, entries)
    index = {entry["name"]: entry for entry in entries}
    skills: dict[str, Path] = {}
    servers: dict[str, dict] = {}
    unsupported: list[str] = []
    visited: set[str] = set()
    packages: dict[str, dict] = {}
    payloads: dict[Path, Path] = {}
    provenance: dict[str, dict] = {}

    def visit(package: str, trail: set[str]) -> None:
        if package in trail:
            raise ValueError(f"Cyclic dependency: {package}")
        if package in visited:
            return
        entry = index.get(package)
        if not entry:
            raise ValueError(f"Unknown package: {package}")
        if release is not None and package in release["packages"]:
            from clio_kit.client_adapters import expand

            record = release["packages"][package]
            manifest = record["manifest"]
            if manifest.get("skills"):
                raise ValueError(
                    f"{package}: this installer requires skills/*/SKILL.md layout"
                )
            if "native" not in record and record["unsupported"]:
                raise ValueError(
                    f"{package}: release lacks native adapter metadata; upgrade the launcher"
                )
            installed = artifact_path(record["artifact"], release)
            native = record.get("native", {"agents": {}, "commands": {}, "hooks": {}})
            packages[package] = {"installed": installed, "native": native}
            if any(native.values()):
                artifacts.add(record["artifact"])
            unsupported.extend(f"{package}: {field}" for field in record["unsupported"])
            for dependency in manifest.get("dependencies", []):
                visit(dependency, trail | {package})
            for skill in record["skills"]:
                key = release["skills"][skill]["artifact"]
                path = artifact_path(key, release)
                if skill in skills and skills[skill] != path:
                    raise ValueError(f"Duplicate skill: {skill}")
                skills[skill] = path
                artifacts.add(key)
            for server, settings in record["servers"].items():
                converted = expand(settings, installed)
                if converted != settings:
                    artifacts.add(record["artifact"])
                if server in servers and servers[server] != converted:
                    raise ValueError(f"Conflicting MCP definitions: {server}")
                servers[server] = converted
            visited.add(package)
            return
        import re

        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", package):
            raise ValueError(f"Invalid package name: {package}")
        source = entry["source"]
        external = not isinstance(source, str)
        if external:
            if stack is None or project is None:
                raise ValueError(
                    f"{package} is indexed externally; use plugin install to resolve it"
                )
            from clio_kit.external_plugins import package_digest

            lock = safe_destination(project, f".clio-kit/sources/{package}.json")
            pinned = (
                json.loads(lock.read_text()) if lock.is_file() and not update else None
            )
            if pinned is not None and pinned["source"] == source:
                digest = pinned["sha256"]
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)
                ):
                    raise ValueError("Invalid publisher package digest")
                installed = safe_destination(
                    project, f".clio-kit/packages/{package}/{digest}"
                )
                if package_digest(installed) != digest:
                    raise ValueError(
                        f"Modified publisher payload: {package}; use --update to fetch it again"
                    )
                directory, provenance[package] = installed, pinned
            else:
                temporary = Path(
                    stack.enter_context(
                        tempfile.TemporaryDirectory(prefix="clio-publisher-")
                    )
                )
                directory, provenance[package] = fetch_source(source, temporary)
                installed = safe_destination(
                    project,
                    f".clio-kit/packages/{package}/{provenance[package]['sha256']}",
                )
                payloads[installed] = directory
        else:
            directory = (root / source).resolve()
            installed = directory
        if not external and root is not None and not directory.is_relative_to(root):
            raise ValueError(f"Package leaves checkout: {package}")
        problems: list[str]
        standalone = (
            external and not (directory / ".claude-plugin/plugin.json").exists()
        )
        if standalone:
            if any(
                (directory / field).exists()
                for field in ("agents", "commands", "hooks", ".mcp.json")
            ):
                raise ValueError(f"{package}: native components require a manifest")
            manifest, problems = (
                {"name": package, "description": entry.get("description", "")},
                [],
            )
        else:
            manifest, problems = validate_plugin(
                directory, allow_reserved=True, skill_policy=not external
            )
        if problems:
            raise ValueError("; ".join(problems))
        if manifest["name"] != package:
            raise ValueError(
                f"Publisher package name differs from catalogue: {package}"
            )
        packages[package] = {
            "directory": directory,
            "installed": installed,
            "manifest": manifest,
            "native": native_components(directory, manifest),
        }
        for dependency in manifest.get("dependencies", []):
            visit(dependency, trail | {package})
        for field in ("agents", "commands"):
            if manifest.get(field) or any(directory.glob(f"{field}/*.md")):
                unsupported.append(f"{package}: {field}")
        if hook_components(directory, manifest)[0]:
            unsupported.append(f"{package}: hooks")
        if manifest.get("skills"):
            # Do not silently ignore custom paths or pretend they were copied.
            raise ValueError(
                f"{package}: this installer requires skills/*/SKILL.md layout"
            )
        skill_files = (
            [directory / "SKILL.md"]
            if standalone and (directory / "SKILL.md").is_file()
            else list(directory.glob("skills/*/SKILL.md"))
        )
        if standalone and not skill_files:
            raise ValueError(f"{package}: no skill or plugin manifest")
        for path in skill_files:
            skill = read_skill_frontmatter(
                path.parent, check_directory_name=not standalone
            )["name"]
            if skill in skills and skills[skill] != path.parent:
                raise ValueError(f"Duplicate skill: {skill}")
            if standalone and path.parent.name != skill:
                import shutil
                import re

                if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", skill):
                    raise ValueError(f"Invalid skill name: {skill}")
                assert stack is not None
                parent = Path(
                    stack.enter_context(
                        tempfile.TemporaryDirectory(prefix="clio-skill-")
                    )
                )
                shutil.copytree(path.parent, parent / skill)
                skills[skill] = parent / skill
            else:
                skills[skill] = path.parent
        configurations = []
        if (directory / ".mcp.json").is_file():
            configurations.append(json.loads((directory / ".mcp.json").read_text()))
        inline = manifest.get("mcpServers")
        if isinstance(inline, str):
            inline = json.loads((directory / inline).read_text())
        if inline:
            configurations.append(inline)
        for configuration in configurations:
            for server, settings in configuration.get(
                "mcpServers", configuration
            ).items():
                from clio_kit.client_adapters import expand

                settings = expand(settings, installed)
                if external:
                    # Python imports must not mutate the retained, hash-verified
                    # publisher payload with __pycache__ files on first use.
                    settings = {
                        **settings,
                        "env": {
                            "PYTHONDONTWRITEBYTECODE": "1",
                            **settings.get("env", {}),
                        },
                    }
                if server in servers and servers[server] != settings:
                    raise ValueError(f"Conflicting MCP definitions: {server}")
                servers[server] = settings
        visited.add(package)

    visit(name, set())
    return {
        "skills": skills,
        "servers": servers,
        "unsupported": unsupported,
        "packages": packages,
        "payloads": payloads,
        "provenance": provenance,
        "artifacts": sorted(artifacts),
    }


def server_settings(settings: dict, client: str) -> dict:
    """Convert the shared stdio subset; reject host-specific options explicitly."""
    allowed = {"command", "args", "env", "type"}
    if set(settings) - allowed or settings.get("type", "stdio") != "stdio":
        raise ValueError(
            "Project installation currently supports stdio command/args/env only; "
            "use native client configuration for remote or host-specific MCP options"
        )
    if not isinstance(settings.get("command"), str):
        raise ValueError("MCP command must be a string")
    if "${" in json.dumps(settings):
        raise ValueError("Unresolved MCP variables need client-specific configuration")
    result = {key: value for key, value in settings.items() if key != "type"}
    if client == "opencode":
        result = {
            "type": "local",
            "command": [settings["command"], *settings.get("args", [])],
            "enabled": True,
            "timeout": 120000,
        }
        if settings.get("env"):
            result["environment"] = settings["env"]
    elif client in {"vscode", "claude-code"}:
        result["type"] = "stdio"
    elif client == "codex":
        # First use builds the server's locked environment; Codex defaults to 10s.
        result["startup_timeout_sec"] = 300
    return result


def safe_destination(project: Path, relative: str) -> Path:
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"Installation destination leaves project: {relative}")
    path = project / relative
    for part in (path, *path.parents):
        if part == project:
            break
        if part.is_symlink():
            raise ValueError(f"Refusing linked installation destination: {part}")
    return path


@guarded
def install_for_client(
    root: Path | None,
    name: str,
    client: str,
    project: Path,
    *,
    components_only: bool = False,
    replace: bool = False,
    dry_run: bool = False,
    update: bool = False,
) -> dict:
    with ExitStack() as stack:
        return _install(
            root,
            name,
            client,
            project,
            components_only=components_only,
            replace=replace,
            dry_run=dry_run,
            stack=stack,
            update=update,
        )


def _install(
    root, name, client, project, *, components_only, replace, dry_run, stack, update
):
    from clio_kit.client_adapters import plan_native, merge_configuration

    from clio_kit.install_receipts import Receipt

    project = project.resolve()
    receipt = Receipt(project, client, name)
    components = collect_components(
        root.resolve() if root else None,
        name,
        stack=stack,
        project=project,
        update=update,
    )
    native = plan_native(components.get("packages", {}), client, components["servers"])
    unsupported = (
        native["unsupported"]
        if components.get("packages")
        else components["unsupported"]
    )
    if unsupported and not components_only:
        raise ValueError(
            "This package also needs native client adapters for "
            + ", ".join(unsupported)
            + ". Use the native Claude package, or explicitly select --components-only "
            "to install just skills and MCPs."
        )
    if (
        not components["skills"]
        and not components["servers"]
        and (components_only or not components.get("packages"))
    ):
        raise ValueError("No portable skills or stdio MCPs to install")
    skill_path, config_path, key = CLIENTS[client]
    target = safe_destination(project, skill_path)
    config = safe_destination(project, config_path)
    if client == "opencode" and (project / "opencode.jsonc").exists():
        raise ValueError(
            "opencode.jsonc exists; merge configuration manually to preserve its comments and precedence"
        )
    original = config.read_bytes() if config.exists() else None
    data = (
        (
            tomllib.loads(original.decode())
            if client == "codex"
            else json.loads(original)
        )
        if original is not None
        else {}
    )
    if not isinstance(data, dict) or not isinstance(data.get(key, {}), dict):
        raise ValueError(f"Expected an object/table in {config}: {key}")
    owned = {} if components_only else native["configuration"]
    if components["servers"]:
        owned[key] = {
            server: server_settings(settings, client)
            for server, settings in components["servers"].items()
        }
    receipt.configuration(config_path, data, owned)
    configured = data.setdefault(key, {})
    for server, settings in owned.get(key, {}).items():
        if server in configured and configured[server] != settings and not replace:
            raise ValueError(
                f"{server} has different configuration; review before using --replace"
            )
    # A transport is one executable definition, not a mergeable list of args
    # or environment fragments. Hooks and other native settings still merge.
    configured.update(owned.get(key, {}))
    merge_configuration(
        data,
        {field: value for field, value in owned.items() if field != key},
        replace=replace,
    )
    result = {
        "client": client,
        "package": name,
        "skills": sorted(components["skills"]),
        "servers": sorted(components["servers"]),
        "skill_directory": str(target),
        "config": str(config)
        if components["servers"] or native["configuration"]
        else None,
        "not_installed": components["unsupported"] if components_only else unsupported,
        "agents": [] if components_only else native["agents"],
        "warnings": [] if components_only else native["warnings"],
        "sources": components.get("provenance", {}),
    }
    native_files = {} if components_only else native["files"]
    if not components_only and native["claude_hooks"]:
        settings = safe_destination(project, ".claude/settings.json")
        settings_data = json.loads(settings.read_text()) if settings.exists() else {}
        receipt.configuration(
            ".claude/settings.json", settings_data, {"hooks": native["claude_hooks"]}
        )
        merge_configuration(
            settings_data, {"hooks": native["claude_hooks"]}, replace=replace
        )
        native_files[".claude/settings.json"] = (
            json.dumps(settings_data, indent=2) + "\n"
        ).encode()
    for package, source in components.get("provenance", {}).items():
        native_files[f".clio-kit/sources/{package}.json"] = (
            json.dumps(source, indent=2) + "\n"
        ).encode()
    for relative, content in native_files.items():
        path = safe_destination(project, relative)
        if (
            path.exists()
            and path.read_bytes() != content
            and not replace
            and relative != ".claude/settings.json"
            and not relative.startswith(".clio-kit/sources/")
        ):
            raise ValueError(
                f"{relative} contains different content; review before using --replace"
            )
    if dry_run:
        return result
    if root is None:
        from clio_kit.component_store import fetch

        for artifact_key in components["artifacts"]:
            fetch(artifact_key)
    with InstallTransaction() as transaction:
        for payload_target, source in components.get("payloads", {}).items():
            safe_destination(project, str(payload_target.relative_to(project)))
            transaction.directory(payload_target, source)
        for relative, content in native_files.items():
            if relative not in receipt.data["configs"] and not relative.startswith(
                ".clio-kit/sources/"
            ):
                receipt.file(relative, content)
            transaction.file(safe_destination(project, relative), content)
        if root is None and components["artifacts"]:
            register_project(config, components["artifacts"], transaction)
        if components["skills"]:
            for skill, source in components["skills"].items():
                receipt.file(f"{skill_path}/{skill}", source)
            stage_skills(
                components["skills"], target, replace, transaction, skill_policy=False
            )
        if (
            components["servers"]
            or (not components_only and native["configuration"])
            or receipt.old.get("configs", {}).get(config_path, {}).get("owned")
        ):
            rendered = (
                tomli_w.dumps(data)
                if client == "codex"
                else json.dumps(data, indent=2) + "\n"
            ).encode()
            if original != rendered:
                if original is not None:
                    backup = config.with_name(
                        config.name + ".backup-" + uuid.uuid4().hex
                    )
                    transaction.file(backup, original)
                    result["backup"] = str(backup)
                transaction.file(config, rendered)
        receipt.stage(transaction)
        transaction.commit()
    return result
