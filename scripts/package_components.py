"""Build deterministic, separately downloadable component archives and metadata.

This runs in Hatch's isolated build environment, without importing the launcher.
The sdist carries the index, not component payloads; rebuilding its wheel is offline.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
from pathlib import Path
import subprocess
import tarfile

import yaml

try:
    import tomllib
except ImportError:
    import tomli as tomllib

EXCLUDED = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".uv-cache",
    ".virtualenv-app-data",
    ".benchmarks",
    ".github",
    ".coverage",
    "coverage.xml",
    "htmlcov",
    "tests",
    ".env",
    ".clio-coder",
    ".DS_Store",
    "junit.xml",
}


def files_in(root: Path, directory: Path, excluded: set[str]) -> list[Path]:
    """Ignore developer build/cache files; refuse linked package content."""
    if (root / ".git").exists():
        result = subprocess.run(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                str(directory.relative_to(root)),
            ],
            cwd=root,
            capture_output=True,
            check=True,
        )
        candidates = [
            root / name for name in result.stdout.decode().split("\0") if name
        ]
    else:
        candidates = []
        for current, directories, filenames in os.walk(directory, followlinks=False):
            for name in directories:
                if name not in excluded and (Path(current) / name).is_symlink():
                    raise ValueError(
                        f"Linked component content: {Path(current) / name}"
                    )
            directories[:] = [name for name in directories if name not in excluded]
            candidates.extend(Path(current) / name for name in filenames)
    selected = []
    for path in sorted(set(candidates)):
        relative = path.relative_to(directory)
        if set(relative.parts) & excluded or path.suffix == ".pyc":
            continue
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root):
            raise ValueError(f"Linked component content: {path}")
        if path.is_file():
            selected.append(path)
    return selected


# String-list fields of a [prerequisites.<server>] table; see clio_kit.doctor.
PREREQUISITE_LISTS = (
    "executables",
    "executable-environment",
    "executable-paths",
    "environment",
    "environment-files",
)


def build_components(root: Path, output: Path) -> dict:
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    inventory_path = root / "mcp-server-versions.toml"
    prerequisites = (
        tomllib.loads(inventory_path.read_text()).get("prerequisites", {})
        if inventory_path.is_file()
        else {}
    )
    if not isinstance(prerequisites, dict):
        raise ValueError("prerequisites must be a table")
    for name, checks in prerequisites.items():
        if not isinstance(checks, dict) or set(checks) - {
            *PREREQUISITE_LISTS,
            "note",
        }:
            raise ValueError(f"Invalid prerequisites for {name}")
        for field in PREREQUISITE_LISTS:
            values = checks.get(field, [])
            if not isinstance(values, list) or not all(
                isinstance(v, str) and v.strip() for v in values
            ):
                raise ValueError(f"Invalid {field} prerequisites for {name}")
        if "note" in checks and not isinstance(checks["note"], str):
            raise ValueError(f"Invalid prerequisite note for {name}")
    index: dict = {
        "schema": 1,
        "version": version,
        "base_url": f"https://github.com/iowarp/clio-kit/releases/download/v{version}",
        "artifacts": {},
        "servers": {},
        "skills": {},
        "packages": {},
        "prompts": {},
    }
    output.mkdir(parents=True, exist_ok=True)

    def archive(
        key: str,
        directory: Path,
        excluded: set[str] | None = None,
        selected: list[Path] | None = None,
    ) -> str:
        paths = (
            selected
            if selected is not None
            else files_in(root, directory, EXCLUDED | (excluded or set()))
        )
        if not (directory / "LICENSE").is_file() and (root / "LICENSE").is_file():
            paths = [*paths, root / "LICENSE"]
        files = {}
        packed = io.BytesIO()
        with gzip.GzipFile(
            fileobj=packed, mode="wb", filename="", mtime=0
        ) as compressed:
            with tarfile.open(
                fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT
            ) as tar:
                for path in paths:
                    name = (
                        "LICENSE"
                        if path == root / "LICENSE"
                        else path.relative_to(directory).as_posix()
                    )
                    content = path.read_bytes()
                    mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
                    member = tarfile.TarInfo(name)
                    member.size, member.mode = len(content), mode
                    tar.addfile(member, io.BytesIO(content))
                    files[name] = {
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "size": len(content),
                        "mode": mode,
                    }
        content = packed.getvalue()
        digest = hashlib.sha256(content).hexdigest()
        filename = f"clio-component-{digest}.tar.gz"
        (output / filename).write_bytes(content)
        index["artifacts"][key] = {
            "file": filename,
            "sha256": digest,
            "size": len(content),
            "files": files,
        }
        return key

    for descriptor in sorted((root / "mcp-servers").glob("*/clio-server.toml")):
        data = tomllib.loads(descriptor.read_text())
        directory = descriptor.parent
        name = data.get("name")
        if not isinstance(name, str) or not re.fullmatch(
            r"[a-z0-9]+(?:-[a-z0-9]+)*", name
        ):
            raise ValueError(f"Invalid server name: {name!r}")
        if name in index["servers"]:
            raise ValueError(f"Duplicate server: {name}")
        lock = {"python": "uv.lock", "node": "package-lock.json", "go": "go.sum"}[
            data["runtime"]
        ]
        if not (directory / lock).is_file():
            raise ValueError(f"Missing runtime lock: {directory / lock}")
        excluded = (
            {"dist", "build"}
            if data["runtime"] == "python"
            else {"bin"}
            if data["runtime"] == "go"
            else set()
        )
        key = archive(f"server/{data['name']}", directory, excluded)
        scope = (
            json.loads((directory / "server.json").read_text()).get(
                "scope", "scientific"
            )
            if (directory / "server.json").is_file()
            else "scientific"
        )
        index["servers"][data["name"]] = {
            **data,
            "directory": directory.name,
            "scope": scope,
            "prerequisites": prerequisites.get(name, {}),
            "artifact": key,
        }

    if set(prerequisites) - index["servers"].keys():
        raise ValueError("Prerequisites refer to unknown servers")

    # Discover folders directly so a new contribution never needs a manual index edit.
    for kind in ("plugins", "skills", "agents", "hooks"):
        for manifest_path in sorted((root / kind).glob("*/.claude-plugin/plugin.json")):
            directory = manifest_path.parent.parent
            manifest = json.loads(manifest_path.read_text())
            name = manifest["name"]
            if name != directory.name or not re.fullmatch(
                r"[a-z0-9]+(?:-[a-z0-9]+)*", name
            ):
                raise ValueError(f"Invalid package name: {name}")
            if name in index["packages"]:
                raise ValueError(f"Duplicate package: {name}")
            skill_names = []
            for path in sorted((directory / "skills").glob("*/SKILL.md")):
                fields = yaml.safe_load(path.read_text().split("---", 2)[1])
                skill = fields["name"]
                if skill != path.parent.name or not re.fullmatch(
                    r"[a-z0-9]+(?:-[a-z0-9]+)*", skill
                ):
                    raise ValueError(f"Invalid skill name: {skill}")
                if skill in index["skills"]:
                    raise ValueError(f"Duplicate skill: {skill}")
                metadata = fields.get("metadata", fields.get("clio-kit", {}))
                extra_metadata = {
                    key: str(value)
                    for key, value in metadata.items()
                    if key in {"provenance", "eval-status"}
                }
                index["skills"][skill] = {
                    **extra_metadata,
                    "name": skill,
                    "description": fields["description"],
                    "bundle": metadata.get("bundle", name),
                    "servers": metadata.get("servers", "unspecified"),
                    "artifact": archive(f"skill/{skill}", path.parent),
                    "package": name,
                }
                skill_names.append(skill)
            configurations = []
            if (directory / ".mcp.json").is_file():
                configurations.append(json.loads((directory / ".mcp.json").read_text()))
            inline = manifest.get("mcpServers")
            if isinstance(inline, str):
                target = directory / inline
                if (
                    not inline.startswith("./")
                    or ".." in Path(inline).parts
                    or not target.resolve().is_relative_to(directory.resolve())
                ):
                    raise ValueError(
                        f"MCP configuration must stay inside its package: {inline}"
                    )
                if target.is_symlink() or any(
                    parent.is_symlink() for parent in target.parents if parent != root
                ):
                    raise ValueError(f"Linked MCP configuration: {inline}")
                inline = json.loads(target.read_text())
            if inline:
                configurations.append(inline)
            servers: dict = {}
            for configuration in configurations:
                for server, settings in configuration.get(
                    "mcpServers", configuration
                ).items():
                    if server in servers and servers[server] != settings:
                        raise ValueError(f"Conflicting MCP definitions: {server}")
                    servers[server] = settings
            unsupported = [
                field
                for field in ("agents", "commands", "hooks")
                if manifest.get(field) or any((directory / field).glob("*"))
            ]
            index["packages"][name] = {
                "manifest": manifest,
                "skills": skill_names,
                "servers": servers,
                "unsupported": unsupported,
                "artifact": archive(f"package/{name}", directory, {"skills"}),
            }
    marketplace = root / ".claude-plugin/marketplace.json"
    index["external"] = (
        [
            entry["name"]
            for entry in json.loads(marketplace.read_text())["plugins"]
            if not isinstance(entry["source"], str)
        ]
        if marketplace.exists()
        else []
    )
    for path in sorted((root / "prompts").rglob("*.md")):
        name = path.relative_to(root / "prompts").with_suffix("").as_posix()
        index["prompts"][name] = archive(f"prompt/{name}", path.parent, selected=[path])
    # Rebuilds may share an output directory; never publish obsolete payloads.
    expected = {record["file"] for record in index["artifacts"].values()}
    for old in output.glob("clio-component-*.tar.gz"):
        if old.name not in expected:
            old.unlink()
    (output / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    return index
