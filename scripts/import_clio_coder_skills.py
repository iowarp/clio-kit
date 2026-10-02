#!/usr/bin/env python3
"""Reproduce the optional Clio Coder collection from a reviewed Git revision.

An existing collection is replaced only if every imported file still matches
its recorded digest. Local adaptations therefore require deliberate review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
REVISION = "c841a46101d6d9df5fd3bcb1d337e59d92fb660d"
COLLECTION = "clio-coder-skills"
REPOSITORY = "https://github.com/iowarp/clio-coder"
SKILL_PREFIX = "clio-kit-"


def adapt_references(text: str, names: set[str]) -> str:
    """Rename skill invocations, not similarly named programs or native agents."""
    alternatives = "|".join(
        re.escape(name) for name in sorted(names, key=len, reverse=True)
    )
    text = re.sub(
        rf"(/skill[ :]|skill:)({alternatives})(?![\w-])",
        lambda match: match[1] + SKILL_PREFIX + match[2],
        text,
    )
    return re.sub(
        rf"`({alternatives})`(?= skill\b)",
        lambda match: "`" + SKILL_PREFIX + match[1] + "`",
        text,
    )


def evaluation_scenarios(text: str) -> str:
    """Keep reusable scenarios; upstream retains historical run narratives."""
    sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)
    return (
        "".join(
            section
            for section in sections
            if not re.match(
                r"## (?:Smoke record|Battletest record|Empirical Battletest|"
                r"Instruction correction|Observed|Live interactive confirmation)\b",
                section,
            )
        ).rstrip()
        + "\n"
    )


def hashes(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Linked import resource: {path}")
        if path.is_file() and path.name != "import-lock.json":
            result[path.relative_to(root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return result


def validate_shared_assets(root: Path, records: list[dict]) -> None:
    """Keep packed skills self-contained without allowing shared copies to drift."""
    groups: dict[str, tuple[str, dict]] = {}
    for record in records:
        if not record["packed"]:
            continue
        package = str(Path(record["path"]).parents[1])
        resources = hashes(root / "skills" / record["name"] / "assets")
        if package in groups and groups[package][1] != resources:
            raise ValueError(
                f"Shared assets differ: {groups[package][0]} and {record['name']}; "
                "refresh all copies through the importer"
            )
        groups[package] = (record["name"], resources)


def generate(
    source: Path,
    destination: Path,
    revision: str = REVISION,
    version: str | None = None,
) -> dict:
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != revision:
        raise ValueError(f"Expected {revision}, got {actual}")
    dirty = subprocess.check_output(
        [
            "git",
            "-C",
            str(source),
            "status",
            "--porcelain",
            "--",
            "library",
            "LICENSE",
            "NOTICE",
        ],
        text=True,
    )
    if dirty.strip():
        raise ValueError("Import source has uncommitted changes")
    if destination.exists():
        lock = json.loads((destination / "import-lock.json").read_text())
        if hashes(destination) != lock["files"]:
            raise ValueError(
                "Imported collection has local changes; review before refreshing"
            )
        if version is None:
            version = json.loads(
                (destination / ".claude-plugin/plugin.json").read_text()
            )["version"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=".import-"
    ) as temporary:
        staging = Path(temporary) / COLLECTION
        staging.mkdir()
        records = []
        paths = sorted(source.glob("library/skills/*/*/SKILL.md")) + sorted(
            source.glob("library/plugins/*/skills/*/SKILL.md")
        )
        upstream_names = {
            yaml.safe_load(path.read_text()[4:].split("\n---\n", 1)[0])["name"]
            for path in paths
        }
        for path in paths:
            hashes(path.parent)  # Reject linked resources before copying them.
            original = path.read_text()
            header, body = original[4:].split("\n---\n", 1)
            fields = yaml.safe_load(header)
            upstream_name = fields["name"]
            name = SKILL_PREFIX + upstream_name
            if upstream_name == "context-handoff":
                body = body.replace(
                    "6. **Redact.** Remove API keys, tokens, secrets, passwords, and PII unless it is\n"
                    "   genuinely part of the project. Replace with `[REDACTED]` and note what was\n"
                    "   removed.",
                    "6. **Redact.** Remove credentials, tokens, passwords and unnecessary PII from\n"
                    "   the handoff and every response about it. Replace values with `[REDACTED]`.\n"
                    "   Describe only the category removed; never repeat a value to demonstrate\n"
                    "   redaction. Check both the saved document and the final response before\n"
                    "   returning them. This also applies to synthetic test credentials.",
                )
            if upstream_name == "worktree-create":
                body = body.replace(
                    "3. Derive safe filesystem paths under `--root` for each branch:",
                    "3. Preserve an explicitly requested worktree destination exactly after path validation. "
                    "Derive a path from the branch name only when the user did not provide a destination:",
                )
            if upstream_name == "scientific-debugging":
                start, end = body.index("## Worked Example"), body.index("## Red Flags")
                body = (
                    body[:start]
                    + (
                        "## Worked Example\n\n"
                        "Report: a weighted mean changed after a reduction refactor. Preserve the failing input and inspect the current implementation before choosing a fix.\n\n"
                        "- Goal: match an independently computed reference within a predeclared absolute tolerance on cancellation-heavy inputs.\n"
                        "- H1 (numerics): sequential floating-point addition loses small terms between large opposing terms. Refute if sequential addition and `math.fsum` agree on the failing input.\n"
                        "- H2 (data): values or weights changed. Refute by comparing input checksums and the exact arrays supplied to both implementations.\n"
                        "- H3 (environment): runtime-dependent reduction behavior explains the observation. Compare the same explicit loop and `math.fsum` under the recorded Python version; do not assume built-in `sum` uses naive accumulation.\n"
                        "- If the same input yields different sequential and compensated reductions, report that observation as evidence for H1. It does not by itself prove when the regression was introduced.\n"
                        "- Use a detached temporary checkout to compare revisions when needed; preserve the user's working tree. A clean prior revision supports a regression hypothesis rather than refuting it.\n"
                        "- Keep absolute tolerance near zero and relative tolerance away from zero explicit. Do not loosen tolerances to hide a numerical error.\n\n"
                    )
                    + body[end:]
                )
            target = staging / "skills" / name
            if target.exists():
                raise ValueError(f"Duplicate upstream skill: {name}")
            shutil.copytree(
                path.parent,
                target,
                ignore=shutil.ignore_patterns(
                    "__pycache__",
                    "*.pyc",
                    "node_modules",
                    ".claude-plugin",
                    "plugin.json",
                ),
            )
            packed = "plugins" in path.relative_to(source / "library").parts
            if upstream_name == "archify":
                # The upstream instruction overlay intentionally does not ship
                # its separately installed renderer's reference directory.
                body = body.replace(
                    "`references/authoring-contract.md`",
                    "`<renderer>/references/authoring-contract.md`",
                )
            if upstream_name == "herdr":
                body = body.replace(
                    "Control commands return JSON. Read every identifier from the response;",
                    "Commands such as `pane split` and `pane list` return JSON; `pane read` and help return plain text. Some successful actions, including `pane run`, return no output; check their exit status instead of parsing an empty response. Read every identifier from creation responses;",
                )
            if packed:
                package = path.parents[2]
                hashes(package / "assets")
                shutil.copytree(
                    package / "assets",
                    target / "assets",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
                # This helper generates whole native packages, not individual
                # skills, and requires the unshipped native component graph.
                (target / "assets/scripts/project_plugin.py").unlink(missing_ok=True)
                body = body.replace("../../assets/", "assets/")
                shutil.copy2(
                    package / "ai.iowarp.portability" / "provenance.json",
                    target / "upstream-provenance.json",
                )
            for filename in ("LICENSE", "NOTICE"):
                if (source / filename).exists() and not (target / filename).exists():
                    shutil.copy2(source / filename, target / filename)
            upstream_meta = fields.get("clio-coder", {})
            description = fields["description"]
            if not description.startswith("Use when"):
                description = "Use when this workflow is requested: " + description
            if "Triggers on" not in description:
                triggers = fields.get("triggers", [name.replace("-", " ")])
                description += (
                    " Triggers on "
                    + ", ".join(json.dumps(t) for t in triggers[:2])
                    + "."
                )
            scenarios = (target / "evals.md").exists()
            if scenarios:
                evaluations = target / "evals.md"
                evaluations.write_text(evaluation_scenarios(evaluations.read_text()))
            else:
                (target / "evals.md").write_text(
                    f"# {name} acceptance scenarios\n\n"
                    "These are pending CLIO Kit evaluations, not recorded passes.\n\n"
                    f"- Request the workflow described by {name}; verify the skill loads and follows its artifact contract.\n"
                    "- Install this skill alone and resolve each supporting resource it uses.\n"
                    "- Remove a required capability; the result must identify the missing prerequisite rather than invent success.\n"
                )
            normalized = {
                "name": name,
                "description": description,
                "compatibility": "Clio Coder procedures adapted for skill discovery. Host tools, external services and native Clio execution gates require separate configuration; see the host compatibility note.",
                "metadata": {
                    "bundle": "clio-coder",
                    "servers": "none",
                    "provenance": "adapted",
                    "eval-status": "scenarios-recorded",
                    "source": f"{REPOSITORY}/tree/{revision}/{path.parent.relative_to(source).as_posix()}",
                    "upstream-name": upstream_name,
                    "upstream-eval-status": str(
                        upstream_meta.get("eval-status", "unspecified")
                    ),
                },
            }
            if fields.get("license"):
                normalized["license"] = fields["license"]
            note = (
                "## Host compatibility\n\n"
                "This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. "
                "The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; "
                "they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. "
                "If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. "
                "A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.\n\n"
                "Adapted skills use the `clio-kit-` prefix to distinguish them from upstream audited skills. "
                "For a companion skill named below, select its `clio-kit-` copy from this collection; "
                "native agents, fleets and external programs retain their original names.\n\n"
            )
            (target / "SKILL.md").write_text(
                "---\n"
                + yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True)
                + "---\n\n"
                + note
                + adapt_references(body.lstrip(), upstream_names)
            )
            records.append(
                {
                    "name": name,
                    "upstream_name": upstream_name,
                    "path": path.parent.relative_to(source).as_posix(),
                    "upstream_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "packed": packed,
                    "original_frontmatter": fields,
                }
            )
        manifest = staging / ".claude-plugin" / "plugin.json"
        manifest.parent.mkdir()
        manifest.write_text(
            json.dumps(
                {
                    "name": COLLECTION,
                    "version": version or "1.0.0",
                    "description": "Optional Clio Coder coding and research skills, including Materio procedures. Native Clio agents and fleets require Clio Coder.",
                    "author": {"name": "IOWarp"},
                },
                indent=2,
            )
            + "\n"
        )
        # Clio Coder's native library reads the portable root manifest;
        # Claude reads the nested manifest. Both expose the same skill tree.
        portable = {
            "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
            **json.loads(manifest.read_text()),
        }
        (staging / "plugin.json").write_text(json.dumps(portable, indent=2) + "\n")
        validate_shared_assets(staging, records)
        result = {
            "schema": 1,
            "repository": REPOSITORY,
            "revision": revision,
            "adaptations": [
                "Normalized Agent Skills frontmatter; upstream metadata retained here",
                "Namespaced adapted skills and invocations with clio-kit- to avoid upstream audit identity collisions",
                "Added host compatibility note; no foreign tool allowlist enforced",
                "Made Materio assets self-contained for individual skill installation",
                "Omitted the whole-package exporter from individual Materio skills",
                "Qualified Archify's separately installed renderer reference",
                "Corrected Herdr guidance for successful actions with empty stdout",
                "Corrected scientific-debugging's contradictory worked-example verdicts and tolerance guidance",
                "Clarified handoff redaction applies to saved documents and final responses",
                "Preserved explicit worktree destinations ahead of branch-derived defaults",
                "Recorded CLIO Kit evaluation status independently of upstream status",
                "Retained evaluation scenarios; historical run narratives remain in the pinned upstream source",
            ],
            "skills": records,
            "files": hashes(staging),
        }
        (staging / "import-lock.json").write_text(json.dumps(result, indent=2) + "\n")
        if destination.exists():
            backup = Path(temporary) / "previous"
            destination.rename(backup)
            try:
                staging.rename(destination)
            except OSError:
                backup.rename(destination)
                raise
        else:
            staging.rename(destination)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="Clean checkout at the pinned revision"
    )
    parser.add_argument(
        "--revision",
        default=REVISION,
        help="Explicit reviewed revision for an upstream refresh",
    )
    parser.add_argument(
        "--destination", type=Path, default=ROOT / "skills" / COLLECTION
    )
    parser.add_argument(
        "--version",
        help="Collection plugin version for a reviewed content release; otherwise preserve the existing version",
    )
    args = parser.parse_args()
    result = generate(
        args.source.resolve(), args.destination.resolve(), args.revision, args.version
    )
    print(f"Imported {len(result['skills'])} skills at {result['revision']}")
