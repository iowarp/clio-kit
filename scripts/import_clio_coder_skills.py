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
import shutil
import subprocess
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
REVISION = "c841a46101d6d9df5fd3bcb1d337e59d92fb660d"
COLLECTION = "clio-coder-skills"
REPOSITORY = "https://github.com/iowarp/clio-coder"


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
        for path in paths:
            hashes(path.parent)  # Reject linked resources before copying them.
            original = path.read_text()
            header, body = original[4:].split("\n---\n", 1)
            fields = yaml.safe_load(header)
            name = fields["name"]
            if name == "scientific-debugging":
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
            if name == "archify":
                # The upstream instruction overlay intentionally does not ship
                # its separately installed renderer's reference directory.
                body = body.replace(
                    "`references/authoring-contract.md`",
                    "`<renderer>/references/authoring-contract.md`",
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
            if not scenarios:
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
            )
            (target / "SKILL.md").write_text(
                "---\n"
                + yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True)
                + "---\n\n"
                + note
                + body.lstrip()
            )
            records.append(
                {
                    "name": name,
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
        result = {
            "schema": 1,
            "repository": REPOSITORY,
            "revision": revision,
            "adaptations": [
                "Normalized Agent Skills frontmatter; upstream metadata retained here",
                "Added host compatibility note; no foreign tool allowlist enforced",
                "Made Materio assets self-contained for individual skill installation",
                "Omitted the whole-package exporter from individual Materio skills",
                "Qualified Archify's separately installed renderer reference",
                "Corrected scientific-debugging's contradictory worked-example verdicts and tolerance guidance",
                "Recorded CLIO Kit evaluation status independently of upstream status",
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
