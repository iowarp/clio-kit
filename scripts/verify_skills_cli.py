#!/usr/bin/env python3
"""Exercise the optional skills CLI against CLIO's real repository layout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SKILLS_VERSION = "1.5.25"


def contents(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def verify(cli: str) -> None:
    executable = shutil.which(cli)
    if executable is None:
        raise RuntimeError(
            f"Install skills@{SKILLS_VERSION} or pass --cli /path/to/skills"
        )
    environment = dict(os.environ, DISABLE_TELEMETRY="1")
    with tempfile.TemporaryDirectory(prefix="clio-skills-cli-") as temporary:
        workspace = Path(temporary)
        project = workspace / "project"
        project.mkdir()

        def run(*args: str, cwd: Path = project) -> str:
            result = subprocess.run(
                [executable, *args],
                cwd=cwd,
                env=environment,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            return result.stdout

        assert run("--version").strip() == SKILLS_VERSION
        sources = {
            p.parent.name: p.parent for p in ROOT.glob("skills/*/skills/*/SKILL.md")
        }
        assert sources, "No canonical skills found"
        listing = run("add", str(ROOT), "--list")
        assert all(name in listing for name in sources), "Discovery missed a CLIO skill"
        run(
            "add",
            str(ROOT),
            "--skill",
            "*",
            "--agent",
            "codex",
            "claude-code",
            "antigravity",
            "--copy",
            "--yes",
        )
        for relative in (".agents/skills", ".claude/skills"):
            target = project / relative
            installed = {p.parent.name for p in target.glob("*/SKILL.md")}
            assert installed == sources.keys(), (relative, installed)
            for name, source in sources.items():
                assert contents(target / name) == contents(source), (relative, name)
        records = json.loads(run("list", "--json"))
        assert {record["name"] for record in records} == sources.keys()
        assert (
            set(json.loads((project / "skills-lock.json").read_text())["skills"])
            == sources.keys()
        )
        print(
            f"PASS: discovered and copied {len(sources)} complete skills for three agent targets"
        )

        # Exercise an actual source update without changing repository files.
        update_source = workspace / "source.git"
        name = "choosing-a-storage-format"
        shutil.copytree(sources[name], update_source / "skills" / name)

        def git(*args: str) -> None:
            subprocess.run(
                ["git", *args],
                cwd=update_source,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )

        def commit() -> None:
            git("add", ".")
            git(
                "-c",
                "user.name=CLIO Test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-m",
                "Update test skill",
            )

        git("init", "-b", "main")
        commit()
        update_project = workspace / "update-project"
        update_project.mkdir()
        run(
            "add",
            update_source.as_uri(),
            "--skill",
            name,
            "--agent",
            "codex",
            "claude-code",
            "--copy",
            "--yes",
            cwd=update_project,
        )
        reference = update_source / "skills" / name / "update-check.txt"
        reference.write_text(
            "An updated supporting file must reach the installed skill.\n"
        )
        commit()
        # Repeat explicit agent selection: upstream `update` can leave copied
        # directories for agents other than the detected agent stale.
        run(
            "add",
            update_source.as_uri(),
            "--skill",
            name,
            "--agent",
            "codex",
            "claude-code",
            "--copy",
            "--yes",
            cwd=update_project,
        )
        for relative in (".agents/skills", ".claude/skills"):
            assert contents(update_project / relative / name) == contents(
                reference.parent
            )
        print("PASS: explicit refresh copied changed supporting files to both targets")

        # Named removal without agent filters also removes the shared .agents copy.
        for directory, names in ((project, sorted(sources)), (update_project, [name])):
            run("remove", *names, "--yes", cwd=directory)
            for relative in (".agents/skills", ".claude/skills"):
                assert not list((directory / relative).glob("*/SKILL.md")), relative
            assert (
                json.loads((directory / "skills-lock.json").read_text())["skills"] == {}
            )
        print("PASS: named removal cleaned skill folders and lock entries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", default="skills", help="Path to the pinned skills CLI")
    verify(parser.parse_args().cli)
