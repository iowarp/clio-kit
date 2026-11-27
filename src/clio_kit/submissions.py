"""Publish a reviewed one-file contribution using the contributor's GitHub fork."""

from __future__ import annotations

import json
import subprocess
import tempfile
import uuid
from pathlib import Path


def open_submission(name: str, entry: str, *, target: str = "iowarp/clio-kit") -> str:
    """Create a branch in the authenticated user's fork and open its PR.

    This is called only for the explicit --open-pr option. Argument arrays and
    a body file keep contributor descriptions out of shell command evaluation.
    """

    def run(*args: str, cwd: Path | None = None) -> str:
        result = subprocess.run(
            args, cwd=cwd, text=True, capture_output=True, check=True, timeout=180
        )
        return result.stdout.strip()

    login = run("gh", "api", "user", "--jq", ".login")
    # GitHub may return an existing fork whose repository name differs from
    # upstream. Use its actual identity, and do not try to fork our own repo.
    fork = target
    if target.split("/")[0].casefold() != login.casefold():
        fork = run(
            "gh",
            "api",
            "--method",
            "POST",
            f"repos/{target}/forks",
            "--jq",
            ".full_name",
        )
    base = json.loads(run("gh", "repo", "view", target, "--json", "defaultBranchRef"))[
        "defaultBranchRef"
    ]["name"]
    branch = f"contribute/{name}-{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory(prefix="clio-submit-") as temporary:
        checkout = Path(temporary) / "checkout"
        run("gh", "repo", "clone", fork, str(checkout))
        # Existing forks can lag or contain unrelated commits. Base the one-file
        # contribution on upstream, while pushing only to the contributor's fork.
        run(
            "git",
            "fetch",
            "--",
            f"https://github.com/{target}.git",
            base,
            cwd=checkout,
        )
        run("git", "checkout", "-b", branch, "FETCH_HEAD", cwd=checkout)
        destination = checkout / "community" / "entries" / f"{name}.toml"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ValueError(
                f"{name} already exists; submit an explicit update instead"
            )
        destination.write_text(entry)
        run("git", "add", "--", str(destination), cwd=checkout)
        run("git", "commit", "-m", f"feat(community): index {name}", cwd=checkout)
        run("git", "push", "origin", branch, cwd=checkout)
        body = Path(temporary) / "body.md"
        body.write_text(
            f"Index `{name}` as an external contribution.\n\nOnly its marketplace entry is added; implementation and releases remain with the contributor.\n\nThe local contribution passed CLIO structural validation. Review the source and declared components before accepting.\n"
        )
        return run(
            "gh",
            "pr",
            "create",
            "--repo",
            target,
            "--base",
            base,
            "--head",
            f"{login}:{branch}",
            "--title",
            f"Index {name}",
            "--body-file",
            str(body),
            cwd=checkout,
        )
