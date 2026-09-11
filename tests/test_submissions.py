"""Exercise real contribution Git operations without opening a public test PR."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from clio_kit.submissions import open_submission


def test_submission_pushes_only_the_entry_and_preserves_pr_body(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "test@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "test@example.invalid")
    original_run = subprocess.run
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    original_run(
        ["git", "init", "-b", "main", str(upstream)], check=True, capture_output=True
    )
    (upstream / "README.md").write_text("Contribution test repository\n")
    original_run(["git", "-C", str(upstream), "add", "."], check=True)
    original_run(
        ["git", "-C", str(upstream), "commit", "-m", "Initial"],
        check=True,
        capture_output=True,
    )
    fork = tmp_path / "fork.git"
    original_run(
        ["git", "clone", "--bare", str(upstream), str(fork)],
        check=True,
        capture_output=True,
    )
    (upstream / "README.md").write_text("Upstream advanced after the fork\n")
    original_run(
        ["git", "-C", str(upstream), "commit", "-am", "Advance upstream"],
        check=True,
        capture_output=True,
    )
    upstream_head = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    captured = {}

    def run(args, **kwargs):
        if args[0] != "gh":
            args = tuple(
                str(upstream)
                if arg == "https://github.com/iowarp/clio-kit.git"
                else arg
                for arg in args
            )
            return original_run(args, **kwargs)
        if args[1:3] == ("api", "user"):
            output = "contributor"
        elif args[1:3] == ("repo", "fork"):
            output = ""
        elif args[1:3] == ("repo", "clone"):
            return original_run(["git", "clone", str(fork), args[4]], **kwargs)
        elif args[1:3] == ("repo", "view"):
            output = json.dumps({"defaultBranchRef": {"name": "main"}})
        elif args[1:3] == ("pr", "create"):
            captured["args"] = args
            captured["body"] = Path(args[args.index("--body-file") + 1]).read_text()
            output = "https://example.invalid/pull/1"
        else:
            raise AssertionError(args)
        return subprocess.CompletedProcess(args, 0, stdout=output, stderr="")

    monkeypatch.setattr("clio_kit.submissions.subprocess.run", run)
    entry = 'name = "crystal"\ndescription = "A quoted \\"value\\""\n'
    assert open_submission("crystal", entry) == "https://example.invalid/pull/1"
    args = captured["args"]
    branch = args[args.index("--head") + 1].split(":", 1)[1]
    changed = subprocess.check_output(
        ["git", "--git-dir", str(fork), "diff", "--name-only", upstream_head, branch],
        text=True,
    ).splitlines()
    assert changed == ["community/entries/crystal.toml"]
    stored = subprocess.check_output(
        [
            "git",
            "--git-dir",
            str(fork),
            "show",
            f"{branch}:community/entries/crystal.toml",
        ],
        text=True,
    )
    assert stored == entry
    assert "\n\nOnly its marketplace entry" in captured["body"]
    assert args[args.index("--base") + 1] == "main"
