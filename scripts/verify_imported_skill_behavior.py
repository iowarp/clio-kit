#!/usr/bin/env python3
"""Run a real Codex skill/MCP composition against a controlled numerical bug.

Requires Codex model access and clio-kit[verification]. Uses a fresh project;
preserves the user's authentication/configuration and records execution evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from clio_kit.skill_cli import install_skills, selected_skills
from verify_external_contributions import SERVER

REQUEST = """Use clio-kit-scientific-debugging to diagnose the failing weighted mean reference test in this directory.
Do not edit calculation.py or test_calculation.py. Inspect the installed skill.
Run python3 -m unittest -q, investigate at least two fault classes, and compare
the cancellation input with the configured numerics MCP weighted_mean tool.
Write diagnosis.md with commands, observations, and the cause. Then use
clio-kit-experiment-protocol to write a validation protocol in VALIDATION.md for comparing
this implementation to a corrected one; record numerical tolerance and input
identity before performing any new timing measurement. Inspect both skills.
Do not push, contact external services other than the configured MCP, or change
user settings. Work only in this temporary project and its installed skills.
"""


def main(output: Path, sandbox: str) -> bool:
    output.mkdir(parents=True, exist_ok=False)
    project = output / "project"
    project.mkdir()
    skills = selected_skills(
        ("clio-kit-scientific-debugging", "clio-kit-experiment-protocol"), None
    )
    install_skills(skills, project / ".agents/skills", False)
    (project / "calculation.py").write_text(
        "def weighted_mean(values, weights):\n"
        "    total = 0.0\n"
        "    for value, weight in zip(values, weights):\n"
        "        total += value * weight\n"
        "    return total / sum(weights)\n"
    )
    (project / "test_calculation.py").write_text(
        "import unittest, math\nfrom calculation import weighted_mean\n"
        "class Reference(unittest.TestCase):\n"
        " def test_cancellation(self):\n"
        "  values=[1e16, 6e-4, -1e16, 1e-4, 2e-4, -3e-4]\n"
        "  self.assertAlmostEqual(weighted_mean(values,[1]*6), math.fsum(values)/6, delta=1e-12)\n"
    )
    original = {
        name: (project / name).read_bytes()
        for name in ("calculation.py", "test_calculation.py")
    }
    baseline = subprocess.run(
        [sys.executable, "-m", "unittest", "-q"],
        cwd=project,
        capture_output=True,
        text=True,
    )
    (output / "baseline.log").write_text(baseline.stdout + baseline.stderr)
    assert baseline.returncode != 0 and "FAIL: test_cancellation" in baseline.stderr
    home = output / "codex"
    home.mkdir()
    user_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    if (user_home / "auth.json").is_file():
        (home / "auth.json").symlink_to(user_home / "auth.json")
    (output / "server.py").write_text(SERVER)
    (home / "config.toml").write_text(
        "[mcp_servers.numerics]\ncommand = "
        + json.dumps(sys.executable)
        + "\nargs = ["
        + json.dumps(str(output / "server.py"))
        + "]\n"
    )
    (output / "request.txt").write_text(REQUEST)
    command = [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--sandbox",
        sandbox,
        "--json",
        "--output-last-message",
        str(output / "answer.md"),
        "-",
    ]
    try:
        with (
            (output / "trace.jsonl").open("w") as trace,
            (output / "stderr.log").open("w") as errors,
        ):
            completed = subprocess.run(
                command,
                input=REQUEST,
                text=True,
                cwd=project,
                env={**os.environ, "CODEX_HOME": str(home)},
                stdout=trace,
                stderr=errors,
                timeout=600,
            )
        events = [
            json.loads(line)
            for line in (output / "trace.jsonl").read_text().splitlines()
            if line.strip()
        ]
        items = [e["item"] for e in events if e.get("type") == "item.completed"]
        queries = [
            i
            for i in items
            if i.get("type") == "mcp_tool_call"
            and i.get("server") == "numerics"
            and i.get("tool") == "weighted_mean"
            and not i.get("error")
        ]
        commands = "\n".join(i.get("command", "") for i in items)
        checks = {
            "client_exit": completed.returncode == 0,
            "both_skills_read": all(
                name in commands and "SKILL.md" in commands for name in skills
            ),
            "actual_mcp_query": bool(queries),
            "sources_preserved": all(
                (project / name).read_bytes() == content
                for name, content in original.items()
            ),
            "diagnosis_artifact": (project / "diagnosis.md").is_file(),
            "protocol_artifact": (project / "VALIDATION.md").is_file(),
        }
        passed = all(checks.values())
        result = {
            "status": "passed" if passed else "failed",
            "checks": checks,
            "input_sha256": {
                name: hashlib.sha256(content).hexdigest()
                for name, content in original.items()
            },
            "mcp_queries": queries,
            "review": "Inspect diagnosis.md and VALIDATION.md for scientific correctness; this gate verifies execution and artifacts, not all skill behavior.",
        }
    except subprocess.TimeoutExpired:
        passed = False
        result = {
            "status": "blocked",
            "reason": "Codex exceeded the 600-second evaluation timeout",
        }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['status']}: evidence in {output}")
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sandbox",
        choices=["workspace-write", "danger-full-access"],
        default="workspace-write",
        help="Use danger-full-access only in an already isolated test environment without working sandbox support",
    )
    args = parser.parse_args()
    raise SystemExit(0 if main(args.output.resolve(), args.sandbox) else 1)
