#!/usr/bin/env python3
"""Live installed-planner regression cases; semantic outcomes require review.

Uses an existing Claude sign-in and temporary profiles. No scientific operations
are delegated to the model. An optional saved fixture reproduces a prior failure
without changing its evidence. Outputs and authentication links stay outside Git.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
AGENT = "scientific-workflow-planner"
CASES = ("missing-reader", "preview-only", "available-reader")
READER = '''"""Compute the mean of a named CSV column without retaining its rows."""
import csv
import json
import math
import sys

count = 0
with open(sys.argv[1], newline="") as stream:
    rows = csv.DictReader(stream)
    if sys.argv[2] not in (rows.fieldnames or []):
        raise ValueError("Unknown column")
    def values():
        global count
        for row in rows:
            value = float(row[sys.argv[2]])
            if not math.isfinite(value):
                raise ValueError("Non-finite value")
            count += 1
            yield value
    total = math.fsum(values())
if not count:
    raise ValueError("Empty column")
print(json.dumps({"count": count, "mean": total / count, "coverage": "all rows"}))
'''


def hashes(folder):
    return {
        str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in folder.rglob("*")
        if p.is_file()
    }


def prepare(project: Path, case: str):
    project.mkdir()
    if case == "available-reader":
        (project / "readings.csv").write_text(
            "time_s,signal_mV\n0,2\n1,4\n2,8\n3,16\n4,32\n5,64\n"
        )
        (project / "column_mean.py").write_text(READER)
        result = subprocess.run(
            [
                sys.executable,
                str(project / "column_mean.py"),
                str(project / "readings.csv"),
                "signal_mV",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        assert json.loads(result.stdout) == {
            "count": 6,
            "mean": 21.0,
            "coverage": "all rows",
        }
        (project / "environment.json").write_text(
            json.dumps(
                {
                    "python": sys.executable,
                    "helper": "column_mean.py",
                    "verification": "Helper executed successfully on this CSV; independently checked all rows and column selection.",
                    "units": {"time_s": "s", "signal_mV": "mV"},
                }
            )
        )
        objective = (
            "Plan how to obtain the full-data mean of signal_mV in readings.csv, "
            "using the supplied local capabilities. Memory budget is 512 MiB. "
            "Do not perform the calculation or modify files."
        )
    else:
        source = ROOT / "mcp-servers/hdf5/src/hdf5_mcp/server.py"
        tree = ast.parse(source.read_text())
        names = {
            "open_file",
            "close_file",
            "get_shape",
            "get_dtype",
            "read_partial_dataset",
            "read_full_dataset",
            "hdf5_aggregate_stats",
        }
        (project / "tool-implementations.py").write_text(
            "\n\n".join(
                ast.get_source_segment(source.read_text(), node)
                for node in tree.body
                if isinstance(node, ast.AsyncFunctionDef) and node.name in names
            )
        )
        shutil.copy2(
            ROOT / "mcp-servers/hdf5/src/hdf5_mcp/statistics.py",
            project / "aggregate-contract.py",
        )
        for name in ("dataset-explore", "large-data-read"):
            shutil.copy2(
                ROOT / f"skills/clio-scientific-io-skills/skills/{name}/SKILL.md",
                project / f"{name}.md",
            )
        rows = 70_000_000 if case == "missing-reader" else 200
        (project / "layout.json").write_text(
            json.dumps(
                {
                    "dataset": "/readings",
                    "shape": [rows, 2],
                    "dtype": "float64",
                    "columns": ["time_s", "signal_mV"],
                    "units": ["s", "mV"],
                    "source": "Fixture metadata; target filesystem path and live connection not supplied",
                    "bytes": rows * 2 * 8,
                    "MiB": rows * 2 * 8 / 2**20,
                    "budget_bytes": 512 * 2**20,
                }
            )
        )
        objective = (
            "Plan a bounded preview and exact full-data mean of the signal_mV "
            "column in /readings if feasible under a 512 MiB process memory budget. "
            "Use the supplied local documentation and implementations."
            if case == "missing-reader"
            else "Plan to export all 200 rows of /readings to a CSV and check whether "
            "every adjacent time_s value increases, using read_partial_dataset. "
            "Use the supplied local documentation and implementations."
        )
    (project / "objective.md").write_text(objective + "\n")


def run(args, case, repeat):
    folder = args.output / f"{case}-{repeat}"
    folder.mkdir()
    project = folder / "project"
    if case == "reproducer":
        shutil.copytree(args.fixture, project)
    else:
        prepare(project, case)
    before = hashes(project)
    profile = folder / "profile"
    profile.mkdir()
    env = dict(
        os.environ,
        CLAUDE_CONFIG_DIR=str(profile),
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
    )
    for key in list(env):
        if (
            key.startswith(("ANTHROPIC_", "CLAUDE_CODE_USE_"))
            or key == "CLAUDE_CODE_OAUTH_TOKEN"
        ):
            env.pop(key)

    def command(label, argv):
        try:
            result = subprocess.run(
                argv,
                cwd=project,
                env=env,
                text=True,
                capture_output=True,
                timeout=args.timeout,
            )
        except subprocess.TimeoutExpired as error:
            (folder / f"{label}.log").write_bytes(
                (error.stdout or b"") + (error.stderr or b"")
            )
            raise
        (folder / f"{label}.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"{label} failed; see {folder}")
        return result.stdout

    auth = profile / ".credentials.json"
    installed = False
    try:
        command(
            "marketplace-add", ["claude", "plugin", "marketplace", "add", str(ROOT)]
        )
        command("install", ["claude", "plugin", "install", "clio-agents@clio-kit"])
        installed = True
        entries = json.loads(
            command("installed", ["claude", "plugin", "list", "--json"])
        )
        item = next(e for e in entries if e["id"] == "clio-agents@clio-kit")
        agent_path = Path(item["installPath"]) / f"agents/{AGENT}.md"
        assert (
            agent_path.read_bytes()
            == (ROOT / f"plugins/clio-agents/agents/{AGENT}.md").read_bytes()
        )
        agent_digest = hashlib.sha256(agent_path.read_bytes()).hexdigest()
        metadata = yaml.safe_load(agent_path.read_text().split("---", 2)[1])
        effort = metadata.get("effort")
        model = args.model or metadata.get("model")
        model_args = ["--model", model] if model and model != "inherit" else []
        # --agent selects a main-session persona; this client does not apply
        # its subagent effort field there. Match the installed field explicitly.
        effort_args = ["--effort", effort] if effort else []
        original = (
            Path(os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude")))
            / ".credentials.json"
        )
        if not original.is_file():
            raise RuntimeError("Claude sign-in credentials are unavailable")
        auth.symlink_to(original)
        prompt = (
            "Read objective.md and the supplied local evidence and documentation. "
            "Return a bounded scientific workflow plan with concrete handoffs, "
            "prerequisites, artifacts and validation. Do not run it. "
            "Use only read-only tools and stay within this project."
        )
        (folder / "request.txt").write_text(prompt)
        log = command(
            "runtime",
            [
                "claude",
                "-p",
                "--verbose",
                "--output-format",
                "stream-json",
                *model_args,
                *effort_args,
                "--setting-sources",
                "user",
                "--agent",
                f"clio-agents:{AGENT}",
                "--strict-mcp-config",
                "--mcp-config",
                '{"mcpServers":{}}',
                "--tools",
                "Read,Glob,Grep",
                "--permission-mode",
                "bypassPermissions",
                "--no-session-persistence",
                "--",
                prompt,
            ],
        )
        events = [json.loads(line) for line in log.splitlines() if line.startswith("{")]
        final = next(e for e in reversed(events) if e.get("type") == "result")
        assert not final.get("is_error"), "Model did not complete successfully"
        uses = [
            b
            for e in events
            if e.get("type") == "assistant"
            for b in e.get("message", {}).get("content", [])
            if b.get("type") == "tool_use"
        ]
        assert any(u["name"] == "Read" for u in uses)
        assert all(u["name"] in {"Read", "Glob", "Grep"} for u in uses)
        assert hashes(project) == before, "Input evidence changed"
        (folder / "answer.md").write_text(final["result"])
        record = {
            "case": case,
            "repeat": repeat,
            "model": model or "client default",
            "actual_models": sorted(
                {
                    event["message"]["model"]
                    for event in events
                    if event.get("type") == "assistant"
                    and event.get("message", {}).get("model")
                }
            ),
            "effort": effort or "client default",
            "completed": True,
            "inputs_unchanged": True,
            "tools": [u["name"] for u in uses],
            "input_hashes": before,
            "agent_sha256": agent_digest,
            "usage": final.get("usage"),
            "quality": "requires independent review",
        }
        (folder / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"Completed {case}-{repeat}; review answer.md", flush=True)
        return record
    finally:
        if auth.is_symlink():
            auth.unlink()
        if installed:
            command(
                "uninstall", ["claude", "plugin", "uninstall", "clio-agents@clio-kit"]
            )
            command(
                "marketplace-remove",
                ["claude", "plugin", "marketplace", "remove", "clio-kit"],
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument(
        "--fixture", type=Path, help="Also repeat a trusted saved fixture directory"
    )
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument(
        "--model", help="Override the installed agent's model for comparison"
    )
    args = parser.parse_args()
    if not 1 <= args.repeat <= 3 or not 1 <= args.workers <= 2 or args.timeout < 1:
        parser.error("Use 1..3 repeats, 1..2 workers and a positive timeout")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = args.case or list(CASES)
    if args.fixture:
        args.fixture = args.fixture.resolve()
        if not (args.fixture / "objective.md").is_file():
            parser.error("Saved fixture must contain objective.md")
        cases = ["reproducer", *cases]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [
            pool.submit(run, args, case, n)
            for case in cases
            for n in range(1, args.repeat + 1)
        ]
        results = [job.result() for job in jobs]
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
