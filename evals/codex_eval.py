#!/usr/bin/env python3
"""Paired live-model evaluation with isolated skills, actual tools and token records.

Uses saved Codex ChatGPT authentication without copying credentials. No public
writes. Each task has a fresh workspace and a baseline with the same tools and
prompt. Results stay in an explicitly selected directory, never release docs.
A passed execution check is not a scientific-quality certification.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time

import tomli_w

from clio_kit.client_install import install_for_client
from clio_kit.skill_cli import install_skills, selected_skills, skill_records
from codex_cases import CASES
from codex_fixtures import ROOT, check_artifacts, prepare
from scoring import call_failed

STOP = threading.Event()
BUDGET_LOCK = threading.Lock()
UNCACHED_USED = 0
COMMON = """Work only in this disposable research project. Use any relevant installed
skill, but do not read evals.md or evaluation harness files. Do the requested
work using real inputs and tools, preserve scientific meaning and report actual
failures. Do not push, publish, send messages, install packages, modify user
settings, read credentials, or start other agents. Read-only public literature
retrieval through the configured MCP is allowed when the task requests it.
Avoid unnecessary exploration. If a backend is unavailable, report its concrete
error and stop that part instead of inventing results. You have no interactive
operator; record material unknowns rather than silently guessing. Keep the final
answer concise and list artifacts and observed validation results.
"""


def read_events(path):
    events = []
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events


def summarize_events(events, servers=None):
    items = [e["item"] for e in events if e.get("type") == "item.completed"]
    usage = {}
    for event in events:
        if event.get("type") == "turn.completed":
            for key, value in event.get("usage", {}).items():
                if isinstance(value, int):
                    usage[key] = usage.get(key, 0) + value
    calls = [
        i
        for i in items
        if i.get("type") == "mcp_tool_call"
        and (servers is None or i.get("server") in servers)
    ]
    failed = [i for i in calls if tool_failed(i)]
    commands = [i for i in items if i.get("type") == "command_execution"]
    return dict(
        usage=usage,
        mcp_calls=len(calls),
        mcp_failed=len(failed),
        mcp_tools=[f"{i.get('server')}/{i.get('tool')}" for i in calls],
        shell_calls=len(commands),
        shell_failed=sum(i.get("exit_code", 0) != 0 for i in commands),
        completed=any(e.get("type") == "turn.completed" for e in events),
        skill_reads=[
            i.get("command", "")
            for i in commands
            if "SKILL.md" in i.get("command", "") and i.get("exit_code") == 0
        ],
    )


def tool_failed(item):
    result = item.get("result") or {}
    if item.get("error") or result.get("isError") or result.get("is_error"):
        return True
    structured = result.get("structured_content", result.get("structuredContent"))
    if isinstance(structured, dict) and call_failed(False, json.dumps(structured)):
        return True
    return any(
        call_failed(False, block.get("text", ""))
        for block in result.get("content", [])
        if block.get("type") == "text"
    )


def run(case, mode, args):
    global UNCACHED_USED
    label = case["skill"]
    folder = args.output / label / mode
    if folder.exists():
        raise ValueError(f"Output already exists: {folder}; use a fresh directory")
    project = folder / "project"
    protected = prepare(project, case)
    home = folder / "codex"
    home.mkdir()
    auth = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "auth.json"
    if not auth.is_file():
        raise ValueError(
            "Saved Codex auth.json missing; run codex login before evaluation"
        )
    (home / "auth.json").symlink_to(auth)
    installation = {}
    if case.get("package"):
        installation = install_for_client(
            ROOT, case["package"], "codex", project, components_only=True
        )
        # Both arms resolve/install identical tools; baseline removes only instructions.
        if mode == "baseline":
            shutil.rmtree(project / ".agents", ignore_errors=True)
    elif mode == "skill":
        install_skills(
            selected_skills((label,), None), project / ".agents/skills", False
        )
    # The benchmark's expected outcomes must not be visible to the model.
    for path in (project / ".agents").rglob("evals.md"):
        path.unlink()
    configured_servers = set(case["servers"]) | {
        name.removeprefix("clio-") for name in installation.get("servers", [])
    }
    servers = {
        name: {
            "command": "uv",
            "args": [
                "run",
                "--frozen",
                "--project",
                str(ROOT),
                "clio-kit",
                "mcp-server",
                name,
            ],
            "startup_timeout_sec": 120,
            "tool_timeout_sec": 90,
            "default_tools_approval_mode": "approve"
            if case["kind"] not in {"boundary"}
            else "auto",
        }
        for name in sorted(configured_servers)
    }
    # Use the same checkout-backed MCP commands in both arms, independent of PATH.
    config = {
        "model": args.model,
        "model_reasoning_effort": "low",
        "features": {"apps": False},
        "sandbox_workspace_write": {"network_access": args.network_access},
        "mcp_servers": servers,
        "projects": {str(project): {"trust_level": "trusted"}},
    }
    (home / "config.toml").write_text(tomli_w.dumps(config))
    # Package configuration has been checked above; use the common evaluation
    # config to avoid overriding it with a globally installed release launcher.
    (project / ".codex/config.toml").unlink(missing_ok=True)
    prompt = COMMON + "\nTask:\n" + case["prompt"]
    (folder / "request.txt").write_text(prompt)
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in (
            "OPENAI_API_KEY",
            "CODEX_API_KEY",
            "GH_TOKEN",
            "GITHUB_TOKEN",
            "ANTHROPIC_API_KEY",
            "CLAUDE_CODE_OAUTH_TOKEN",
        )
    }
    env["CODEX_HOME"] = str(home)
    command = [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--sandbox",
        args.sandbox,
        "--json",
        "--output-last-message",
        str(folder / "answer.md"),
        "-",
    ]
    started = time.monotonic()
    timed_out = False
    with (
        (folder / "trace.jsonl").open("w") as trace,
        (folder / "stderr.log").open("w") as errors,
    ):
        process = subprocess.Popen(
            command,
            cwd=project,
            env=env,
            stdin=subprocess.PIPE,
            stdout=trace,
            stderr=errors,
            text=True,
            start_new_session=True,
        )
        try:
            process.communicate(prompt, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    events = read_events(folder / "trace.jsonl")
    record = summarize_events(events, configured_servers)
    answer = (
        (folder / "answer.md").read_text() if (folder / "answer.md").exists() else ""
    )
    errors = "\n".join(
        str(e) for e in events if e.get("type") in {"error", "turn.failed"}
    )
    if any(
        x in errors.lower()
        for x in ("usage limit", "quota", "rate limit", "unauthorized", "refresh token")
    ):
        STOP.set()
    checks = check_artifacts(project, case, answer, protected)
    checks["model_completed"] = (
        process.returncode == 0 and record["completed"] and not timed_out
    )
    if case["kind"] in {
        "mcp",
        "plot",
        "interpolate",
        "retrieval",
        "bibliography",
        "report",
    }:
        checks["actual_mcp_call"] = record["mcp_calls"] > 0
    if mode == "skill":
        checks["skill_read"] = any(label in c for c in record["skill_reads"])
    runtime_blocked = any(
        marker in (folder / "trace.jsonl").read_text()
        for marker in ("bwrap: loopback:", "bwrap: setting up uid map:")
    )
    if runtime_blocked:
        STOP.set()
    usage = record["usage"]
    with BUDGET_LOCK:
        UNCACHED_USED += max(
            0, usage.get("input_tokens", 0) - usage.get("cached_input_tokens", 0)
        ) + usage.get("output_tokens", 0)
        if UNCACHED_USED >= args.max_uncached_tokens:
            STOP.set()
    record.update(
        skill=label,
        mode=mode,
        model=args.model,
        kind=case["kind"],
        seconds=round(time.monotonic() - started, 1),
        exit_code=process.returncode,
        timed_out=timed_out,
        sandbox=args.sandbox,
        runtime_blocked=runtime_blocked,
        execution_status="environment-blocked"
        if runtime_blocked
        else "timed-out"
        if timed_out
        else "checks-passed"
        if all(checks.values())
        else "checks-failed",
        checks=checks,
        checks_passed=all(checks.values()),
        outcome_screen_passed=all(
            value
            for key, value in checks.items()
            if key not in {"skill_read", "actual_mcp_call"}
        ),
        quality="requires independent review",
        installation=installation,
        servers=sorted(configured_servers),
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        source_hashes=protected,
        skill_sha256={
            str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (project / ".agents").rglob("SKILL.md")
        },
        errors=errors,
    )
    (folder / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                "skill": label,
                "mode": mode,
                "checks": record["checks_passed"],
                "usage": record["usage"],
                "seconds": record["seconds"],
            }
        ),
        flush=True,
    )
    return record


def pair(case, args):
    records = []
    # Alternate arm order to reduce systematic warm-cache advantage.
    modes = args.modes.split(",")
    if int(hashlib.sha256(case["skill"].encode()).hexdigest()[:2], 16) % 2:
        modes = list(reversed(modes))
    for mode in modes:
        if STOP.is_set():
            break
        records.append(run(case, mode, args))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--skill", action="append")
    parser.add_argument(
        "--modes",
        choices=("baseline,skill", "skill", "baseline"),
        default="baseline,skill",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--max-uncached-tokens",
        type=int,
        default=250000,
        help="Stop scheduling after this observed uncached input + output budget; in-flight runs may exceed it",
    )
    parser.add_argument(
        "--sandbox",
        choices=("workspace-write", "danger-full-access"),
        default="workspace-write",
        help="Full access requires explicit operator authorization or an externally isolated runner",
    )
    parser.add_argument(
        "--network-access",
        action="store_true",
        help="Allow network in the workspace sandbox when required by the host runtime",
    )
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    cases = [c for c in CASES if not args.skill or c["skill"] in args.skill]
    inventory = skill_records()
    if {c["skill"] for c in CASES} != set(inventory):
        raise ValueError(
            "Evaluation cases must cover the current complete skill inventory"
        )
    if args.skill and set(args.skill) - {c["skill"] for c in cases}:
        raise ValueError("Unknown requested case")
    if args.list:
        print(json.dumps(cases, indent=2))
        return
    if not 1 <= args.workers <= 3 or args.timeout < 1 or args.max_uncached_tokens < 1:
        raise ValueError("Use 1..3 workers and positive timeout/token budgets")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "planned.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "modes": args.modes,
                "cases": cases,
                "sandbox": args.sandbox,
                "max_uncached_tokens": args.max_uncached_tokens,
                "scope": "One case per skill; boundary/draft cases do not prove live delivery. Native Claude hooks/agents are not translated to Codex.",
            },
            indent=2,
        )
        + "\n"
    )
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(pair, c, args): c["skill"] for c in cases}
        for future in as_completed(futures):
            try:
                records.extend(future.result())
            except Exception as exc:
                records.append(
                    {
                        "skill": futures[future],
                        "harness_error": str(exc),
                        "checks_passed": False,
                    }
                )
            (args.output / "results.json").write_text(
                json.dumps(records, indent=2) + "\n"
            )
    print(
        f"Recorded {len(records)} runs for {len(cases)} requested skills. Quality needs review; see {args.output}"
    )


if __name__ == "__main__":
    main()
