#!/usr/bin/env python3
"""Install a community hook plugin and exercise Claude's real hook runtime.

By default a local scripted model endpoint makes CI deterministic without model
credentials. --live uses authenticated Claude instead. Both modes execute hooks
through the installed client, not by invoking the hook scripts directly.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import threading

import tomli_w

from clio_kit.community import read_community_entries
from clio_kit.plugins import validate_plugin

HOOK = """import json, sys
from pathlib import Path
data = json.load(sys.stdin)
root = Path(data["cwd"])
event = data.get("hook_event_name")
file = data.get("tool_input", {}).get("file_path", "")
with (root / "hook-events.jsonl").open("a") as output:
    output.write(json.dumps({"event": event, "file": file,
                            "plugin_root": str(Path(__file__).parent)}) + "\\n")
if event == "PreToolUse" and Path(file).name == "blocked.txt":
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
        "permissionDecision": "deny", "permissionDecisionReason": "HOOK_TEST_DENIED: do not retry or bypass."}}))
"""


def model_server(
    project: Path, actions: list[dict] | None = None
) -> ThreadingHTTPServer:
    if actions is None:
        actions = [
            {
                "name": "Write",
                "input": {"file_path": str(project / name), "content": "HOOK_TEST"},
            }
            for name in ("allowed.txt", "blocked.txt")
        ]

    class Endpoint(BaseHTTPRequestHandler):
        count = 0

        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if "count_tokens" in self.path:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"input_tokens": 100}')
                return
            with (project / "scripted-model-requests.jsonl").open("ab") as evidence:
                evidence.write(payload + b"\n")
            Endpoint.count += 1
            # Background client requests must not consume the main conversation's
            # scripted actions. Advance from tools present in its own history.
            previous = re.findall(rb"toolu_hook_(\d+)", payload)
            number = max((int(value) for value in previous), default=0) + 1
            if Endpoint.count > 32:
                self.send_error(500, "Unexpected retry loop")
                return
            content = (
                {
                    "type": "tool_use",
                    "id": f"toolu_hook_{number}",
                    **actions[number - 1],
                }
                if number <= len(actions)
                else {"type": "text", "text": "Hook test completed."}
            )
            message = {
                "id": f"msg_hook_{number}",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 100, "output_tokens": 0},
            }
            chunks = [{"type": "message_start", "message": message}]
            if content["type"] == "tool_use":
                chunks += [
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {**content, "input": {}},
                    },
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(content["input"]),
                        },
                    },
                ]
            else:
                chunks += [
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": content["text"]},
                    },
                ]
            chunks += [
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "tool_use"
                        if number <= len(actions)
                        else "end_turn",
                        "stop_sequence": None,
                    },
                    "usage": {"output_tokens": 25},
                },
                {"type": "message_stop"},
            ]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(
                    f"event: {chunk['type']}\ndata: {json.dumps(chunk)}\n\n".encode()
                )
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def verify(output: Path, claude: str, live: bool) -> None:
    output.mkdir(parents=True, exist_ok=False)
    publisher = output / "publisher"
    project = output / "project"
    project.mkdir()
    profile = output / "profile"
    profile.mkdir()
    env = dict(
        os.environ,
        CLAUDE_CONFIG_DIR=str(profile),
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
    )
    # Never forward real model credentials to the deterministic endpoint.
    if not live:
        for key in list(env):
            if key.startswith(("ANTHROPIC_", "CLAUDE_CODE_USE_")) or key in (
                "CLAUDE_CODE_OAUTH_TOKEN",
                "CLAUDE_CODE_API_KEY_HELPER_TTL_MS",
            ):
                env.pop(key)
    records = []

    def run(label, args, cwd=output):
        result = subprocess.run(
            args, cwd=cwd, env=env, text=True, capture_output=True, timeout=180
        )
        (output / f"{label}.log").write_text(result.stdout + result.stderr)
        records.append({"step": label, "exit": result.returncode})
        assert result.returncode == 0, f"{label}: see {output / (label + '.log')}"
        return result.stdout

    run(
        "scaffold",
        [
            sys.executable,
            "-c",
            "from clio_kit.plugins import plugin_group; plugin_group()",
            "init",
            str(publisher),
            "--name",
            "hook-check",
            "--hook",
        ],
    )
    run(
        "scaffold-client-validation",
        [claude, "plugin", "validate", str(publisher), "--strict"],
    )
    (publisher / "hook.py").write_text(HOOK)
    events = {
        event: [
            {
                **({"matcher": "Write"} if event != "SessionStart" else {}),
                "hooks": [
                    {
                        "type": "command",
                        "command": 'python3 "${CLAUDE_PLUGIN_ROOT}/hook.py"',
                        "timeout": 10,
                    }
                ],
            }
        ]
        for event in ("SessionStart", "PreToolUse", "PostToolUse")
    }
    (publisher / "hooks/hooks.json").write_text(json.dumps({"hooks": events}))
    assert validate_plugin(publisher)[1] == []
    for label, args in [
        ("git-init", ["git", "init", "-q"]),
        ("git-add", ["git", "add", "."]),
        (
            "git-commit",
            [
                "git",
                "-c",
                "user.name=Hook test",
                "-c",
                "user.email=test@localhost",
                "commit",
                "-qm",
                "Hook fixture",
            ],
        ),
    ]:
        run(label, args, publisher)
    catalog = output / "catalogue"
    (catalog / "community/entries").mkdir(parents=True)
    (catalog / "community/entries/hook-check.toml").write_text(
        tomli_w.dumps(
            {
                "name": "hook-check",
                "description": "Temporary hook test",
                "source": {"type": "url", "url": publisher.as_uri()},
            }
        )
    )
    (catalog / ".claude-plugin").mkdir()
    (catalog / ".claude-plugin/marketplace.json").write_text(
        json.dumps(
            {
                "name": "hook-test",
                "owner": {"name": "IOWarp"},
                "metadata": {"description": "Hook acceptance"},
                "plugins": read_community_entries(catalog),
            }
        )
    )
    run("marketplace-add", [claude, "plugin", "marketplace", "add", str(catalog)])
    run("install", [claude, "plugin", "install", "hook-check@hook-test"])
    server = None
    auth = profile / ".credentials.json"
    try:
        if live:
            original = (
                Path(os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude")))
                / ".credentials.json"
            )
            if original.exists():
                auth.symlink_to(original)
        else:
            server = model_server(project)
            env.update(
                ANTHROPIC_BASE_URL=f"http://127.0.0.1:{server.server_port}",
                ANTHROPIC_API_KEY="local-hook-test-only",
            )
        request = "Use Write to create allowed.txt containing HOOK_TEST. Then attempt exactly once to Write HOOK_TEST to blocked.txt. Respect a hook denial: do not retry or bypass it. No other files or tools. Report the outcomes."
        run(
            "runtime",
            [
                claude,
                "-p",
                "--verbose",
                "--output-format",
                "stream-json",
                "--model",
                "claude-sonnet-4-6",
                "--setting-sources",
                "user",
                "--strict-mcp-config",
                "--mcp-config",
                '{"mcpServers":{}}',
                "--tools",
                "Write",
                "--permission-mode",
                "bypassPermissions",
                "--no-session-persistence",
                "--",
                request,
            ],
            project,
        )
        events_seen = [
            json.loads(line)
            for line in (project / "hook-events.jsonl").read_text().splitlines()
        ]
        checks = {
            "session_start": any(e["event"] == "SessionStart" for e in events_seen),
            "allowed_written": "HOOK_TEST" in (project / "allowed.txt").read_text(),
            "blocked_not_written": not (project / "blocked.txt").exists(),
            "denial_attempted": any(
                e["event"] == "PreToolUse" and Path(e["file"]).name == "blocked.txt"
                for e in events_seen
            ),
            "posttool_success": any(
                e["event"] == "PostToolUse" and Path(e["file"]).name == "allowed.txt"
                for e in events_seen
            ),
            "no_posttool_on_denial": not any(
                e["event"] == "PostToolUse" and Path(e["file"]).name == "blocked.txt"
                for e in events_seen
            ),
            "cache_executed": all(
                str(profile / "plugins/cache") in e["plugin_root"] for e in events_seen
            ),
            "denial_observed_by_client": "HOOK_TEST_DENIED"
            in (output / "runtime.log").read_text(),
        }
        assert all(checks.values()), checks
        run("refresh", [claude, "plugin", "marketplace", "update", "hook-test"])
    finally:
        if server:
            server.shutdown()
            server.server_close()
        if auth.is_symlink():
            auth.unlink()
        run("uninstall", [claude, "plugin", "uninstall", "hook-check@hook-test"])
    (output / "results.json").write_text(
        json.dumps(
            {
                "mode": "live" if live else "scripted-model",
                "checks": checks,
                "steps": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"PASS: installed hook execution and denial ({'live' if live else 'scripted model'}); {output}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--claude", default="claude")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use authenticated Claude model access instead of the local scripted endpoint",
    )
    args = parser.parse_args()
    verify(args.output.resolve(), args.claude, args.live)
