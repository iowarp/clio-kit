#!/usr/bin/env python3
"""Drop four component types into an isolated checkout and consume them for real.

No model credentials or public writes. Scripted model responses drive the real
Claude runtime; its MCP subprocess and hooks are actually executed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import tomli_w

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client

from clio_kit.plugin_components import write_agent
from clio_kit.plugins import validate_plugin
from clio_kit.client_install import CLIENTS, install_for_client, tomllib
from generate_marketplace import generate
from generate_website_catalogue import generate as catalogue
from verify_external_contributions import NUMERICS, SERVER
from verify_plugin_hooks import HOOK, model_server

ROOT = Path(__file__).resolve().parents[1]


def verify(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    checkout = output / "checkout"
    checkout.mkdir()
    for name in ("plugins", "skills", ".claude-plugin", "community", "src"):
        shutil.copytree(
            ROOT / name,
            checkout / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for name in ("pyproject.toml", "README.md", "mcp-server-versions.toml"):
        shutil.copy2(ROOT / name, checkout / name)
    (checkout / "scripts").mkdir()
    shutil.copy2(
        ROOT / "scripts/package_skill_assets.py",
        checkout / "scripts/package_skill_assets.py",
    )
    shutil.copy2(
        ROOT / "scripts/package_components.py",
        checkout / "scripts/package_components.py",
    )
    # Empty unrelated shared-data trees keep this fixture build focused on skills.
    for name in ("mcp-servers", "prompts"):
        (checkout / name).mkdir()
    # This fixture deliberately omits maintained server payloads, so it must not
    # ship their prerequisite records as if those servers were present.
    inventory_path = checkout / "mcp-server-versions.toml"
    inventory = tomllib.loads(inventory_path.read_text())
    inventory.pop("prerequisites", None)
    inventory_path.write_text(tomli_w.dumps(inventory))
    profile = output / "profile"
    profile.mkdir()
    project = output / "project"
    project.mkdir()
    env = dict(os.environ, CLAUDE_CONFIG_DIR=str(profile))
    for key in list(env):
        if key.startswith(("ANTHROPIC_", "CLAUDE_CODE_USE_")) or key in (
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CLAUDE_CODE_API_KEY_HELPER_TTL_MS",
        ):
            env.pop(key)
    records = []

    def run(label, args, cwd=project):
        result = subprocess.run(
            args, cwd=cwd, env=env, text=True, capture_output=True, timeout=240
        )
        (output / f"{label}.log").write_text(result.stdout + result.stderr)
        records.append({"step": label, "exit": result.returncode})
        (output / "results.json").write_text(json.dumps(records, indent=2))
        assert result.returncode == 0, (label, result.stderr[-1000:])
        print("PASS", label, flush=True)
        return result.stdout

    def package(kind, name):
        directory = checkout / kind / name
        (directory / ".claude-plugin").mkdir(parents=True)
        (directory / ".claude-plugin/plugin.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "version": "1.0.0",
                    "description": "Local component acceptance package.",
                    "author": {"name": "IOWarp"},
                    "license": "BSD-3-Clause",
                }
            )
        )
        return directory

    tools = package("plugins", "dropin-tools")
    (tools / "server.py").write_text(SERVER)
    (tools / "numerics.py").write_text(NUMERICS)
    (tools / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "lab": {
                        "command": sys.executable,
                        "args": ["${CLAUDE_PLUGIN_ROOT}/server.py"],
                    }
                }
            }
        )
    )
    skills = package("skills", "dropin-skills")
    skill = skills / "skills/weighing-lab-results"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: weighing-lab-results\ndescription: 'Use when calculating weighted lab measurements. Triggers on \"weighted lab result\". Not for unweighted statistics.'\n---\n# Weigh lab results\nCall weighted_mean with explicit values and weights. Report the returned result; do not claim a population inference.\n"
    )
    (skill / "evals.md").write_text(
        "# Scenario\nValues 2,4,8 and weights 1,1,2 yield 5.5.\n"
    )
    agents = package("agents", "dropin-agents")
    write_agent(agents)
    hooks = package("hooks", "dropin-hooks")
    (hooks / "hooks").mkdir()
    (hooks / "hook.py").write_text(HOOK)
    (hooks / "hooks/hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": 'python3 "${CLAUDE_PLUGIN_ROOT}/hook.py"',
                                    "timeout": 10,
                                }
                            ]
                        }
                    ]
                }
            }
        )
    )
    packed = package("plugins", "dropin-workflow")
    path = packed / ".claude-plugin/plugin.json"
    manifest = json.loads(path.read_text())
    manifest["dependencies"] = [
        "dropin-tools",
        "dropin-skills",
        "dropin-agents",
        "dropin-hooks",
    ]
    path.write_text(json.dumps(manifest))
    for directory in (tools, skills, agents, hooks, packed):
        assert validate_plugin(directory)[1] == [], directory
        run(
            "validate-" + directory.name,
            ["claude", "plugin", "validate", str(directory), "--strict"],
        )
    original = {
        str(p.relative_to(checkout)): p.read_bytes()
        for directory in (tools, skills, agents, hooks, packed)
        for p in directory.rglob("*")
        if p.is_file()
    }
    generate(checkout)
    for relative, content in original.items():
        assert (checkout / relative).read_bytes() == content
    items = {item["id"]: item for item in catalogue(checkout)["items"]}
    for name in manifest["dependencies"] + ["dropin-workflow"]:
        assert "plugin/" + name in items
    assert "skill/weighing-lab-results" in items
    assert "agent/dropin-agents/workflow-reviewer" in items
    assert "hook/dropin-hooks" in items
    assert items["plugin/dropin-tools"]["servers"] == ["plugin:dropin-tools:lab"]
    assert items["plugin/dropin-tools"]["kind"] == "mcp"
    assert items["plugin/dropin-tools"]["installation"] == "native-package"
    for name in ("dropin-skills", "dropin-agents", "dropin-hooks"):
        assert items["plugin/" + name]["kind"] == "collection"
    assert items["plugin/dropin-workflow"]["kind"] == "plugin"
    assert items["plugin/dropin-workflow"]["componentTypes"] == [
        "agent",
        "hook",
        "mcp",
        "skill",
    ]
    run(
        "marketplace-validate",
        ["claude", "plugin", "validate", str(checkout), "--strict"],
    )
    run("add", ["claude", "plugin", "marketplace", "add", str(checkout)])
    run("install", ["claude", "plugin", "install", "dropin-workflow@clio-kit"])
    installed = json.loads(run("list", ["claude", "plugin", "list", "--json"]))
    assert {e["id"].split("@")[0] for e in installed} == set(
        manifest["dependencies"]
    ) | {"dropin-workflow"}
    entry = next(e for e in installed if e["id"] == "dropin-tools@clio-kit")
    tool_dir = Path(entry["installPath"])

    async def query():
        async with Client(
            stdio_client(
                StdioServerParameters(
                    command=sys.executable, args=[str(tool_dir / "server.py")]
                )
            ),
            mode="legacy",
        ) as client:
            reply = await client.call_tool(
                "weighted_mean", {"values": [2, 4, 8], "weights": [1, 1, 2]}
            )
            assert not reply.is_error
            assert json.loads(reply.content[0].text)["mean"] == 5.5
            (output / "mcp-result.json").write_text(
                reply.model_dump_json(by_alias=True)
            )

    asyncio.run(query())

    async def client_projects():
        for host, (_, config_name, key) in CLIENTS.items():
            destination = output / "client-projects" / host
            installed_components = install_for_client(
                checkout, "dropin-workflow", host, destination, components_only=True
            )
            assert len(installed_components["not_installed"]) == 2
            config_path = destination / config_name
            text = config_path.read_text()
            settings = (tomllib.loads(text) if host == "codex" else json.loads(text))[
                key
            ]["lab"]
            if host == "opencode":
                command, *arguments = settings["command"]
            else:
                command, arguments = settings["command"], settings["args"]
            async with Client(
                stdio_client(StdioServerParameters(command=command, args=arguments)),
                mode="legacy",
            ) as client:
                result = await client.call_tool(
                    "weighted_mean", {"values": [2, 4, 8], "weights": [1, 1, 2]}
                )
                assert not result.is_error
                assert json.loads(result.content[0].text)["mean"] == 5.5
            records.append({"step": "client-project-" + host, "exit": 0})
            print("PASS project configuration and real MCP call:", host, flush=True)

    asyncio.run(client_projects())
    actions = [
        {"name": "Skill", "input": {"skill": "dropin-skills:weighing-lab-results"}},
        {
            "name": "mcp__plugin_dropin-tools_lab__weighted_mean",
            "input": {"values": [2, 4, 8], "weights": [1, 1, 2]},
        },
    ]
    server = model_server(project, actions)
    try:
        env.update(
            ANTHROPIC_BASE_URL=f"http://127.0.0.1:{server.server_port}",
            ANTHROPIC_API_KEY="local-component-test-only",
        )
        run(
            "native-use",
            [
                "claude",
                "-p",
                "--verbose",
                "--output-format",
                "stream-json",
                "--model",
                "claude-sonnet-4-6",
                "--setting-sources",
                "user",
                "--tools",
                "Skill",
                "--permission-mode",
                "bypassPermissions",
                "--no-session-persistence",
                "--",
                "Use the weighted lab workflow.",
            ],
        )
        context = (project / "scripted-model-requests.jsonl").read_text()
        assert "Weigh lab results" in context and "5.5" in context
        events = [
            json.loads(line)
            for line in (project / "hook-events.jsonl").read_text().splitlines()
        ]
        hook_install = Path(
            next(
                e["installPath"]
                for e in installed
                if e["id"] == "dropin-hooks@clio-kit"
            )
        ).resolve()
        assert any(
            e["event"] == "SessionStart"
            # Local marketplaces may execute hooks from the checkout rather
            # than the copied cache; both must name this installed package.
            and Path(e["plugin_root"]).resolve() in {hook_install, hooks.resolve()}
            for e in events
        )
        server.shutdown()
        server.server_close()
        server = model_server(
            project,
            [{"name": "Read", "input": {"file_path": str(output / "mcp-result.json")}}],
        )
        env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{server.server_port}"
        run(
            "native-agent",
            [
                "claude",
                "-p",
                "--verbose",
                "--output-format",
                "stream-json",
                "--model",
                "claude-sonnet-4-6",
                "--setting-sources",
                "user",
                "--agent",
                "dropin-agents:workflow-reviewer",
                "--tools",
                "Read,Glob,Grep",
                "--permission-mode",
                "bypassPermissions",
                "--no-session-persistence",
                "--",
                "Read the recorded MCP evidence.",
            ],
        )
        assert (
            "Review the supplied workflow and its recorded outputs"
            in (project / "scripted-model-requests.jsonl").read_text()
        )
    finally:
        server.shutdown()
        server.server_close()
    manifest["version"] = "1.1.0"
    path.write_text(json.dumps(manifest))
    generate(checkout)
    run("refresh", ["claude", "plugin", "marketplace", "update", "clio-kit"])
    run("update", ["claude", "plugin", "update", "dropin-workflow@clio-kit"])
    updated = json.loads(run("updated-list", ["claude", "plugin", "list", "--json"]))
    assert (
        next(e for e in updated if e["id"] == "dropin-workflow@clio-kit")["version"]
        == "1.1.0"
    )
    run(
        "uninstall",
        ["claude", "plugin", "uninstall", "dropin-workflow@clio-kit", "--prune", "-y"],
    )
    assert json.loads(run("empty-list", ["claude", "plugin", "list", "--json"])) == []
    run("remove", ["claude", "plugin", "marketplace", "remove", "clio-kit"])
    # New plugin-local resources must join the wheel without a pyproject entry.
    extra = tools / "skills/local-tool-guide"
    shutil.copytree(skill, extra)
    (extra / "SKILL.md").write_text(
        (extra / "SKILL.md")
        .read_text()
        .replace("weighing-lab-results", "local-tool-guide")
    )
    (extra / "references").mkdir()
    (extra / "references/formula.txt").write_text("sum(v*w)/sum(w)\n")
    run("build", ["uv", "build", "--out-dir", str(output / "dist")], cwd=checkout)
    env["CLIO_KIT_COMPONENT_BASE_URL"] = (output / "dist/components").resolve().as_uri()
    env["CLIO_KIT_CACHE_DIR"] = str(output / "component-cache")
    run("venv", ["uv", "venv", str(output / "installed")])
    python = output / "installed/bin/python"
    run(
        "install-wheel",
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            str(next((output / "dist").glob("*.whl"))),
        ],
    )
    cli = output / "installed/bin/clio-kit"
    run("skill-list", [str(cli), "skill", "list"])
    for name in ("weighing-lab-results", "local-tool-guide"):
        run(
            "portable-" + name,
            [
                str(cli),
                "skill",
                "install",
                name,
                "--target",
                str(project / ".agents/skills"),
            ],
        )
    assert (
        project / ".agents/skills/local-tool-guide/references/formula.txt"
    ).read_bytes() == (extra / "references/formula.txt").read_bytes()
    # Exercise a contributed script after selective installation from the wheel.
    released_project = output / "released-project"
    run(
        "released-workflow-install",
        [
            str(cli),
            "plugin",
            "install",
            "dropin-workflow",
            "--client",
            "codex",
            "--project",
            str(released_project),
            "--components-only",
        ],
    )
    settings = tomllib.loads((released_project / ".codex/config.toml").read_text())[
        "mcp_servers"
    ]["lab"]
    assert Path(settings["args"][0]).is_relative_to(output / "component-cache")

    async def released_query():
        async with Client(
            stdio_client(
                StdioServerParameters(
                    command=settings["command"], args=settings["args"]
                )
            ),
            mode="legacy",
        ) as client:
            reply = await client.call_tool(
                "weighted_mean", {"values": [2, 4, 8], "weights": [1, 1, 2]}
            )
            assert not reply.is_error
            assert json.loads(reply.content[0].text)["mean"] == 5.5
            (output / "released-mcp-result.json").write_text(
                reply.model_dump_json(by_alias=True)
            )

    asyncio.run(released_query())
    print("PASS selective released workflow script: weighted_mean = 5.5", flush=True)
    shutil.rmtree(packed)
    generate(checkout)
    entries = json.loads((checkout / ".claude-plugin/marketplace.json").read_text())[
        "plugins"
    ]
    assert not any(e["name"] == "dropin-workflow" for e in entries)
    print(
        "PASS: folder discovery, all native component types, actual MCP use, update/removal, and wheel-from-sdist portable resources."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    verify(parser.parse_args().output.resolve())
