#!/usr/bin/env python3
"""Exercise community transports using real Git, npm, Claude and an MCP query.

All repositories, registry data, profiles and results stay under --output.
Git URL rewrites stand in for a published GitHub fixture; no remote writes occur.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import tomli_w

from clio_kit.community import read_community_entries
from clio_kit.plugins import validate_plugin

ROOT = Path(__file__).resolve().parents[1]
NAME = "coder-external-check"
SERVER = """import json, sys
from numerics import weighted_mean
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method = request["method"]
    if method == "initialize":
        result = {"protocolVersion": "2025-11-25", "capabilities": {"tools": {}}, "serverInfo": {"name": "external-check", "version": "1.0.0"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "weighted_mean", "description": "Compute a weighted mean", "inputSchema": {"type": "object", "properties": {"values": {"type": "array", "items": {"type": "number"}}, "weights": {"type": "array", "items": {"type": "number"}}}, "required": ["values", "weights"]}}]}
    elif method == "tools/call":
        args = request["params"]["arguments"]
        value = weighted_mean(args["values"], args["weights"])
        result = {"content": [{"type": "text", "text": json.dumps({"mean": value})}]}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
"""


NUMERICS = "import math\ndef weighted_mean(values, weights):\n    return math.fsum(v * w for v, w in zip(values, weights)) / math.fsum(weights)\n"


class Acceptance:
    def __init__(self, output: Path):
        self.output = output
        output.mkdir(parents=True, exist_ok=False)
        self.records: list[dict] = []
        self.env = dict(os.environ)
        self.env["GIT_CONFIG_GLOBAL"] = str(output / "gitconfig")
        self.env["GIT_CONFIG_NOSYSTEM"] = "1"
        self.env["npm_config_cache"] = str(output / "npm-cache")
        for scope in ("userconfig", "globalconfig"):
            config = output / f"npm-{scope}"
            config.write_text("")
            self.env[f"npm_config_{scope}"] = str(config)

    def record(self, **record):
        self.records.append(record)
        (self.output / "results.json").write_text(
            json.dumps(self.records, indent=2) + "\n"
        )

    def run(self, label: str, args: list[str], cwd: Path, env: dict | None = None):
        result = subprocess.run(
            args,
            cwd=cwd,
            env=env or self.env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        (self.output / f"{label}.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"{label}: exit {result.returncode}; see {label}.log")
        return result.stdout

    def fixture(self):
        plugin = self.output / "publisher"
        (plugin / ".claude-plugin").mkdir(parents=True)
        (plugin / ".claude-plugin/plugin.json").write_text(
            json.dumps(
                {
                    "name": NAME,
                    "version": "1.0.0",
                    "description": "Scientific debugging skill with a real numerical MCP fixture.",
                    "author": {"name": "IOWarp"},
                }
            )
        )
        source = ROOT / "skills/clio-coder-skills/skills/clio-kit-scientific-debugging"
        shutil.copytree(source, plugin / "skills/clio-kit-scientific-debugging")
        (plugin / "server.py").write_text(SERVER)
        (plugin / "numerics.py").write_text(NUMERICS)
        (plugin / ".mcp.json").write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "numerics": {
                            "command": sys.executable,
                            "args": ["${CLAUDE_PLUGIN_ROOT}/server.py"],
                        }
                    }
                }
            )
        )
        (plugin / "package.json").write_text(
            json.dumps(
                {
                    "name": NAME,
                    "version": "1.0.0",
                    "files": [
                        ".claude-plugin",
                        ".mcp.json",
                        "skills",
                        "server.py",
                        "numerics.py",
                    ],
                }
            )
        )
        assert validate_plugin(plugin)[1] == []
        self.run(
            "fixture-validate",
            ["claude", "plugin", "validate", str(plugin), "--strict"],
            self.output,
        )
        monorepo = self.output / "monorepo"
        shutil.copytree(plugin, monorepo / "packages" / NAME)
        (monorepo / ".claude-plugin").mkdir()
        (monorepo / ".claude-plugin/marketplace.json").write_text(
            json.dumps(
                {
                    "name": "upstream-check",
                    "owner": {"name": "IOWarp"},
                    "metadata": {"description": "External contribution acceptance"},
                    "plugins": [
                        {
                            "name": NAME,
                            "description": "External skill and MCP",
                            "source": f"./packages/{NAME}",
                        }
                    ],
                }
            )
        )
        for directory in (plugin, monorepo):
            self.run(
                directory.name + "-git-init",
                ["git", "init", "-q", str(directory)],
                self.output,
            )
            self.run(directory.name + "-git-add", ["git", "add", "."], directory)
            self.run(
                directory.name + "-git-commit",
                [
                    "git",
                    "-c",
                    "user.name=CLIO acceptance",
                    "-c",
                    "user.email=acceptance@localhost",
                    "commit",
                    "-qm",
                    "External fixture",
                ],
                directory,
            )
        # Keep the runtime fixture in Git so clients actually fetch it. Rewrites
        # are scoped to these synthetic hosts in this process's isolated config.
        (self.output / "gitconfig").write_text(
            '[protocol "file"]\n\tallow = always\n'
            f'[url "{plugin.as_uri()}"]\n\tinsteadOf = https://github.com/clio-acceptance/fixture.git\n\tinsteadOf = https://github.com/clio-acceptance/fixture\n\tinsteadOf = https://fixture.invalid/plugin.git\n'
            "\tinsteadOf = git@github.com:clio-acceptance/fixture.git\n\tinsteadOf = git@github.com:clio-acceptance/fixture\n"
            f'[url "{monorepo.as_uri()}"]\n\tinsteadOf = https://fixture.invalid/monorepo.git\n'
            "\tinsteadOf = https://github.com/clio-acceptance/marketplace.git\n\tinsteadOf = https://github.com/clio-acceptance/marketplace\n"
        )
        return plugin, monorepo

    async def query(self, installed: Path, route: str):
        from mcp import Client, StdioServerParameters

        config = json.loads((installed / ".mcp.json").read_text())["mcpServers"][
            "numerics"
        ]
        arguments = [
            arg.replace("${CLAUDE_PLUGIN_ROOT}", str(installed))
            for arg in config["args"]
        ]
        async with Client(
            StdioServerParameters(
                command=config["command"], args=arguments, env=config.get("env")
            ),
            mode="legacy",
        ) as client:
            tools = await client.list_tools()
            assert any(tool.name == "weighted_mean" for tool in tools.tools)
            result = await client.call_tool(
                "weighted_mean", {"values": [2, 4, 8], "weights": [1, 1, 2]}
            )
            assert not result.is_error, result
            assert json.loads(result.content[0].text)["mean"] == 5.5
            self.record(
                route=route,
                stage="actual-installed-mcp-query",
                status="passed",
                mean=5.5,
            )

    def prepare_catalogue(self, route: str, source: dict, kind: str = "plugin"):
        project = self.output / route
        (project / "community/entries").mkdir(parents=True)
        entry = {
            "name": NAME,
            "kind": kind,
            "description": "External scientific workflow",
            "category": "research",
            "maintainer": "IOWarp",
            "source": source,
        }
        (project / f"community/entries/{NAME}.toml").write_text(tomli_w.dumps(entry))
        catalogue = project / ".claude-plugin/marketplace.json"
        catalogue.parent.mkdir()
        catalogue.write_text(
            json.dumps(
                {
                    "name": "external-check",
                    "owner": {"name": "IOWarp"},
                    "metadata": {"description": "External contribution acceptance"},
                    "plugins": read_community_entries(project),
                }
            )
        )
        if kind == "marketplace":
            self.run(
                route + "-federate",
                [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; import sys; from clio_kit.federation import refresh_marketplace; refresh_marketplace(Path(sys.argv[1]))",
                    str(project),
                ],
                ROOT,
            )
        return project

    def route(self, route: str, source: dict, kind: str = "plugin"):
        project = self.prepare_catalogue(route, source, kind)
        env = {**self.env, "CLAUDE_CONFIG_DIR": str(project / "profile")}
        if source.get("type") == "npm":
            # The loopback fixture is this isolated client's default registry.
            env["npm_config_registry"] = source["registry"]
        self.run(
            route + "-catalogue",
            ["claude", "plugin", "validate", str(project), "--strict"],
            project,
            env,
        )
        self.run(
            route + "-add",
            ["claude", "plugin", "marketplace", "add", str(project)],
            project,
            env,
        )
        self.run(
            route + "-install",
            ["claude", "plugin", "install", NAME + "@external-check"],
            project,
            env,
        )
        registry = json.loads(
            (project / "profile/plugins/installed_plugins.json").read_text()
        )
        installed = Path(
            registry["plugins"][NAME + "@external-check"][0]["installPath"]
        )
        expected = (
            ROOT / "skills/clio-coder-skills/skills/clio-kit-scientific-debugging"
        )
        for path in expected.rglob("*"):
            if path.is_file():
                assert (
                    installed
                    / "skills/clio-kit-scientific-debugging"
                    / path.relative_to(expected)
                ).read_bytes() == path.read_bytes()
        asyncio.run(asyncio.wait_for(self.query(installed, route), timeout=30))
        self.run(
            route + "-details",
            ["claude", "plugin", "details", NAME + "@external-check"],
            project,
            env,
        )
        self.run(
            route + "-refresh",
            ["claude", "plugin", "marketplace", "update", "external-check"],
            project,
            env,
        )
        self.run(
            route + "-remove",
            ["claude", "plugin", "uninstall", NAME + "@external-check"],
            project,
            env,
        )
        registry = json.loads(
            (project / "profile/plugins/installed_plugins.json").read_text()
        )
        assert not registry["plugins"].get(NAME + "@external-check")
        self.record(
            route=route, stage="install-resources-query-refresh-remove", status="passed"
        )

    def execute(self):
        plugin, monorepo = self.fixture()
        routes = {
            "github": {"type": "github", "repo": "clio-acceptance/fixture"},
            "url": {"type": "url", "url": "https://fixture.invalid/plugin.git"},
            "git-subdir": {
                "type": "git-subdir",
                "url": "https://fixture.invalid/monorepo.git",
                "path": f"packages/{NAME}",
            },
        }
        packed = json.loads(
            self.run("npm-pack", ["npm", "pack", "--json", "--ignore-scripts"], plugin)
        )[0]
        tarball = (plugin / packed["filename"]).read_bytes()

        class Registry(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.endswith(".tgz"):
                    body, mime = tarball, "application/octet-stream"
                else:
                    version = {
                        "name": NAME,
                        "version": "1.0.0",
                        "author": {"name": "IOWarp"},
                        "dist": {
                            "tarball": f"http://127.0.0.1:{self.server.server_port}/{NAME}.tgz",
                            "shasum": hashlib.sha1(tarball).hexdigest(),
                        },
                    }
                    body = json.dumps(
                        {
                            "name": NAME,
                            "dist-tags": {"latest": "1.0.0"},
                            "versions": {"1.0.0": version},
                        }
                    ).encode()
                    mime = "application/json"
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Registry) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            routes["npm"] = {
                "type": "npm",
                "package": NAME,
                "version": "1.0.0",
                "registry": f"http://127.0.0.1:{server.server_port}",
            }
            try:
                for route, source in routes.items():
                    try:
                        self.route(route, source)
                        print(f"PASS {route}", flush=True)
                    except Exception as exc:
                        self.record(route=route, status="failed", error=str(exc))
                        print(f"FAIL {route}: {exc}", flush=True)
                for route, source in {
                    "federation-github": {
                        "type": "github",
                        "repo": "clio-acceptance/marketplace",
                    },
                    "federation-url": {
                        "type": "url",
                        "url": "https://fixture.invalid/monorepo.git",
                    },
                }.items():
                    try:
                        self.route(route, source, "marketplace")
                        print(f"PASS {route}", flush=True)
                    except Exception as exc:
                        self.record(route=route, status="failed", error=str(exc))
                        print(f"FAIL {route}: {exc}", flush=True)
            finally:
                server.shutdown()
                thread.join()
        return not any(record.get("status") == "failed" for record in self.records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New disposable directory for fixtures and evidence",
    )
    args = parser.parse_args()
    raise SystemExit(0 if Acceptance(args.output.resolve()).execute() else 1)
