#!/usr/bin/env python3
"""Build, install, and exercise real marketplace components in isolated directories.

Run with the verification extra installed. No model credentials are needed.
The output directory retains commands, MCP replies, generated data and results.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import PaginatedRequestParams

ROOT = Path(__file__).resolve().parents[1]


class Acceptance:
    def __init__(self, output: Path) -> None:
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        self.environment = dict(os.environ)
        self.environment.update(
            CLAUDE_CONFIG_DIR=str(output / "claude"),
            CLIO_KIT_CACHE_DIR=str(output / "cache"),
        )
        self.records: list[dict] = []
        self.launcher = output / "installed" / "bin" / "clio-kit"

    def command(self, name: str, args: list[str], *, cwd: Path = ROOT) -> str:
        result = subprocess.run(
            args,
            cwd=cwd,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=600,
        )
        (self.output / f"{name}.log").write_text(result.stdout + result.stderr)
        self.records.append({"name": name, "command": args, "exit": result.returncode})
        self.save()
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {self.output / (name + '.log')}")
        print(f"PASS {name}", flush=True)
        return result.stdout

    def save(self) -> None:
        (self.output / "results.json").write_text(
            json.dumps(self.records, indent=2) + "\n"
        )

    async def session(
        self, name: str, args: list[str], calls: list[tuple[str, dict]] | None = None
    ) -> list[dict]:
        async def exchange() -> list[dict]:
            with (self.output / f"{name}.stderr").open("w") as errors:
                async with stdio_client(
                    StdioServerParameters(
                        command=str(self.launcher), args=args, env=self.environment
                    ),
                    errlog=errors,
                ) as (read, write):
                    async with ClientSession(read, write) as client:
                        initialized = await client.initialize()
                        tools = []
                        cursor = None
                        while True:
                            page = await client.list_tools(
                                params=PaginatedRequestParams(cursor=cursor)
                                if cursor
                                else None
                            )
                            tools.extend(tool.name for tool in page.tools)
                            cursor = page.model_dump(by_alias=True).get("nextCursor")
                            if not cursor:
                                break
                        assert tools, f"{name}: no tools"
                        responses = []
                        for tool, arguments in calls or []:
                            result = await client.call_tool(tool, arguments)
                            assert not result.isError, result
                            dumped = result.model_dump(mode="json", by_alias=True)
                            responses.append(dumped)
                        record = {
                            "name": name,
                            "connected": True,
                            "server": initialized.serverInfo.model_dump(),
                            "tools": tools,
                            "calls": responses,
                        }
                        self.records.append(record)
                        self.save()
                        print(
                            f"PASS {name}: {len(tools)} tools, {len(responses)} calls",
                            flush=True,
                        )
                        return responses

        return await asyncio.wait_for(exchange(), timeout=300)

    def install(self) -> None:
        self.command(
            "build-sdist-and-wheel",
            ["uv", "build", "--out-dir", str(self.output / "dist")],
        )
        wheel = next((self.output / "dist").glob("*.whl"))
        self.command(
            "create-environment",
            ["uv", "venv", "--python", sys.executable, str(self.output / "installed")],
        )
        self.command(
            "install-wheel",
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(self.launcher.parent / "python"),
                str(wheel),
            ],
        )
        self.environment["PATH"] = (
            str(self.launcher.parent) + os.pathsep + self.environment["PATH"]
        )
        self.command(
            "installed-inventory", [str(self.launcher), "mcp-servers"], cwd=self.output
        )

    def plugins(self) -> None:
        self.command(
            "strict-marketplace",
            ["claude", "plugin", "validate", str(ROOT), "--strict"],
        )
        self.command(
            "add-marketplace", ["claude", "plugin", "marketplace", "add", str(ROOT)]
        )
        names = [
            "clio-hpc",
            "clio-performance",
            "clio-scientific-io",
            "clio-analysis",
            "clio-geoscience",
            "clio-research",
            "clio-skills",
            "clio-agents",
            "iowarp-dev-setup",
            "iowarp-contributing",
        ]
        for name in names:
            self.command(
                f"install-{name}", ["claude", "plugin", "install", name + "@clio-kit"]
            )
        details = self.command(
            "agent-details", ["claude", "plugin", "details", "clio-agents@clio-kit"]
        )
        assert (
            "scientific-workflow-planner" in details
            and "scientific-evidence-reviewer" in details
        ), details
        self.command(
            "refresh-client", ["claude", "plugin", "marketplace", "update", "clio-kit"]
        )
        self.command(
            "uninstall-agents",
            ["claude", "plugin", "uninstall", "clio-agents@clio-kit"],
        )
        self.command(
            "reinstall-agents",
            ["claude", "plugin", "install", "clio-agents@clio-kit"],
        )

    def portable_skills(self) -> Path:
        project = self.output / "project"
        target = project / ".agents" / "skills"
        self.command(
            "install-portable-skills",
            [str(self.launcher), "skill", "install", "--target", str(target)],
            cwd=self.output,
        )
        inventory = json.loads(
            self.command(
                "installed-skill-inventory",
                [str(self.launcher), "skill", "list", "--json"],
                cwd=self.output,
            )
        )
        assert len(inventory) == 20, inventory
        assert {path.parent.name for path in target.glob("*/SKILL.md")} == {
            skill["name"] for skill in inventory
        }
        return project

    async def codex_skills(self, project: Path) -> None:
        """Ask a real Codex client to discover skills, without a model request."""
        target = project / ".agents" / "skills"
        with (self.output / "codex-skills.stderr").open("w") as errors:
            process = await asyncio.create_subprocess_exec(
                "codex",
                "app-server",
                "--stdio",
                cwd=project,
                env=self.environment,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=errors,
            )
            assert process.stdin is not None and process.stdout is not None

            async def send(message: dict) -> None:
                process.stdin.write((json.dumps(message) + "\n").encode())
                await process.stdin.drain()

            async def request(identifier: int, method: str, params: dict) -> dict:
                await send({"id": identifier, "method": method, "params": params})
                while raw := await process.stdout.readline():
                    reply = json.loads(raw)
                    if reply.get("id") == identifier:
                        if "error" in reply:
                            raise RuntimeError(str(reply["error"]))
                        return reply["result"]
                raise RuntimeError("Codex exited before responding")

            async def discover() -> list[dict]:
                await request(
                    1,
                    "initialize",
                    {"clientInfo": {"name": "clio_acceptance", "version": "1.0.0"}},
                )
                await send({"method": "initialized"})
                result = await request(
                    2,
                    "skills/list",
                    {
                        "cwds": [str(project)],
                        "forceReload": True,
                    },
                )
                records = []
                for row in result["data"]:
                    # Retain only test-project data, never unrelated user skills.
                    records.extend(
                        skill
                        for skill in row["skills"]
                        if Path(skill["path"]).is_relative_to(target)
                    )
                    assert not [
                        error
                        for error in row.get("errors", [])
                        if Path(error["path"]).is_relative_to(target)
                    ]
                return records

            try:
                records = await asyncio.wait_for(discover(), timeout=60)
            finally:
                if process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
        expected = {path.parent.name for path in target.glob("*/SKILL.md")}
        assert len(records) == len(expected) == 20, records
        assert {skill["name"] for skill in records} == expected, records
        assert all(skill["enabled"] for skill in records), records
        (self.output / "codex-skills.json").write_text(json.dumps(records, indent=2))
        self.records.append({"name": "codex-skill-discovery", "enabled": len(records)})
        self.save()
        print(f"PASS Codex discovers and enables {len(records)} skills", flush=True)

    async def workflows(self, all_servers: bool) -> None:
        for runtime in ("typescript", "go"):
            for state in ("cold", "warm"):
                replies = await self.session(
                    f"{runtime}-{state}",
                    [
                        "server",
                        "run",
                        str(ROOT / "tests/fixtures/mcp-servers" / runtime),
                    ],
                    [("multiply", {"a": 6, "b": 7})],
                )
                payload = replies[0].get("structuredContent") or json.loads(
                    replies[0]["content"][0]["text"]
                )
                assert payload["product"] == 42, payload
        data = self.output / "data"
        data.mkdir(exist_ok=True)
        expected = b"CLIO real round trip\n" * 100
        archive = data / "sample.txt.gz"
        archive.write_bytes(gzip.compress(expected))
        await self.session(
            "compression-roundtrip",
            ["mcp-server", "compression"],
            [("decompress_file_tool", {"file_path": str(archive)})],
        )
        assert archive.with_suffix("").read_bytes() == expected
        source = data / "runs.csv"
        source.write_text("machine,runtime\nalpha,2\nalpha,4\nbeta,10\nbeta,14\n")
        response = await self.session(
            "pandas-aggregate",
            ["mcp-server", "pandas"],
            [
                (
                    "groupby_operations",
                    {
                        "file_path": str(source),
                        "group_by": ["machine"],
                        "operations": {"runtime": "mean"},
                    },
                )
            ],
        )
        grouped = response[0].get("structuredContent") or json.loads(
            response[0]["content"][0]["text"]
        )
        transformed = Path(grouped["output_file"])
        with transformed.open() as handle:
            means = {
                row["machine"]: float(row["runtime"]) for row in csv.DictReader(handle)
            }
        assert means == {"alpha": 3.0, "beta": 12.0}, means
        image = data / "means.png"
        await self.session(
            "plot-transformed-data",
            ["mcp-server", "plot"],
            [
                (
                    "line_plot",
                    {
                        "file_path": str(transformed),
                        "x_column": "machine",
                        "y_column": "runtime",
                        "output_path": str(image),
                    },
                )
            ],
        )
        assert (
            image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            and image.stat().st_size > 1000
        )
        if all_servers:
            inventory = self.command(
                "all-server-inventory", [str(self.launcher), "mcp-servers"]
            )
            for name in re.findall(r"^\s+- ([a-z0-9-]+)$", inventory, re.M):
                await self.session("connect-" + name, ["mcp-server", name])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--all-servers", action="store_true")
    parser.add_argument(
        "--codex", action="store_true", help="Verify real Codex skill discovery"
    )
    parser.add_argument(
        "--skip-client",
        action="store_true",
        help="Only for systems without Claude Code installed",
    )
    args = parser.parse_args()
    output = (
        args.output or Path(tempfile.mkdtemp(prefix="clio-acceptance-"))
    ).resolve()
    suite = Acceptance(output)
    suite.install()
    project = suite.portable_skills()
    if args.codex:
        asyncio.run(suite.codex_skills(project))
    if not args.skip_client:
        suite.plugins()
    asyncio.run(suite.workflows(args.all_servers))
    print(f"Acceptance passed. Evidence: {output}")


if __name__ == "__main__":
    main()
