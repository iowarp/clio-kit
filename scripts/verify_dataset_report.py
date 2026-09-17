#!/usr/bin/env python3
"""Exercise the installed dataset-report plugin, real MCPs and native hook runtime.

Default hook testing uses a scripted local model; --live additionally evaluates
an authenticated model using the actual skill. No public repositories are changed.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil

import tomli_w

from clio_kit.community import read_community_entries

from verify_marketplace_install import Acceptance, ROOT
from verify_plugin_hooks import model_server


def external_catalogue(suite: Acceptance) -> Path:
    """Publish an isolated external packed plugin through a real Git URL entry."""
    publisher = suite.output / "publisher"
    shutil.copytree(ROOT / "plugins/clio-dataset-report", publisher)
    path = publisher / ".claude-plugin/plugin.json"
    manifest = json.loads(path.read_text())
    manifest["name"] = "lab-dataset-report"
    path.write_text(json.dumps(manifest))
    for label, command in (
        (
            "validate-external",
            [str(suite.launcher), "plugin", "validate", str(publisher)],
        ),
        (
            "validate-external-native",
            ["claude", "plugin", "validate", str(publisher), "--strict"],
        ),
        (
            "submission-entry",
            [
                str(suite.launcher),
                "plugin",
                "submit",
                str(publisher),
                "--repo",
                "example-lab/dataset-report",
            ],
        ),
        ("git-init", ["git", "init", "-q"]),
        ("git-add", ["git", "add", "."]),
        (
            "git-commit",
            [
                "git",
                "-c",
                "user.name=Acceptance test",
                "-c",
                "user.email=test@localhost",
                "commit",
                "-qm",
                "Packed plugin fixture",
            ],
        ),
    ):
        suite.command(label, command, cwd=publisher)
    catalog = suite.output / "external-catalogue"
    entries = catalog / "community/entries"
    entries.mkdir(parents=True)
    (entries / "lab-dataset-report.toml").write_text(
        tomli_w.dumps(
            {
                "name": "lab-dataset-report",
                "description": manifest["description"],
                "source": {"type": "url", "url": publisher.as_uri()},
            }
        )
    )
    base = json.loads((ROOT / ".claude-plugin/marketplace.json").read_text())
    base["plugins"] = [
        e for e in base["plugins"] if e["name"] in manifest["dependencies"]
    ]
    for entry in base["plugins"]:
        shutil.copytree(ROOT / entry["source"], catalog / entry["source"])
    base["plugins"] += read_community_entries(catalog)
    (catalog / ".claude-plugin").mkdir()
    (catalog / ".claude-plugin/marketplace.json").write_text(json.dumps(base))
    return catalog


def verify(
    output: Path, installed: Path | None, live: bool, external: bool = False
) -> None:
    suite = Acceptance(output)
    if installed:
        suite.launcher = installed / "bin/clio-kit"
        suite.environment["CLIO_KIT_CACHE_DIR"] = str(installed.parent / "cache")
        suite.environment["CLIO_KIT_COMPONENT_BASE_URL"] = (
            installed.parent / "dist/components"
        ).as_uri()
        suite.environment["PATH"] = (
            str(installed / "bin") + os.pathsep + os.environ["PATH"]
        )
    else:
        suite.install()
    env = suite.environment
    profile = Path(env["CLAUDE_CONFIG_DIR"])
    profile.mkdir(exist_ok=True)
    project = output / "project"
    project.mkdir(exist_ok=True)
    catalog = external_catalogue(suite) if external else ROOT
    name = "lab-dataset-report" if external else "clio-dataset-report"
    suite.command(
        "add-marketplace", ["claude", "plugin", "marketplace", "add", str(catalog)]
    )
    suite.command(
        "install-report",
        ["claude", "plugin", "install", f"{name}@clio-kit"],
    )
    entries = json.loads(
        suite.command("installed-components", ["claude", "plugin", "list", "--json"])
    )
    expected = {
        name,
        "clio-hdf5",
        "clio-pandas",
        "clio-plot",
        "clio-agents",
    }
    assert {e["id"].split("@")[0] for e in entries} == expected
    entry = next(e for e in entries if e["id"] == f"{name}@clio-kit")
    plugin = Path(entry["installPath"])
    helper = plugin / "skills/creating-dataset-report/scripts/verify_report.py"
    assert (
        helper.read_bytes()
        == (
            ROOT
            / "plugins/clio-dataset-report/skills/creating-dataset-report/scripts/verify_report.py"
        ).read_bytes()
    )
    source = project / "input.h5"
    suite.command(
        "fixture",
        [
            "uv",
            "run",
            "--frozen",
            "--directory",
            str(ROOT / "mcp-servers/hdf5"),
            "python",
            "-c",
            "import h5py,numpy as np,sys\nwith h5py.File(sys.argv[1],'w') as f:\n d=f.create_dataset('readings',data=np.column_stack([np.arange(6),[2,4,8,16,32,64]]))\n d.attrs['columns']='time_s,signal_mV'\n d.attrs['units']='s,mV'\n",
            str(source),
        ],
    )
    report_dir = project / "report"
    suite.command(
        "prepare",
        [
            "python3",
            str(helper),
            "prepare",
            "--source",
            str(source),
            "--output",
            str(report_dir),
            "--column",
            "signal_mV",
        ],
    )
    baseline = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = report_dir / "clio-dataset-report.json"

    async def science():
        await suite.session(
            "hdf5-report",
            ["mcp-server", "hdf5"],
            [
                ("open_file", {"path": str(source)}),
                ("get_shape", {"path": "/readings"}),
                ("list_attributes", {"path": "/readings"}),
                (
                    "export_dataset",
                    {
                        "path": "/readings",
                        "output_path": str(project / "export.json"),
                        "export_format": "json",
                    },
                ),
                ("close_file", {}),
            ],
        )
        data = json.loads((project / "export.json").read_text())["data"]
        columns = {
            "time_s": [row[0] for row in data],
            "signal_mV": [row[1] for row in data],
        }
        replies = await suite.session(
            "pandas-report",
            ["mcp-server", "pandas"],
            [
                (
                    "save_data",
                    {
                        "data": columns,
                        "file_path": str(report_dir / "data.csv"),
                        "index": False,
                    },
                ),
                (
                    "statistical_summary",
                    {
                        "file_path": str(report_dir / "data.csv"),
                        "columns": ["signal_mV"],
                    },
                ),
            ],
        )
        stats = replies[-1]["structuredContent"]["basic_statistics"]["signal_mV"]
        evidence = json.loads(manifest.read_text())
        evidence["statistics"] = {
            key: stats["50%" if key == "median" else key]
            for key in ("count", "mean", "median", "min", "max")
        }
        manifest.write_text(json.dumps(evidence, indent=2))
        await suite.session(
            "plot-report",
            ["mcp-server", "plot"],
            [
                (
                    "line_plot",
                    {
                        "file_path": str(report_dir / "data.csv"),
                        "x_column": "time_s",
                        "y_column": "signal_mV",
                        "output_path": str(report_dir / "plot.png"),
                    },
                )
            ],
        )

    asyncio.run(science())
    with (report_dir / "data.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [float(r["signal_mV"]) for r in rows] == [2, 4, 8, 16, 32, 64]
    (report_dir / "dataset-report.md").write_text(
        "Known fixture: count 6, mean 21 mV, median 12 mV, min 2 mV, max 64 mV. No model fitted.\n"
    )
    result = json.loads(
        suite.command("verify-report", ["python3", str(helper), "check", str(manifest)])
    )
    assert result["status"] == "PASS" and result["statistics"]["median"] == 12
    good = manifest.read_text()
    bad = json.loads(good)
    bad["statistics"]["mean"] = 22
    actions = [
        {"name": "Skill", "input": {"skill": f"{name}:creating-dataset-report"}},
        {"name": "Read", "input": {"file_path": str(manifest)}},
    ] + [
        {"name": "Write", "input": {"file_path": str(manifest), "content": content}}
        for content in (json.dumps(bad), good)
    ]
    server = model_server(project, actions)
    original_env = env.copy()
    try:
        for key in list(env):
            if key.startswith(("ANTHROPIC_", "CLAUDE_CODE_USE_")) or key in (
                "CLAUDE_CODE_OAUTH_TOKEN",
                "CLAUDE_CODE_API_KEY_HELPER_TTL_MS",
            ):
                env.pop(key)
        env.update(
            ANTHROPIC_BASE_URL=f"http://127.0.0.1:{server.server_port}",
            ANTHROPIC_API_KEY="local-report-test-only",
        )
        log = suite.command(
            "native-hook",
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
                "Skill,Read,Write",
                "--permission-mode",
                "bypassPermissions",
                "--no-session-persistence",
                "--",
                "Exercise the report checker.",
            ],
            cwd=project,
        )
        hook_context = (project / "scripted-model-requests.jsonl").read_text()
        assert "CLIO_DATASET_REPORT_CHECK" in hook_context
        assert "Statistic mismatch" in hook_context
        assert '\\"status\\": \\"PASS\\"' in hook_context
        assert "Create a verified dataset report" in hook_context
        assert json.loads(manifest.read_text())["statistics"]["mean"] == 21
        server.shutdown()
        server.server_close()
        server = model_server(
            project, [{"name": "Read", "input": {"file_path": str(manifest)}}]
        )
        env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{server.server_port}"
        suite.command(
            "native-reviewer",
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
                "clio-agents:scientific-evidence-reviewer",
                "--tools",
                "Read,Glob,Grep",
                "--permission-mode",
                "bypassPermissions",
                "--no-session-persistence",
                "--",
                "Read the supplied report evidence.",
            ],
            cwd=project,
        )
        assert (
            "A numerical correction needs reproducible calculation evidence"
            in (project / "scripted-model-requests.jsonl").read_text()
        )
    finally:
        server.shutdown()
        server.server_close()
        env.clear()
        env.update(original_env)
    if live:
        auth = profile / ".credentials.json"
        original = (
            Path(os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude")))
            / ".credentials.json"
        )
        try:
            auth.symlink_to(original)
            request = f"Use {name}:creating-dataset-report on {source}, /readings, columns time_s (s) and signal_mV (mV). Create outputs in {project / 'live-report'}. Use the installed MCP tools and verifier, and the installed evidence-reviewer agent. Do not modify input. Report the actual checks and limitations."
            log = suite.command(
                "live-workflow",
                [
                    "claude",
                    "-p",
                    "--verbose",
                    "--output-format",
                    "stream-json",
                    "--model",
                    "claude-sonnet-4-6",
                    "--effort",
                    "low",
                    "--setting-sources",
                    "user",
                    "--tools",
                    "Read,Write,Edit,Bash,Skill,Agent",
                    "--permission-mode",
                    "bypassPermissions",
                    "--no-session-persistence",
                    "--",
                    request,
                ],
                cwd=project,
            )
            events = [
                json.loads(line) for line in log.splitlines() if line.startswith("{")
            ]
            final = next(e for e in reversed(events) if e.get("type") == "result")
            assert not final.get("is_error"), final
            (output / "live-result.md").write_text(final["result"])
            uses = [
                b
                for e in events
                if e.get("type") == "assistant"
                for b in e.get("message", {}).get("content", [])
                if b.get("type") == "tool_use"
            ]
            assert any(u["name"] == "Skill" for u in uses)
            for name in ("__export_dataset", "__statistical_summary", "__line_plot"):
                assert any(u["name"].endswith(name) for u in uses), name
            suite.command(
                "live-independent-check",
                [
                    "python3",
                    str(helper),
                    "check",
                    str(project / "live-report/clio-dataset-report.json"),
                ],
            )
        finally:
            if auth.is_symlink():
                auth.unlink()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == baseline
    suite.command(
        "keep-shared-component", ["claude", "plugin", "install", "clio-plot@clio-kit"]
    )
    suite.command(
        "remove-report",
        [
            "claude",
            "plugin",
            "uninstall",
            f"{name}@clio-kit",
            "--prune",
            "-y",
        ],
    )
    remaining = json.loads(
        suite.command("remaining-components", ["claude", "plugin", "list", "--json"])
    )
    assert [e["id"] for e in remaining] == ["clio-plot@clio-kit"]
    suite.command(
        "remove-shared",
        ["claude", "plugin", "uninstall", "clio-plot@clio-kit", "--prune", "-y"],
    )
    suite.command(
        "remove-marketplace", ["claude", "plugin", "marketplace", "remove", "clio-kit"]
    )
    print(
        f"PASS dataset report: actual MCP calculations, native hook fail/recovery, component installation/removal; {output}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--installed",
        type=Path,
        help="Reuse a freshly built isolated wheel environment",
    )
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--external",
        action="store_true",
        help="Consume the packed plugin from a separate local Git publisher through a community entry",
    )
    args = parser.parse_args()
    verify(
        args.output.resolve(),
        args.installed.resolve() if args.installed else None,
        args.live,
        args.external,
    )
