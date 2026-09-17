#!/usr/bin/env python3
"""Prove selective installation using a built wheel and a recording HTTP mirror.

No public writes. Dependencies may be downloaded from their locked registries;
the request log proves which CLIO component payloads were fetched.
"""

from __future__ import annotations

import argparse
import asyncio
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import re
from pathlib import Path
import threading
import zipfile
import tarfile

from verify_marketplace_install import Acceptance, ROOT


async def verify(output: Path) -> None:
    acceptance = Acceptance(output)
    acceptance.install()
    dist = output / "dist"
    index = json.loads((dist / "components/index.json").read_text())
    wheel = next(dist.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        assert not any(
            name.startswith(("mcp-servers/", "skills/", "clio-agentic-search/"))
            or ".data/data/" in name
            for name in archive.namelist()
        )
        assert json.loads(archive.read("clio_kit/_components.json")) == index
    with tarfile.open(next(dist.glob("clio_kit-*.tar.gz"))) as archive:
        assert not any(
            "/mcp-servers/" in name
            or "/clio-agentic-search/" in name
            or "/skills/" in name
            for name in archive.getnames()
        )
    requests: list[str] = []

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path.lstrip("/"))
            super().do_GET()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(Handler, directory=str(dist / "components"))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    acceptance.environment["CLIO_KIT_COMPONENT_BASE_URL"] = (
        f"http://127.0.0.1:{server.server_port}"
    )
    cli = str(acceptance.launcher)

    def command(label, *args):
        return acceptance.command(label, [cli, *args], cwd=output)

    def expect(keys):
        expected = {index["artifacts"][key]["file"] for key in keys}
        assert set(requests) == expected, {
            "expected": sorted(expected),
            "actual": requests,
        }
        assert len(requests) == len(expected), "A cached artifact was downloaded twice"
        (output / "download-requests.json").write_text(json.dumps(requests, indent=2))

    try:
        command("metadata-servers", "mcp-servers")
        command("metadata-skills", "skill", "list", "--json")
        command("metadata-prompts", "prompts")
        command(
            "metadata-dry-run",
            "plugin",
            "install",
            "clio-scientific-io",
            "--client",
            "codex",
            "--project",
            str(output / "project"),
            "--dry-run",
        )
        expect([])
        assert not (output / "project").exists()
        skill = "choosing-a-storage-format"
        target = output / "single-skill"
        command("one-skill", "skill", "install", skill, "--target", str(target))
        expect([f"skill/{skill}"])
        acceptance.environment["CLIO_KIT_OFFLINE"] = "1"
        command(
            "offline-repeat-skill", "skill", "install", skill, "--target", str(target)
        )
        del acceptance.environment["CLIO_KIT_OFFLINE"]
        expect([f"skill/{skill}"])
        command(
            "one-mcp-config",
            "plugin",
            "install",
            "clio-hdf5",
            "--client",
            "codex",
            "--project",
            str(output / "project"),
        )
        expect([f"skill/{skill}"])
        data = output / "numbers.h5"
        acceptance.command(
            "prepare-known-hdf5",
            [
                "uv",
                "run",
                "--frozen",
                "--directory",
                str(ROOT / "mcp-servers/hdf5"),
                "python",
                "-c",
                "import h5py,sys; f=h5py.File(sys.argv[1],'w'); f.create_dataset('values',data=[2.,4.,6.]); f.close()",
                str(data),
            ],
        )
        responses = await acceptance.session(
            "downloaded-hdf5-real-query",
            ["mcp-server", "hdf5"],
            [
                ("open_file", {"path": str(data)}),
                (
                    "hdf5_aggregate_stats",
                    {"paths": "values", "stats": "mean,sum,count"},
                ),
            ],
        )
        text = responses[-1]["structuredContent"]["result"]
        statistics = {
            key: float(value)
            for key, value in re.findall(
                r"^\s+(mean|sum|count): ([0-9.]+)$", text, re.MULTILINE
            )
        }
        assert statistics == {"mean": 4.0, "sum": 12.0, "count": 3.0}, responses
        assert "FULL DATA: 3 of 3 elements" in text
        expect([f"skill/{skill}", "server/hdf5"])
        acceptance.environment["CLIO_KIT_OFFLINE"] = "1"
        await acceptance.session(
            "cached-hdf5-real-query",
            ["mcp-server", "hdf5"],
            [("open_file", {"path": str(data)})],
        )
        del acceptance.environment["CLIO_KIT_OFFLINE"]
        expect([f"skill/{skill}", "server/hdf5"])
        command(
            "selected-workflow",
            "plugin",
            "install",
            "clio-scientific-io",
            "--client",
            "codex",
            "--project",
            str(output / "project"),
        )
        keys = {"server/hdf5"} | {
            f"skill/{name}"
            for name, record in index["skills"].items()
            if record["bundle"] == "clio-scientific-io"
        }
        expect(keys)
        native = output / "selected-native"
        command(
            "selected-native-package",
            "plugin",
            "fetch",
            "clio-dataset-report",
            "--target",
            str(native),
        )
        packages = json.loads((native / ".claude-plugin/marketplace.json").read_text())[
            "plugins"
        ]
        names = {p["name"] for p in packages}
        assert names == {
            "clio-dataset-report",
            "clio-hdf5",
            "clio-pandas",
            "clio-plot",
            "clio-agents",
        }
        for name in names:
            keys.add(f"package/{name}")
            keys.update(f"skill/{skill}" for skill in index["packages"][name]["skills"])
        expect(keys)
        acceptance.command(
            "validate-selected-native",
            ["claude", "plugin", "validate", str(native), "--strict"],
            cwd=output,
        )
        acceptance.command(
            "register-selected-native",
            ["claude", "plugin", "marketplace", "add", str(native)],
            cwd=output,
        )
        acceptance.command(
            "install-selected-native",
            ["claude", "plugin", "install", "clio-dataset-report@clio-kit"],
            cwd=output,
        )
        expect(keys)
        summary = {
            "wheel_bytes": wheel.stat().st_size,
            "sdist_bytes": next(dist.glob("clio_kit-*.tar.gz")).stat().st_size,
            "component_requests": requests,
            "downloaded_payload_bytes": sum(
                index["artifacts"][key]["size"] for key in keys
            ),
            "hdf5_only_server_downloaded": True,
            "offline_repeat": True,
            "native_packages": sorted(names),
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(
            "PASS selective downloads, real HDF5 query, cached reuse, workflow composition and native plugin installation",
            flush=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(verify(args.output.resolve()))
