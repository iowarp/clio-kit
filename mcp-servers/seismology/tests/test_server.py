"""In-memory MCP tests: the tools exist and run on real SAC fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from seismology_mcp.server import mcp


@pytest.mark.asyncio
async def test_tools_registered() -> None:
    async with Client(mcp) as client:
        tools = {t.name for t in await client.list_tools()}
    assert {"inspect_archive", "compute_trace_statistics", "plot_traces"} <= tools


@pytest.mark.asyncio
async def test_resource_and_prompt_registered() -> None:
    async with Client(mcp) as client:
        resources = {str(r.uri) for r in await client.list_resources()}
        prompts = {p.name for p in await client.list_prompts()}
    assert "seismology://capabilities" in resources
    assert "analyze_sac_archive" in prompts


@pytest.mark.asyncio
async def test_inspect_archive_on_archive(sac_archive: Path) -> None:
    async with Client(mcp) as client:
        result = await client.call_tool(
            "inspect_archive", {"filepath": str(sac_archive)}
        )
    data = result.data
    assert data["status"] == "success"
    assert data["sac_trace_count"] == 3  # notes.txt ignored
    assert "P" in data["phases"]
    assert "S" in data["phases"]


@pytest.mark.asyncio
async def test_inspect_archive_member_filter(sac_archive: Path) -> None:
    async with Client(mcp) as client:
        result = await client.call_tool(
            "inspect_archive", {"filepath": str(sac_archive), "member_filter": "bhn"}
        )
    assert result.data["sac_trace_count"] == 1


@pytest.mark.asyncio
async def test_inspect_single_sac_file(sac_file: Path) -> None:
    async with Client(mcp) as client:
        result = await client.call_tool("inspect_archive", {"filepath": str(sac_file)})
    assert result.data["sac_trace_count"] == 1
    assert result.data["sample_members"] == ["IU.ANMO.00.BHZ.sac"]


@pytest.mark.asyncio
async def test_compute_trace_statistics(sac_archive: Path) -> None:
    async with Client(mcp) as client:
        result = await client.call_tool(
            "compute_trace_statistics", {"filepath": str(sac_archive), "max_traces": 2}
        )
    data = result.data
    assert data["status"] == "success"
    assert data["sac_trace_count"] == 3
    assert data["traces_analyzed"] == 2
    assert data["traces_truncated"] is True
    first = data["traces"][0]
    for key in ("min", "max", "mean", "std", "peak_abs", "npts", "delta_s"):
        assert key in first
    assert first["peak_abs"] >= abs(first["max"]) - 1e-6


@pytest.mark.asyncio
async def test_compute_statistics_on_single_file(sac_file: Path) -> None:
    async with Client(mcp) as client:
        result = await client.call_tool(
            "compute_trace_statistics", {"filepath": str(sac_file)}
        )
    data = result.data
    assert data["traces_analyzed"] == 1
    assert data["traces"][0]["npts"] == 200
    assert data["traces"][0]["delta_s"] == pytest.approx(0.01, rel=1e-4)


@pytest.mark.asyncio
async def test_plot_traces(sac_archive: Path, tmp_path: Path) -> None:
    out = tmp_path / "traces.png"
    async with Client(mcp) as client:
        result = await client.call_tool(
            "plot_traces",
            {"filepath": str(sac_archive), "max_traces": 3, "output_path": str(out)},
        )
    data = result.data
    assert data["status"] == "success"
    assert data["traces_plotted"] == 3
    assert out.is_file() and out.stat().st_size > 0


@pytest.mark.asyncio
async def test_plot_traces_default_output(
    sac_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Run with cwd set to a temp dir so the default artifact path does not
    # pollute the server directory.
    monkeypatch.chdir(tmp_path)
    async with Client(mcp) as client:
        result = await client.call_tool("plot_traces", {"filepath": str(sac_file)})
    out = Path(result.data["output_path"])
    assert out.is_file() and out.stat().st_size > 0
    assert tmp_path in out.parents


@pytest.mark.asyncio
async def test_missing_file_raises_tool_error() -> None:
    async with Client(mcp) as client:
        with pytest.raises(ToolError):
            await client.call_tool("inspect_archive", {"filepath": "/no/such/file.sac"})


@pytest.mark.asyncio
async def test_filter_with_no_match_raises(sac_archive: Path) -> None:
    async with Client(mcp) as client:
        with pytest.raises(ToolError):
            await client.call_tool(
                "compute_trace_statistics",
                {"filepath": str(sac_archive), "member_filter": "zzz-nope"},
            )


def _sac_with_header(station: bytes = b"", phase: bytes = b"") -> bytes:
    from .conftest import make_sac_bytes

    payload = bytearray(make_sac_bytes([0.0, 1.0, -1.0]))
    payload[440:448] = station.ljust(8)  # KSTNM
    payload[480:488] = phase.ljust(8)  # KA (first-arrival phase)
    return bytes(payload)


@pytest.mark.asyncio
async def test_station_and_phase_come_from_the_sac_header(tmp_path: Path) -> None:
    event_dir = tmp_path / "ev1"
    event_dir.mkdir()
    labelled = event_dir / "one.sac"
    labelled.write_bytes(_sac_with_header(b"AAA", b"Pn"))
    unset = event_dir / "XX.BBB.00.BHN.sac"
    unset.write_bytes(_sac_with_header(b"-12345"))
    async with Client(mcp) as client:
        inspected = (
            await client.call_tool("inspect_archive", {"filepath": str(labelled)})
        ).data
        stats = (
            await client.call_tool(
                "compute_trace_statistics", {"filepath": str(labelled)}
            )
        ).data["traces"][0]
        fallback = (
            await client.call_tool("compute_trace_statistics", {"filepath": str(unset)})
        ).data["traces"][0]
    assert (inspected["stations"], inspected["phases"]) == (["AAA"], ["Pn"])
    assert inspected["station_sources"] == inspected["phase_sources"] == ["header"]
    assert (stats["station"], stats["station_source"]) == ("AAA", "header")
    assert (stats["phase"], stats["phase_source"]) == ("Pn", "header")
    assert (fallback["station"], fallback["station_source"]) == ("BBB", "path")
    assert fallback["phase_source"] == "path"


@pytest.mark.asyncio
async def test_inspect_rejects_a_sac_named_file_that_is_not_sac(
    tmp_path: Path, sac_file: Path
) -> None:
    import tarfile

    garbage = tmp_path / "garbage.sac"
    garbage.write_bytes(b"not a sac file at all" * 40)  # longer than a SAC header
    archive_path = tmp_path / "mixed.tar"
    with tarfile.open(archive_path, "w") as archive:
        archive.add(garbage, arcname="garbage.sac")
        archive.add(sac_file, arcname="P/IU.ANMO.00.BHZ.sac")
    async with Client(mcp) as client:
        with pytest.raises(ToolError, match="not a SAC file"):
            await client.call_tool("inspect_archive", {"filepath": str(garbage)})
        with pytest.raises(ToolError, match="not a SAC file"):
            await client.call_tool(
                "compute_trace_statistics", {"filepath": str(garbage)}
            )
        mixed = (
            await client.call_tool("inspect_archive", {"filepath": str(archive_path)})
        ).data
    assert mixed["sac_trace_count"] == 1
    assert mixed["sample_members"] == ["P/IU.ANMO.00.BHZ.sac"]
    assert [item["member"] for item in mixed["invalid_members"]] == ["garbage.sac"]


def test_server_version_matches_release_manifest() -> None:
    import re

    manifest = (Path(__file__).parents[1] / "clio-server.toml").read_text()
    assert mcp.version == re.search(r'^version = "(.+)"', manifest, re.M).group(1)
