"""In-memory MCP tests: the tools exist and run on real synthetic catalogs.

The seismic science (Mc, b-value, Omori decay) is verified on synthetic
catalogs; the data-vs-verdict separation is asserted; and rendering is exercised
on disk. Nothing here touches the network.
"""

from __future__ import annotations

import json

from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from seismology_mcp.implementation import (
    _b_value,
    _magnitude_of_completeness,
    _omori_decay,
)
from seismology_mcp.server import mcp

from .conftest import gr_magnitudes, mainshock_aftershock_events


# ----------------------------- wiring ------------------------------------


@pytest.mark.asyncio
async def test_tools_registered() -> None:
    async with Client(mcp) as client:
        tools = {t.name for t in await client.list_tools()}
    assert {"analyze_sequence", "plot_sequence"} <= tools


@pytest.mark.asyncio
async def test_query_catalog_is_gone() -> None:
    async with Client(mcp) as client:
        tools = {t.name for t in await client.list_tools()}
    assert "query_catalog" not in tools


@pytest.mark.asyncio
async def test_resource_and_prompt_registered() -> None:
    async with Client(mcp) as client:
        resources = {str(r.uri) for r in await client.list_resources()}
        prompts = {p.name for p in await client.list_prompts()}
    assert "seismology://capabilities" in resources
    assert "characterize_sequence" in prompts


@pytest.mark.asyncio
async def test_capabilities_resource_separates_the_two_tool_families() -> None:
    """Reading the resource, not just listing it: an agent needs the content."""
    async with Client(mcp) as client:
        result = await client.read_resource("seismology://capabilities")

    payload = json.loads(result[0].text)
    assert payload["waveform_tools"] == [
        "inspect_archive",
        "compute_trace_statistics",
        "plot_traces",
    ]
    assert payload["catalog_tools"] == ["analyze_sequence", "plot_sequence"]
    assert ".sac" in payload["accepted_inputs"]
    assert ".csv" in payload["accepted_catalog_inputs"]


@pytest.mark.asyncio
async def test_characterize_sequence_prompt_names_the_catalog_and_both_steps() -> None:
    """The ported prompt must still carry the workflow it encoded."""
    async with Client(mcp) as client:
        rendered = await client.get_prompt(
            "characterize_sequence", {"catalog_path": "/data/quakes.geojson"}
        )

    text = " ".join(m.content.text for m in rendered.messages)
    assert "/data/quakes.geojson" in text
    assert "analyze_sequence" in text
    assert "plot_sequence" in text


# ----------------------------- science -----------------------------------


def test_b_value_recovers_known_slope() -> None:
    mags = gr_magnitudes(b=1.0, mc=2.0)
    stats = _b_value(mags, mc=2.0)
    assert stats["b_value"] is not None
    assert 0.85 <= stats["b_value"] <= 1.15, stats


def test_b_value_declines_on_too_few_events() -> None:
    assert _b_value([5.0, 5.1, 5.2], mc=4.5)["b_value"] is None


def test_mc_estimates_near_true_completeness() -> None:
    mags = gr_magnitudes(b=1.0, mc=2.0)
    mc = _magnitude_of_completeness(mags)
    assert mc is not None and abs(mc - 2.0) <= 0.4, mc


def test_omori_decay_is_monotonic_for_aftershocks() -> None:
    events = mainshock_aftershock_events()
    t0 = max(events, key=lambda e: e["mag"])["time_ms"]
    decay = _omori_decay(events, t0)
    rates = [b["rate_per_day"] for b in decay["rate_buckets"][:5]]
    assert rates[0] > rates[-1]  # decaying
    assert decay["omori_p_estimate"] is not None and decay["omori_p_estimate"] > 0


# ----------------- analyze_sequence (data, not verdict) ------------------


@pytest.mark.asyncio
async def test_analyze_returns_stats_not_classification(
    aftershock_geojson: Path,
) -> None:
    async with Client(mcp) as client:
        result = await client.call_tool(
            "analyze_sequence", {"catalog_path": str(aftershock_geojson)}
        )
    res = result.data
    st = res["statistics"]
    # data is present
    assert st["largest_event"]["magnitude"] == 6.5
    assert st["bath_gap"] is not None and st["bath_gap"] > 1.0
    # The fixture has one smaller event simultaneous with the largest event.
    assert st["events_after_largest"] == res["event_count"] - 2
    assert st["fraction_after_largest"] == round(
        (res["event_count"] - 2) / res["event_count"], 3
    )
    assert st["temporal_decay"]["omori_p_estimate"] is not None
    # the tool must NOT make the judgment
    blob = json.dumps(res).lower()
    assert "sequence_type" not in blob
    assert "classification" not in blob
    assert "aftershock" not in blob and "swarm" not in blob


@pytest.mark.asyncio
@pytest.mark.parametrize("times, after", [("0,1,2", 0), ("1,2,0", 2), ("1,2,2", 0)])
async def test_largest_event_is_not_its_own_aftershock(tmp_path, times, after):
    path = tmp_path / "events.csv"
    path.write_text(
        "time,magnitude\n"
        + "".join(
            f"2026-01-01T00:00:0{t},{m}\n" for t, m in zip(times.split(","), [2, 3, 5])
        )
    )
    async with Client(mcp) as client:
        result = await client.call_tool("analyze_sequence", {"catalog_path": str(path)})
    stats = result.data["statistics"]
    assert stats["events_after_largest"] == after
    assert stats["fraction_after_largest"] == round(after / 3, 3)
    assert stats["temporal_decay"]["rate_buckets"][0]["count"] == after


@pytest.mark.asyncio
async def test_analyze_reads_feature_collection(
    aftershock_feature_collection: Path,
) -> None:
    async with Client(mcp) as client:
        result = await client.call_tool(
            "analyze_sequence", {"catalog_path": str(aftershock_feature_collection)}
        )
    assert result.data["statistics"]["largest_event"]["magnitude"] == 6.5


@pytest.mark.asyncio
async def test_analyze_reads_csv(aftershock_csv: Path) -> None:
    async with Client(mcp) as client:
        result = await client.call_tool(
            "analyze_sequence", {"catalog_path": str(aftershock_csv)}
        )
    res = result.data
    assert res["event_count"] == 25
    assert res["statistics"]["largest_event"]["magnitude"] == 6.5


@pytest.mark.asyncio
async def test_analyze_recovers_b_value_from_gr_catalog(gr_geojson: Path) -> None:
    async with Client(mcp) as client:
        result = await client.call_tool(
            "analyze_sequence", {"catalog_path": str(gr_geojson)}
        )
    st = result.data["statistics"]
    assert st["b_value"] is not None and 0.85 <= st["b_value"] <= 1.15


@pytest.mark.asyncio
async def test_analyze_empty_catalog_is_graceful(empty_geojson: Path) -> None:
    async with Client(mcp) as client:
        result = await client.call_tool(
            "analyze_sequence", {"catalog_path": str(empty_geojson)}
        )
    res = result.data
    assert res["ok"] and res["event_count"] == 0 and "statistics" not in res


@pytest.mark.asyncio
async def test_analyze_missing_file_raises_tool_error() -> None:
    async with Client(mcp) as client:
        with pytest.raises(ToolError):
            await client.call_tool(
                "analyze_sequence", {"catalog_path": "/no/such/catalog.json"}
            )


@pytest.mark.asyncio
async def test_analyze_unsupported_extension_raises(tmp_path: Path) -> None:
    bad = tmp_path / "catalog.txt"
    bad.write_text("not a catalog", encoding="utf-8")
    async with Client(mcp) as client:
        with pytest.raises(ToolError):
            await client.call_tool("analyze_sequence", {"catalog_path": str(bad)})


# ----------------------------- plot_sequence -----------------------------


@pytest.mark.asyncio
async def test_plot_renders_figure(aftershock_geojson: Path, tmp_path: Path) -> None:
    out = tmp_path / "seq.png"
    async with Client(mcp) as client:
        result = await client.call_tool(
            "plot_sequence",
            {
                "catalog_path": str(aftershock_geojson),
                "title": "t",
                "mc": 3.0,
                "b_value": 1.0,
                "output_path": str(out),
            },
        )
    res = result.data
    assert res["ok"] and out.stat().st_size > 5000
    assert res["panels"] == ["epicenter_map", "gutenberg_richter", "temporal_evolution"]


@pytest.mark.asyncio
async def test_plot_default_output(
    aftershock_csv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    async with Client(mcp) as client:
        result = await client.call_tool(
            "plot_sequence", {"catalog_path": str(aftershock_csv)}
        )
    out = Path(result.data["figure_path"])
    assert out.is_file() and out.stat().st_size > 5000
    assert tmp_path in out.parents


@pytest.mark.asyncio
async def test_plot_rejects_empty_catalog(empty_geojson: Path) -> None:
    async with Client(mcp) as client:
        with pytest.raises(ToolError):
            await client.call_tool(
                "plot_sequence", {"catalog_path": str(empty_geojson)}
            )


@pytest.mark.parametrize("end_day", [1, 2, 4, 8, 16, 32])
def test_omori_counts_every_event_at_the_final_bucket_boundary(end_day) -> None:
    days = [0.5, end_day, end_day]  # simultaneous events at the endpoint count too
    events = [{"time_ms": int(day * 86_400_000)} for day in days]
    buckets = _omori_decay(events, 0)["rate_buckets"]
    assert sum(bucket["count"] for bucket in buckets) == len(events)
    assert buckets[-1]["day_end"] == end_day
    assert all(bucket["day_end"] > bucket["day_start"] for bucket in buckets)


def test_omori_last_bucket_is_truncated_to_the_catalogue_end() -> None:
    """A steady rate must not read as a decay because the catalogue stops mid-bucket."""
    t0 = 1_700_000_000_000
    events = [{"mag": 6.0, "time_ms": t0}] + [
        {"mag": 3.0, "time_ms": t0 + k * 7_200_000}
        for k in range(1, 67)  # 12 per day; ends 5.5 days in
    ]
    decay = _omori_decay(events, t0)
    buckets = decay["rate_buckets"]
    assert [(b["day_start"], b["day_end"]) for b in buckets] == [
        (0, 1),
        (1, 2),
        (2, 4),
        (4, 5.5),
    ]
    assert sum(b["count"] for b in buckets) == 66
    assert all(11 <= b["rate_per_day"] <= 13 for b in buckets)
    assert decay["decay_ratio_first_to_last"] < 1.0  # was 2.3 with a full 4-day divisor


@pytest.mark.asyncio
async def test_non_catalogue_inputs_and_bad_mag_bin_are_errors(
    tmp_path: Path, aftershock_csv: Path
) -> None:
    sites = tmp_path / "sites.csv"
    sites.write_text("site,lat,lon\nchi,41.8781,-87.6298\n", encoding="utf-8")
    points = tmp_path / "points.geojson"
    points.write_text(
        '{"type": "FeatureCollection", "features": [{"type": "Feature", '
        '"properties": {"name": "chi"}, "geometry": null}]}',
        encoding="utf-8",
    )
    async with Client(mcp) as client:
        with pytest.raises(ToolError, match="no magnitude column"):
            await client.call_tool("analyze_sequence", {"catalog_path": str(sites)})
        with pytest.raises(ToolError, match="'mag' property"):
            await client.call_tool("analyze_sequence", {"catalog_path": str(points)})
        with pytest.raises(ToolError, match="mag_bin must be greater than 0"):
            await client.call_tool(
                "analyze_sequence", {"catalog_path": str(aftershock_csv), "mag_bin": 0}
            )
