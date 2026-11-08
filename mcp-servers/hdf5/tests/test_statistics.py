"""Check real dataset coverage and the statistics exposed through MCP."""

import asyncio

import h5py
from fastmcp import Client

from hdf5_mcp.server import mcp
from hdf5_mcp.statistics import compute_dataset_stats, format_statistics


STATS = ["mean", "min", "max", "sum", "count"]


def write_data(path):
    with h5py.File(path, "w") as file:
        # Real 534 MiB logical dataset, without allocating 534 MiB in a test.
        file.create_dataset("large", shape=(70_000_000,), dtype="f8", fillvalue=1)
        file.create_dataset("small", data=[2.0, 4.0])
        file.create_dataset("single", data=[9.0])


def test_large_sample_has_coverage_and_no_misleading_overall_totals(tmp_path):
    path = tmp_path / "stats.h5"
    write_data(path)
    with h5py.File(path, "r") as file:
        large = compute_dataset_stats(file, "large", STATS)
        small = compute_dataset_stats(file, "small", STATS)
    assert large["sampled"]
    assert large["elements_processed"] == large["count"] == 700_000
    assert large["total_elements"] == 70_000_000
    assert large["sum"] == 700_000 and large["mean"] == 1
    assert not small["sampled"]
    summary = format_statistics({"large": large, "small": small}, STATS)
    assert "SAMPLED: 700,000 of 70,000,000 elements (1.00% coverage)" in summary
    assert "sum/count are not full-dataset totals" in summary
    assert "FULL DATA: 2 of 2 elements" in summary
    assert "Cross-dataset aggregation omitted" in summary
    assert "Total sum:" not in summary and "Global max:" not in summary


def test_full_data_aggregation_weights_mean_even_without_count_requested(tmp_path):
    path = tmp_path / "stats.h5"
    write_data(path)
    with h5py.File(path, "r") as file:
        results = {
            name: compute_dataset_stats(file, name, ["mean"])
            for name in ["small", "single"]
        }
    summary = format_statistics(results, ["mean"])
    assert "Overall mean: 5.000000" in summary  # (2 + 4 + 9) / 3
    assert "SAMPLED:" not in summary


def test_sampling_warning_reaches_actual_mcp_response(tmp_path):
    path = tmp_path / "stats.h5"
    write_data(path)

    async def exercise():
        async with Client(mcp) as client:
            await client.call_tool("open_file", {"path": str(path)})
            try:
                response = await client.call_tool(
                    "hdf5_aggregate_stats",
                    {
                        "paths": "large,small",
                        "stats": "mean,sum,count,min,max",
                    },
                )
                text = "\n".join(
                    item.text for item in response.content if item.type == "text"
                )
                assert "SAMPLED: 700,000 of 70,000,000" in text
                assert "sum/count are not full-dataset totals" in text
                assert "Cross-dataset aggregation omitted" in text
            finally:
                await client.call_tool("close_file", {})

    asyncio.run(exercise())
