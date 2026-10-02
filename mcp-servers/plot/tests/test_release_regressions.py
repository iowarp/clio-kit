"""Regressions from the 2026-10-02 pre-release acceptance test, driven over MCP."""

import asyncio
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from fastmcp import Client

from plot_mcp.server import mcp

RAW = "configuration,trial,runtime_s\n" + "".join(
    f"{name},{trial},{value}\n"
    for name, values in (("tuned", (9, 10, 8, 11)), ("baseline", (12, 13, 11, 14)))
    for trial, value in enumerate(values, 1)
)
CALLS = {
    "line_plot": {"x_column": "trial", "y_column": "runtime_s"},
    "bar_plot": {"x_column": "configuration", "y_column": "runtime_s"},
    "scatter_plot": {"x_column": "trial", "y_column": "runtime_s"},
    "histogram_plot": {"column": "runtime_s"},
    "heatmap_plot": {},
    "plot_timeseries": {"x_column": "trial", "y_columns": ["runtime_s"]},
}


def call(tool, arguments):
    async def exchange():
        async with Client(mcp) as client:
            return await client.call_tool(tool, arguments, raise_on_error=False)

    return asyncio.run(exchange())


@pytest.fixture
def runs(tmp_path):
    path = tmp_path / "runs.csv"
    path.write_text(RAW)
    return str(path)


def source(tool, path):
    return {"data_path" if tool == "plot_timeseries" else "file_path": path}


@pytest.mark.parametrize("tool", CALLS)
@pytest.mark.parametrize("suffix,magic", [(".svg", b"<?xml"), (".pdf", b"%PDF")])
def test_vector_output_succeeds_without_a_preview(tool, suffix, magic, runs, tmp_path):
    output = tmp_path / f"plot{suffix}"
    result = call(tool, CALLS[tool] | source(tool, runs) | {"output_path": str(output)})
    assert not result.is_error, result.content
    assert output.read_bytes().startswith(magic)
    assert result.structured_content["output_path"] == str(output)
    assert str(output) in result.content[0].text


@pytest.fixture
def bars(monkeypatch):
    """Record what bar_plot actually draws."""
    drawn = {}
    real_bar = plt.bar

    def bar(x, height, **kwargs):
        drawn.update(zip(x, height))
        return real_bar(x, height, **kwargs)

    monkeypatch.setattr(plt, "bar", bar)
    return drawn


def test_bar_plot_averages_repeated_categories(runs, tmp_path, bars):
    arguments = CALLS["bar_plot"] | {"output_path": str(tmp_path / "bar.png")}
    result = call("bar_plot", arguments | {"file_path": runs})
    assert result.structured_content["aggregated"] is True
    assert bars == {"baseline": 12.5, "tuned": 9.5}  # the mean, not the max 14 / 11


def test_bar_plot_leaves_a_grouped_file_as_is(tmp_path, bars):
    grouped = tmp_path / "runs_grouped.csv"
    grouped.write_text("configuration,runtime_s\ntuned,9.5\nbaseline,12.5\n")
    arguments = CALLS["bar_plot"] | {"output_path": str(tmp_path / "bar.png")}
    result = call("bar_plot", arguments | {"file_path": str(grouped)})
    assert result.structured_content["aggregated"] is False
    assert list(bars.items()) == [("tuned", 9.5), ("baseline", 12.5)]  # file order


@pytest.mark.parametrize("tool", ["line_plot", "bar_plot", "scatter_plot"])
def test_text_y_column_is_rejected(tool, runs, tmp_path):
    arguments = {"x_column": "trial", "y_column": "configuration"}
    arguments |= {"file_path": runs, "output_path": str(tmp_path / "bad.png")}
    result = call(tool, arguments)
    assert result.is_error
    assert "'configuration' is not numeric" in result.content[0].text
    assert not (tmp_path / "bad.png").exists()


def test_relative_output_path_is_reported_absolute(runs, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = call("line_plot", CALLS["line_plot"] | {"file_path": runs})
    assert result.structured_content["output_path"] == str(tmp_path / "line_plot.png")
    assert (tmp_path / "line_plot.png").exists()


def test_server_reports_the_packaged_version():
    async def version():
        async with Client(mcp) as client:
            return client.server_info.version

    descriptor = Path(__file__).parents[1] / "clio-server.toml"
    expected = tomllib.loads(descriptor.read_text())["version"]
    assert asyncio.run(version()) == expected
