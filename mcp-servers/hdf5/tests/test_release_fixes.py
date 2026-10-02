"""Regression tests for the pre-release acceptance findings."""

import asyncio
import json

import h5py
import numpy as np
import pytest
from fastmcp import Client

from hdf5_mcp.server import mcp


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "a.h5"
    with h5py.File(path, "w") as file:
        file.attrs["version"] = 3
        file.attrs["title"] = "acceptance"
        file.create_dataset("temp", data=np.arange(120, dtype="f8").reshape(10, 12))
        file.create_dataset("ints", data=np.arange(1500, dtype="i4"))
        file.create_dataset("vec", data=np.array([1.0, np.nan, 4.0]))
        file.create_dataset("scalar", data=42.5)
    return path


def run(source, exercise):
    async def session():
        async with Client(mcp) as client:
            if source is not None:
                await client.call_tool("open_file", {"path": str(source)})
            try:
                return await exercise(client)
            finally:
                await client.call_tool("close_file", {}, raise_on_error=False)

    return asyncio.run(session())


async def text(client, tool, arguments, error=False):
    result = await client.call_tool(tool, arguments, raise_on_error=False)
    assert result.is_error is error, result
    return result.content[0].text


def test_server_version_matches_release():
    async def exercise(client):
        return client.server_info.version

    assert run(None, exercise) == "2.2.6"


def test_prompts_render(source):
    async def exercise(client):
        prompts = await client.list_prompts()
        assert len(prompts) == 4
        for prompt in prompts:
            arguments = {argument.name: "x" for argument in prompt.arguments}
            rendered = await client.get_prompt(prompt.name, arguments)
            assert rendered.messages[0].content.text

    run(None, exercise)


def test_open_file_rejects_non_hdf5_and_unsupported_mode(source, tmp_path):
    csv = tmp_path / "c.csv"
    csv.write_text("a,b\n1,2\n")

    async def exercise(client):
        assert "not an HDF5 file" in await text(
            client, "open_file", {"path": str(csv)}, error=True
        )
        assert "read-only" in await text(
            client, "open_file", {"path": str(source), "mode": "w"}, error=True
        )
        assert "No file currently open" in await text(
            client, "get_filename", {}, error=True
        )

    run(None, exercise)


def test_reads_return_values_with_stated_bound(source):
    async def exercise(client):
        full = await text(client, "read_full_dataset", {"path": "/temp"})
        assert "shape (10, 12)" in full and "118.0, 119.0]]" in full
        assert "Values: 42.5" in await text(
            client, "read_full_dataset", {"path": "/scalar"}
        )
        big = await text(client, "read_full_dataset", {"path": "/ints"})
        assert "first 1000 of 1500" in big and "500 omitted" in big
        assert "998, 999]" in big and "1000]" not in big
        part = await text(
            client,
            "read_partial_dataset",
            {"path": "/temp", "start": "2,3", "count": "2,4"},
        )
        assert "[[27.0, 28.0, 29.0, 30.0], [39.0, 40.0, 41.0, 42.0]]" in part
        assert "out of range" in await text(
            client,
            "read_partial_dataset",
            {"path": "/temp", "start": "10,0"},
            error=True,
        )

    run(source, exercise)


def test_numpy_export_writes_file_and_json_nan_is_null(source, tmp_path):
    npy, vec = tmp_path / "temp.npy", tmp_path / "vec.json"

    async def exercise(client):
        await text(
            client,
            "export_dataset",
            {"path": "/temp", "output_path": str(npy), "export_format": "numpy"},
        )
        await text(
            client,
            "export_dataset",
            {"path": "/vec", "output_path": str(vec), "export_format": "json"},
        )
        assert "No file written" in await text(
            client, "export_dataset", {"path": "/temp", "export_format": "numpy"}
        )

    run(source, exercise)
    assert np.load(npy).shape == (10, 12)
    assert json.loads(vec.read_text(), parse_constant=pytest.fail)["data"] == [
        1.0,
        None,
        4.0,
    ]


def test_metadata_resource_serializes_integer_attributes(source, monkeypatch):
    monkeypatch.chdir(source.parent)

    async def exercise(client):
        contents = await client.read_resource(f"hdf5://{source.name}/metadata")
        return json.loads(contents[0].text)["attrs"]

    assert run(None, exercise) == {"version": 3, "title": "acceptance"}


def test_low_severity_truthfulness(source):
    async def exercise(client):
        assert "Debug" not in await text(client, "analyze_dataset_structure", {})
        assert "Dataset not found: /nope" in await text(
            client, "identify_io_bottlenecks", {"analysis_paths": ["/nope"]}, error=True
        )

    run(source, exercise)
