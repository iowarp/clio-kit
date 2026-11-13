"""Tests for the CLIO structured-file provider (hdf5_mcp.provider).

Organized around the conformance checks in section 8 of the draft RFC
"CLIO Structured File Integration", plus HDF5-specific edge cases.
"""

import json
import time

import h5py
import numpy as np
import pytest

from hdf5_mcp.provider import (
    ERROR_CODES,
    HDF5Provider,
    ProviderError,
    StructuredFileProvider,
)


@pytest.fixture
def provider():
    return HDF5Provider()


@pytest.fixture
def sample(tmp_path):
    """A small file covering every node kind the provider maps."""
    path = tmp_path / "sample.h5"
    with h5py.File(path, "w") as f:
        f.attrs["title"] = "sample"
        entry = f.create_group("entry_0000")
        entry.attrs["NX_class"] = "NXentry"
        results = entry.create_group("results")
        intensity = results.create_dataset("I", data=np.arange(100, dtype="f8"))
        intensity.attrs["units"] = "counts"
        intensity.attrs["big"] = np.arange(1000)
        results.create_dataset("short", data=np.array([1, 2, 3], dtype="i4"))
        results.create_dataset(
            "image",
            data=np.arange(64 * 64, dtype="f4").reshape(64, 64),
            chunks=(16, 16),
            compression="gzip",
        )
        results.create_dataset("q.values", data=np.linspace(0.0, 1.0, 4))
        f["scalar"] = 42
        f["text"] = "x" * 1000
        f["labels"] = np.array(["alpha", "beta"], dtype=h5py.string_dtype())
        f["table"] = np.array(
            [(1.0, 2), (3.0, 4), (5.0, 6)], dtype=[("x", "f8"), ("n", "i4")]
        )
        f["flags"] = np.array([True, False, True, True])
        f["empty"] = h5py.Empty("f8")
        f["with_nan"] = np.array([[1.0, np.nan], [np.inf, 3.0]])
        f["named_type"] = np.dtype("f4")
        f.create_group("empty_group")
        f["soft"] = h5py.SoftLink("/entry_0000/results/I")
        f["broken"] = h5py.SoftLink("/nowhere")
        f["external"] = h5py.ExternalLink("missing.h5", "/x")
    return path


def _assert_released(path):
    """The provider holds no handle: the file opens for writing again."""
    with h5py.File(path, "a"):
        pass


# -- protocol shape ------------------------------------------------------


def test_satisfies_protocol(provider):
    assert isinstance(provider, StructuredFileProvider)
    assert provider.id == "hdf5"
    assert "application/x-hdf5" in provider.media_types


# -- detect ----------------------------------------------------------------


def test_detect_hdf5_file(provider, sample):
    assert provider.detect(sample)


def test_detect_ignores_extension(provider, tmp_path):
    fake = tmp_path / "not_really.h5"
    fake.write_bytes(b"this is not hdf5" * 100)
    assert not provider.detect(fake)

    renamed = tmp_path / "data.bin"
    with h5py.File(renamed, "w") as f:
        f["x"] = 1
    assert provider.detect(renamed)


def test_detect_finds_signature_after_user_block(provider, tmp_path):
    path = tmp_path / "userblock.h5"
    with h5py.File(path, "w", userblock_size=1024) as f:
        f["x"] = 1
    with open(path, "r+b") as f:
        f.write(b"user block text")
    assert provider.detect(path)


def test_detect_missing_and_directory(provider, tmp_path):
    assert not provider.detect(tmp_path / "missing.h5")
    assert not provider.detect(tmp_path)


# -- children --------------------------------------------------------------


def test_children_is_one_level(provider, sample):
    nodes = provider.children(sample, "/")
    by_name = {n["name"]: n for n in nodes}
    assert "entry_0000" in by_name
    assert "results" not in by_name
    assert all(n["locator"].count("/") == 1 for n in nodes)
    _assert_released(sample)


def test_children_node_fields(provider, sample):
    by_name = {n["name"]: n for n in provider.children(sample, "/")}

    assert by_name["entry_0000"]["kind"] == "container"
    assert by_name["entry_0000"]["has_children"] is True
    assert by_name["empty_group"]["has_children"] is False

    assert by_name["scalar"]["kind"] == "scalar"
    assert by_name["scalar"]["shape"] == ()
    assert by_name["text"]["kind"] == "text"
    assert by_name["text"]["dtype"] == "string(utf-8)"
    assert by_name["labels"]["kind"] == "array"
    assert by_name["table"]["kind"] == "table"
    assert by_name["table"]["dtype"] == "compound"
    assert by_name["flags"]["dtype"] == "bool"
    assert by_name["empty"]["shape"] is None
    assert by_name["named_type"]["kind"] == "blob"

    # A valid soft link resolves to its target; broken links stay links.
    assert by_name["soft"]["kind"] == "array"
    assert by_name["soft"]["shape"] == (100,)
    assert by_name["broken"]["kind"] == "link"
    assert by_name["external"]["kind"] == "link"


def test_children_reads_no_data(provider, sample, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("children() read dataset values")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", refuse)
    monkeypatch.setattr(h5py.Dataset, "read_direct", refuse)
    provider.children(sample, "/")
    provider.children(sample, "/entry_0000/results")


def test_children_bounded_time_on_wide_group(provider, tmp_path):
    path = tmp_path / "wide.h5"
    with h5py.File(path, "w") as f:
        for i in range(2000):
            f.create_dataset(f"d{i:05d}", shape=(1000, 1000), dtype="f8")
    start = time.monotonic()
    nodes = provider.children(path, "/")
    assert len(nodes) == 2000
    assert time.monotonic() - start < 10


def test_children_of_dataset_is_unsupported(provider, sample):
    with pytest.raises(ProviderError) as err:
        provider.children(sample, "/scalar")
    assert err.value.code == "unsupported"


# -- describe --------------------------------------------------------------


def test_describe_1d_preview_head_and_tail(provider, sample):
    d = provider.describe(sample, "/entry_0000/results/I")
    assert d["node"]["kind"] == "array"
    assert d["preview"]["head"] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert d["preview"]["tail"] == [95.0, 96.0, 97.0, 98.0, 99.0]
    assert d["preview"]["count"] == 100
    assert d["render_hint"] == "line"
    assert d["attributes"]["units"] == "counts"
    assert d["attributes"]["big"] == {
        "shape": (1000,),
        "dtype": "int64",
        "omitted": True,
    }
    assert d["convention"] is None


def test_describe_short_1d_shows_all_values(provider, sample):
    d = provider.describe(sample, "/entry_0000/results/short")
    assert d["preview"] == {"values": [1, 2, 3], "count": 3}


def test_describe_nd_stats_and_storage(provider, sample):
    d = provider.describe(sample, "/entry_0000/results/image")
    preview = d["preview"]
    assert preview["min"] == 0.0
    assert preview["max"] == 64 * 64 - 1
    assert preview["mean"] == pytest.approx((64 * 64 - 1) / 2)
    assert preview["sampled"] is False
    assert d["render_hint"] == "heatmap"
    assert d["storage"]["layout"] == "chunked"
    assert d["storage"]["chunks"] == (16, 16)
    assert d["storage"]["filters"][0]["name"] == "deflate"
    assert d["storage"]["logical_bytes"] == 64 * 64 * 4


def test_describe_stats_skip_non_finite(provider, sample):
    preview = provider.describe(sample, "/with_nan")["preview"]
    assert preview["min"] == 1.0
    assert preview["max"] == 3.0


def test_describe_stats_are_sampled_on_large_dataset(sample):
    small_budget = HDF5Provider(stats_max_bytes=64 * 4 * 8)
    preview = small_budget.describe(sample, "/entry_0000/results/image")["preview"]
    assert preview["sampled"] is True
    assert preview["sampled_rows"] == 8


def test_describe_scalar_text_table(provider, sample):
    assert provider.describe(sample, "/scalar")["preview"] == 42

    text = provider.describe(sample, "/text")
    assert text["render_hint"] == "text"
    assert text["preview"].startswith("xxx")
    assert len(text["preview"]) == provider.text_preview_chars + 1

    table = provider.describe(sample, "/table")
    assert table["render_hint"] == "table"
    assert table["preview"]["rows"][0] == {"x": 1.0, "n": 2}
    assert table["preview"]["total_rows"] == 3


def test_describe_group_and_links(provider, sample):
    group = provider.describe(sample, "/entry_0000")
    assert group["node"]["kind"] == "container"
    assert group["preview"] == {"children": 1}
    assert group["attributes"]["NX_class"] == "NXentry"

    broken = provider.describe(sample, "/broken")
    assert broken["preview"] == {"target": "/nowhere", "resolved": False}

    root = provider.describe(sample, "/")
    assert root["node"]["name"] == "/"
    assert root["attributes"]["title"] == "sample"


def test_describe_empty_dataset(provider, sample):
    d = provider.describe(sample, "/empty")
    assert d["preview"] is None
    assert d["node"]["shape"] is None


def test_describe_preview_under_byte_cap(tmp_path):
    path = tmp_path / "wide_strings.h5"
    with h5py.File(path, "w") as f:
        f["s"] = np.array(["y" * 150] * 1000, dtype=h5py.string_dtype())
        f["t"] = np.zeros(1000, dtype=[(f"c{i}", "f8") for i in range(200)])
    tight = HDF5Provider(preview_max_bytes=512)
    for locator in ("/s", "/t"):
        preview = tight.describe(path, locator)["preview"]
        assert len(json.dumps(preview, default=str)) <= 512


def test_describe_is_json_serializable(provider, sample):
    for node in provider.children(sample, "/"):
        json.dumps(provider.describe(sample, node["locator"]), default=str)


# -- read ------------------------------------------------------------------


def test_read_selection(provider, sample):
    p = provider.read(sample, "/entry_0000/results/I", "10:15")
    assert p["data"].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0]
    assert p["shape"] == (5,)
    assert p["dtype"] == "float64"
    assert p["truncated"] is False

    p = provider.read(sample, "/entry_0000/results/image", "3, ::16")
    assert p["shape"] == (4,)
    assert p["data"].tolist() == [192.0, 208.0, 224.0, 240.0]


def test_read_whole_and_scalar(provider, sample):
    assert provider.read(sample, "/entry_0000/results/short", None)[
        "data"
    ].tolist() == [1, 2, 3]
    assert provider.read(sample, "/scalar", None)["data"] == 42
    assert provider.read(sample, "/text", None)["data"] == "x" * 1000
    assert provider.read(sample, "/labels", None)["data"].tolist() == ["alpha", "beta"]
    assert provider.read(sample, "/empty", None)["data"] is None


def test_read_honors_payload_cap(sample):
    capped = HDF5Provider(read_max_bytes=10 * 64 * 4)
    p = capped.read(sample, "/entry_0000/results/image", None)
    assert p["truncated"] is True
    assert p["shape"] == (10, 64)
    assert p["data"].nbytes <= capped.read_max_bytes

    p = capped.read(sample, "/entry_0000/results/image", "5:60:2, 1:")
    assert p["truncated"] is True
    assert p["shape"] == (10, 63)
    assert p["data"][0, 0] == 5 * 64 + 1


def test_read_cap_trims_inner_axis_when_one_row_is_too_big(sample):
    capped = HDF5Provider(read_max_bytes=32 * 4)
    p = capped.read(sample, "/entry_0000/results/image", None)
    assert p["truncated"] is True
    assert p["shape"] == (1, 32)


def test_read_element_over_cap_is_too_large(sample):
    with pytest.raises(ProviderError) as err:
        HDF5Provider(read_max_bytes=4).read(sample, "/table", None)
    assert err.value.code == "too_large"


def test_soft_link_reads_target(provider, sample):
    assert provider.read(sample, "/soft", "0:3")["data"].tolist() == [0.0, 1.0, 2.0]


def test_dots_are_not_separators(provider, sample):
    d = provider.describe(sample, "/entry_0000/results/q.values")
    assert d["node"]["name"] == "q.values"


# -- canonicalize ----------------------------------------------------------


@pytest.mark.parametrize(
    "locator, expected",
    [
        ("/", "/"),
        ("", "/"),
        ("entry", "/entry"),
        ("/entry/", "/entry"),
        ("//entry///results", "/entry/results"),
        ("/entry/./results", "/entry/results"),
        ("/results/q.values", "/results/q.values"),
    ],
)
def test_canonicalize(provider, locator, expected):
    assert provider.canonicalize(locator) == expected
    assert provider.canonicalize(expected) == expected


def test_equivalent_locators_describe_the_same_object(provider, sample):
    a = provider.describe(sample, "entry_0000//results/I/")
    b = provider.describe(sample, "/entry_0000/results/I")
    assert a["node"] == b["node"]


# -- errors ----------------------------------------------------------------


@pytest.mark.parametrize(
    "call, code",
    [
        (lambda p, f: p.describe(f, "/nope"), "not_found"),
        (lambda p, f: p.children(f, "/entry_0000/nope"), "not_found"),
        (lambda p, f: p.describe(f.parent / "missing.h5", "/"), "not_found"),
        (lambda p, f: p.read(f, "/entry_0000", None), "unsupported"),
        (lambda p, f: p.read(f, "/entry_0000/results/I", "a:b"), "unsupported"),
        (lambda p, f: p.read(f, "/entry_0000/results/I", "1, 2"), "unsupported"),
        (lambda p, f: p.read(f, "/entry_0000/results/I", "::-1"), "unsupported"),
        (lambda p, f: p.read(f, "/scalar", "0"), "unsupported"),
    ],
)
def test_error_codes(provider, sample, call, code):
    with pytest.raises(ProviderError) as err:
        call(provider, sample)
    assert err.value.code == code
    assert err.value.code in ERROR_CODES


def test_non_hdf5_file_is_unsupported(provider, tmp_path):
    fake = tmp_path / "fake.h5"
    fake.write_bytes(b"not hdf5")
    with pytest.raises(ProviderError) as err:
        provider.children(fake, "/")
    assert err.value.code == "unsupported"


def test_lock_failure_maps_to_locked(provider, sample, monkeypatch):
    def locked(*args, **kwargs):
        raise OSError(
            "Unable to synchronously open file (unable to lock file, errno = 11)"
        )

    monkeypatch.setattr(h5py, "File", locked)
    with pytest.raises(ProviderError) as err:
        provider.children(sample, "/")
    assert err.value.code == "locked"


# -- completion ------------------------------------------------------------


def test_complete_is_superset_of_matching_children(provider, sample):
    names = {n["name"] for n in provider.children(sample, "/entry_0000/results")}
    completed = provider.complete(sample, "/entry_0000/results/")
    assert {c.rsplit("/", 1)[-1] for c in completed} >= names

    assert provider.complete(sample, "/entry_0000/results/q") == [
        "/entry_0000/results/q.values"
    ]
    assert "/entry_0000" in provider.complete(sample, "ent")


def test_complete_unknown_parent_is_empty(provider, sample):
    assert provider.complete(sample, "/nope/x") == []


# -- statelessness ---------------------------------------------------------


def test_every_call_releases_the_file(provider, sample):
    provider.children(sample, "/")
    provider.describe(sample, "/entry_0000/results/image")
    provider.read(sample, "/entry_0000/results/I", "0:10")
    provider.complete(sample, "/e")
    _assert_released(sample)
