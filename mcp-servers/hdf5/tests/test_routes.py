"""Tests for the h5grove-compatible data route (hdf5_mcp.provider.routes)."""

import io
import json

import h5py
import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi import Depends, FastAPI, Header, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from h5grove import fastapi_utils  # noqa: E402

from hdf5_mcp.provider.routes import base_dir_resolver, create_router  # noqa: E402


@pytest.fixture
def data_dir(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    with h5py.File(root / "sample.h5", "w") as f:
        f.attrs["title"] = "sample"
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        intensity = entry.create_dataset("I", data=np.arange(100, dtype="f8"))
        intensity.attrs["units"] = "counts"
        entry.create_dataset(
            "image", data=np.arange(64 * 64, dtype="f4").reshape(64, 64)
        )
        f["labels"] = np.array(["alpha", "beta"], dtype=h5py.string_dtype())
        f["soft"] = h5py.SoftLink("/entry/I")
        f["broken"] = h5py.SoftLink("/nowhere")
    (root / "fake.h5").write_bytes(b"not hdf5")
    with h5py.File(tmp_path / "outside.h5", "w") as f:
        f["secret"] = 1
    return root


def _client(data_dir, **kwargs):
    app = FastAPI()
    app.include_router(
        create_router(base_dir_resolver(data_dir), **kwargs), prefix="/h5grove"
    )
    return TestClient(app)


@pytest.fixture
def client(data_dir):
    return _client(data_dir)


@pytest.fixture
def stock_client(data_dir, monkeypatch):
    monkeypatch.setattr(fastapi_utils.settings, "base_dir", str(data_dir))
    app = FastAPI()
    app.include_router(fastapi_utils.router, prefix="/h5grove")
    return TestClient(app)


# -- compatibility with h5grove's own router -----------------------------


@pytest.mark.parametrize(
    "query",
    [
        "/",
        "/meta?file=sample.h5&path=/",
        "/meta?file=sample.h5&path=/entry",
        "/meta?file=sample.h5&path=/entry/I",
        "/meta?file=sample.h5&path=/broken",
        "/meta?file=sample.h5&path=/soft&resolve_links=none",
        "/attr?file=sample.h5&path=/entry/I",
        "/attr?file=sample.h5&path=/&attr_keys=title",
        "/data?file=sample.h5&path=/entry/I",
        "/data?file=sample.h5&path=/entry/I&selection=10:20",
        "/data?file=sample.h5&path=/entry/image&selection=3,::16",
        "/data?file=sample.h5&path=/entry/image&selection=0:2&format=npy",
        "/data?file=sample.h5&path=/entry/image&format=bin&dtype=safe&flatten=true",
        "/data?file=sample.h5&path=/labels",
        "/stats?file=sample.h5&path=/entry/image",
        "/stats?file=sample.h5&path=/entry/image&selection=0:8",
        "/paths?file=sample.h5&path=/",
        "/paths?file=sample.h5&path=/entry",
    ],
)
def test_matches_stock_h5grove(client, stock_client, query):
    ours = client.get("/h5grove" + query)
    stock = stock_client.get("/h5grove" + query)
    assert ours.status_code == stock.status_code == 200
    assert ours.content == stock.content
    assert ours.headers.get("content-type") == stock.headers.get("content-type")


def test_npy_round_trip(client):
    r = client.get(
        "/h5grove/data",
        params={"file": "sample.h5", "path": "/entry/image", "selection": "1:3"},
        headers={},
    )
    assert r.json()[0][0] == 64.0
    r = client.get(
        "/h5grove/data",
        params={
            "file": "sample.h5",
            "path": "/entry/image",
            "selection": "1:3",
            "format": "npy",
        },
    )
    array = np.load(io.BytesIO(r.content))
    assert array.shape == (2, 64)
    assert array[0, 0] == 64.0


# -- file resolution -----------------------------------------------------


@pytest.mark.parametrize(
    "file", ["../outside.h5", "/etc/passwd", "sub/../../outside.h5"]
)
def test_cannot_escape_base_dir(client, file):
    r = client.get("/h5grove/meta", params={"file": file, "path": "/"})
    assert r.status_code == 403


def test_symlink_out_of_base_dir_is_refused(client, data_dir):
    (data_dir / "link.h5").symlink_to(data_dir.parent / "outside.h5")
    r = client.get("/h5grove/meta", params={"file": "link.h5", "path": "/"})
    assert r.status_code == 403


def test_stock_router_escapes_base_dir(stock_client):
    """Why the stock router can't be mounted in CLIO: `../` walks out."""
    r = stock_client.get(
        "/h5grove/data", params={"file": "../outside.h5", "path": "/secret"}
    )
    assert r.status_code == 200
    assert r.json() == 1


def test_missing_file_is_404(client):
    r = client.get("/h5grove/meta", params={"file": "nope.h5", "path": "/"})
    assert r.status_code == 404
    assert "message" in r.json()


def test_file_parameter_is_required(client):
    assert client.get("/h5grove/meta", params={"path": "/"}).status_code == 422


def test_async_resolver_and_auth_dependency(data_dir):
    async def resolve(file_id: str):
        if file_id != "abc123":
            raise FileNotFoundError(file_id)
        return data_dir / "sample.h5"

    def require_token(authorization: str = Header(default="")):
        if authorization != "Bearer good":
            raise HTTPException(status_code=401)

    app = FastAPI()
    app.include_router(
        create_router(resolve),
        prefix="/h5grove",
        dependencies=[Depends(require_token)],
    )
    client = TestClient(app)
    params = {"file": "abc123", "path": "/entry/I", "selection": "0:3"}

    assert client.get("/h5grove/data", params=params).status_code == 401
    r = client.get(
        "/h5grove/data", params=params, headers={"Authorization": "Bearer good"}
    )
    assert r.json() == [0.0, 1.0, 2.0]
    r = client.get(
        "/h5grove/data",
        params={**params, "file": "other"},
        headers={"Authorization": "Bearer good"},
    )
    assert r.status_code == 404


# -- limits --------------------------------------------------------------


def test_data_over_cap_is_413(data_dir):
    client = _client(data_dir, data_max_bytes=10 * 64 * 4)
    params = {"file": "sample.h5", "path": "/entry/image"}

    r = client.get("/h5grove/data", params=params)
    assert r.status_code == 413
    assert "smaller selection" in r.json()["message"]

    r = client.get("/h5grove/data", params={**params, "selection": "0:10"})
    assert r.status_code == 200
    assert len(r.json()) == 10

    r = client.get("/h5grove/stats", params=params)
    assert r.status_code == 413


def test_paths_over_cap_is_413(data_dir):
    client = _client(data_dir, max_paths=3)
    r = client.get("/h5grove/paths", params={"file": "sample.h5", "path": "/"})
    assert r.status_code == 413
    r = client.get("/h5grove/paths", params={"file": "sample.h5", "path": "/entry"})
    assert r.json() == ["/entry", "/entry/I", "/entry/image"]


# -- errors --------------------------------------------------------------


@pytest.mark.parametrize(
    "endpoint, params, status",
    [
        ("/data", {"path": "/entry"}, 422),
        ("/stats", {"path": "/entry"}, 422),
        ("/paths", {"path": "/entry/I"}, 422),
        ("/data", {"path": "/entry/I", "selection": "a:b"}, 422),
        ("/data", {"path": "/entry/I", "selection": "1,2"}, 422),
        ("/data", {"path": "/entry/I", "selection": "500"}, 422),
        ("/data", {"path": "/entry/I", "format": "pdf"}, 422),
        ("/data", {"path": "/labels", "dtype": "safe"}, 422),
        ("/meta", {"path": "/nope"}, 404),
        ("/meta", {"path": "/", "resolve_links": "sometimes"}, 422),
        ("/attr", {"path": "/", "attr_keys": "missing"}, 404),
    ],
)
def test_error_status(client, endpoint, params, status):
    r = client.get("/h5grove" + endpoint, params={"file": "sample.h5", **params})
    assert r.status_code == status, r.text
    assert isinstance(json.loads(r.content)["message"], str)


def test_non_hdf5_file_is_422(client):
    r = client.get("/h5grove/meta", params={"file": "fake.h5", "path": "/"})
    assert r.status_code == 422


def test_locked_file_is_423(client, monkeypatch):
    def locked(*args, **kwargs):
        raise OSError("Unable to synchronously open file (unable to lock file)")

    monkeypatch.setattr(h5py, "File", locked)
    r = client.get("/h5grove/meta", params={"file": "sample.h5", "path": "/"})
    assert r.status_code == 423


def test_file_is_released_after_each_request(client, data_dir):
    client.get("/h5grove/data", params={"file": "sample.h5", "path": "/entry/I"})
    client.get("/h5grove/paths", params={"file": "sample.h5", "path": "/"})
    with h5py.File(data_dir / "sample.h5", "a"):
        pass
