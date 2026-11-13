"""h5grove data route for the HDF5 plugin (the RFC's `data_routes`).

Serves h5grove's HTTP API (`/meta`, `/attr`, `/data`, `/stats`, `/paths`), so
h5web's `H5GroveProvider` works against it unchanged. It differs from
`h5grove.fastapi_utils.router` where that router is unsafe to mount inside
CLIO:

- Files are named by an opaque id that CLIO resolves to a path. The stock
  router joins a `file` query parameter onto `base_dir` as a string, which
  lets `../` escape it, and serves any path when `base_dir` is unset.
- `/data` and `/stats` refuse selections over `data_max_bytes` with 413
  instead of reading the whole dataset into memory.
- `/paths` stops at `max_paths` objects.
- Asking for data from a group, a malformed selection or a non-HDF5 file is
  a 4xx, not an unhandled 500.

Authentication is the mounting application's job; add it as a router
dependency when including the router.

Requires the `data-routes` extra (FastAPI).
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import h5py
from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.routing import APIRoute
from h5grove.content import (
    DatasetContent,
    GroupContent,
    ResolvedEntityContent,
    create_content,
)
from h5grove.encoders import encode
from h5grove.models import LinkResolution
from h5grove.utils import (
    NotFoundError,
    QueryArgumentError,
    hdf_path_join,
    parse_link_resolution_arg,
)

from .hdf5 import open_error_code
from .selection import Index, parse_selection, selection_bytes

FileResolver = Callable[[str, Request], Path | Awaitable[Path]]
"""Maps the `file` query parameter to a path on disk.

Called with the parameter and the request, so a host can use path parameters
(a workspace id in the mount prefix) or the authenticated caller. Raises
FileNotFoundError for an unknown id and PermissionError for one the caller
may not read. May be async.
"""

_STATUS = {"not_found": 404, "unsupported": 422, "locked": 423, "io": 500}


class DataRouteError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class _ErrorRoute(APIRoute):
    """Turns DataRouteError into h5grove's `{"message": ...}` error body."""

    def get_route_handler(self) -> Callable:
        handler = super().get_route_handler()

        async def route_handler(request: Request) -> Response:
            try:
                return await handler(request)
            except DataRouteError as e:
                body = encode({"message": e.message}, "json")
                return Response(
                    content=body.content,
                    headers=body.headers,
                    status_code=e.status_code,
                )

        return route_handler


def base_dir_resolver(base_dir: str | Path) -> FileResolver:
    """A resolver serving files under `base_dir` only, by relative path.

    For standalone use and tests; inside CLIO the file API supplies the
    resolver.
    """
    base = Path(base_dir).resolve()

    def resolve(file: str, request: Request) -> Path:
        candidate = (base / file).resolve()
        if not candidate.is_relative_to(base):
            raise PermissionError(file)
        if not candidate.is_file():
            raise FileNotFoundError(file)
        return candidate

    return resolve


def create_router(
    resolve_file: FileResolver,
    *,
    data_max_bytes: int = 256 * 1024 * 1024,
    max_paths: int = 100_000,
) -> APIRouter:
    """Build the h5grove-compatible router.

    Args:
        resolve_file: Maps the `file` query parameter to a path; see
            `FileResolver`.
        data_max_bytes: Largest selection `/data` and `/stats` will read.
        max_paths: Most object paths `/paths` will return.
    """
    router = APIRouter(route_class=_ErrorRoute)

    async def file_path(request: Request, file: str = Query(...)) -> Path:
        try:
            resolved = resolve_file(file, request)
            if inspect.isawaitable(resolved):
                resolved = await resolved
        except FileNotFoundError as e:
            raise DataRouteError(404, "File not found") from e
        except PermissionError as e:
            raise DataRouteError(403, "Access to this file is not allowed") from e
        return Path(resolved)

    def checked_index(dataset: h5py.Dataset, selection: str | None) -> Index:
        if dataset.shape is None:
            return ()
        try:
            index = parse_selection(selection, dataset.ndim)
        except ValueError as e:
            raise DataRouteError(422, str(e)) from e
        size = selection_bytes(dataset.shape, dataset.dtype, index)
        if size > data_max_bytes:
            raise DataRouteError(
                413,
                f"Selection would read {size} bytes, over the "
                f"{data_max_bytes}-byte limit; request a smaller selection",
            )
        return index

    @router.api_route("/", methods=["GET", "HEAD"])
    def get_root() -> Response:
        return Response("ok")

    @router.get("/meta")
    def get_meta(
        file: Path = Depends(file_path),
        path: str = "/",
        resolve_links: str = "only_valid",
    ) -> Response:
        with _open(file) as f:
            content = _content(f, path, resolve_links)
            return _encode(content.metadata(), "json")

    @router.get("/attr")
    def get_attr(
        file: Path = Depends(file_path),
        path: str = "/",
        attr_keys: list[str] | None = Query(default=None),
    ) -> Response:
        with _open(file) as f:
            content = _content(f, path)
            if not isinstance(content, ResolvedEntityContent):
                raise DataRouteError(422, f"{content.path} is not a resolved entity")
            try:
                attributes = content.attributes(attr_keys)
            except KeyError as e:
                raise DataRouteError(404, f"Attribute not found: {e}") from e
            return _encode(attributes, "json")

    @router.get("/data")
    def get_data(
        file: Path = Depends(file_path),
        path: str = "/",
        dtype: str = "origin",
        format: str = "json",
        flatten: bool = False,
        selection: str | None = None,
    ) -> Response:
        with _open(file) as f:
            content = _dataset_content(f, path)
            index = checked_index(f[path], selection)
            try:
                data = content.data(index or None, flatten, dtype)
            except (ValueError, TypeError, IndexError) as e:
                raise DataRouteError(422, str(e)) from e
            return _encode(data, format)

    @router.get("/stats")
    def get_stats(
        file: Path = Depends(file_path),
        path: str = "/",
        selection: str | None = None,
    ) -> Response:
        with _open(file) as f:
            content = _dataset_content(f, path)
            index = checked_index(f[path], selection)
            try:
                stats = content.data_stats(index or None)
            except (ValueError, TypeError, IndexError) as e:
                raise DataRouteError(422, str(e)) from e
            return _encode(stats, "json")

    @router.get("/paths")
    def get_paths(
        file: Path = Depends(file_path),
        path: str = "/",
        resolve_links: str = "only_valid",
    ) -> Response:
        with _open(file) as f:
            content = _content(f, path, resolve_links)
            if not isinstance(content, GroupContent):
                raise DataRouteError(422, f"{content.path} is not a group")
            paths = [content.path]

            def collect(name: bytes) -> bool | None:
                paths.append(hdf_path_join(content.path, name.decode()))
                return True if len(paths) > max_paths else None

            group = f[path]  # keep a reference: a temporary's id closes early
            group.id.links.visit(collect)
            if len(paths) > max_paths:
                raise DataRouteError(
                    413, f"{content.path} holds more than {max_paths} objects"
                )
            return _encode(paths, "json")

    return router


RESOLVER_STATE_KEY = "clio_file_resolver"
"""The `app.state` attribute `router` reads its `FileResolver` from."""


def app_state_resolver(file: str, request: Request) -> Path | Awaitable[Path]:
    """Delegate to the resolver the host application put on `app.state`.

    Until the host sets one, every request fails with 503 rather than
    falling back to some directory.
    """
    resolve = getattr(request.app.state, RESOLVER_STATE_KEY, None)
    if resolve is None:
        raise DataRouteError(503, "No file resolver is configured for this route")
    return resolve(file, request)


router = create_router(app_state_resolver)
"""The plugin's `data_routes` entry (`hdf5_mcp.provider.routes:router`).

The RFC's manifest names a router object, but only the host knows how a file
id maps to a path and who may read it, so the host supplies that at startup::

    app.state.clio_file_resolver = resolve  # a FileResolver
    app.include_router(router, prefix=...)

Hosts that can call a factory should prefer `create_router(resolve)`.
"""


@contextmanager
def _open(file: Path) -> Iterator[h5py.File]:
    try:
        f = h5py.File(file, "r")
    except FileNotFoundError as e:
        raise DataRouteError(404, "File not found") from e
    except PermissionError as e:
        raise DataRouteError(403, "Cannot read file: permission denied") from e
    except OSError as e:
        raise DataRouteError(_STATUS[open_error_code(e)], str(e)) from e
    try:
        with f:
            yield f
    except OSError as e:
        raise DataRouteError(500, str(e)) from e


def _content(f: h5py.File, path: str, resolve_links: str = "only_valid") -> Any:
    try:
        links = parse_link_resolution_arg(
            resolve_links, fallback=LinkResolution.ONLY_VALID
        )
        return create_content(f, path, links)
    except QueryArgumentError as e:
        raise DataRouteError(422, str(e)) from e
    except NotFoundError as e:
        raise DataRouteError(404, str(e)) from e


def _dataset_content(f: h5py.File, path: str) -> DatasetContent:
    content = _content(f, path)
    if not isinstance(content, DatasetContent):
        raise DataRouteError(422, f"{content.path} is not a dataset")
    return content


def _encode(value: Any, encoding: str) -> Response:
    try:
        encoded = encode(value, encoding)
    except QueryArgumentError as e:
        raise DataRouteError(422, str(e)) from e
    return Response(content=encoded.content, headers=encoded.headers)
