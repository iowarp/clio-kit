"""Shared plumbing: errors, HTTP, output paths, regions and service-kind detection."""

from __future__ import annotations

import json
import os
import re
import ssl
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit, urlunsplit

import httpx

USER_AGENT = "clio-kit-geoservices/1.0 (+https://github.com/iowarp/clio-kit)"
CONNECT_TIMEOUT_S = 10.0
READ_TIMEOUT_S = 60.0
MAX_JSON_BYTES = 64 * 1024 * 1024


class GeoServiceError(ValueError):
    """A request could not be completed; the message says why and what to try."""


class ResponseTooLarge(GeoServiceError):
    """One response exceeded the size bound; a smaller page may fit."""


@dataclass(frozen=True)
class Region:
    """A WGS84 longitude/latitude box."""

    west: float
    south: float
    east: float
    north: float

    @classmethod
    def parse(cls, bbox: list[float] | tuple[float, ...] | str | None) -> Region | None:
        """Accept [west, south, east, north] as a list or comma-separated string."""
        if bbox is None or bbox == "" or bbox == []:
            return None
        values = (
            [float(v) for v in re.split(r"[,\s]+", bbox.strip()) if v]
            if isinstance(bbox, str)
            else [float(v) for v in bbox]
        )
        if len(values) != 4:
            raise GeoServiceError(
                "bbox needs four numbers: west, south, east, north (degrees)"
            )
        region = cls(*values)
        if not (
            -180 <= region.west < region.east <= 180
            and -90 <= region.south < region.north <= 90
        ):
            raise GeoServiceError(
                "bbox must be west < east and south < north in WGS84 degrees "
                f"(got {values}); a box crossing the antimeridian must be split"
            )
        return region

    def as_list(self) -> list[float]:
        return [self.west, self.south, self.east, self.north]


@lru_cache(maxsize=1)
def _system_tls() -> ssl.SSLContext:
    """The OS trust store, so networks that inspect TLS with an OS-trusted CA work."""
    import truststore

    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def client() -> httpx.Client:
    """An HTTP client with bounded timeouts, verifying TLS against the OS trust store."""
    return httpx.Client(
        timeout=httpx.Timeout(READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
        verify=_system_tls(),
    )


def raise_for_status(response: httpx.Response) -> None:
    """Like ``raise_for_status``, reading a streamed error body first so its text is reported."""
    if response.is_error:
        response.read()
    response.raise_for_status()


def http_error(exc: Exception, what: str) -> GeoServiceError:
    """Turn a transport or HTTP failure into a typed, human-readable reason."""
    if isinstance(exc, httpx.HTTPStatusError):
        body = exc.response.text[:300].strip().replace("\n", " ")
        return GeoServiceError(
            f"{what} failed with HTTP {exc.response.status_code}"
            + (f": {body}" if body else "")
        )
    if isinstance(exc, httpx.TimeoutException):
        return GeoServiceError(
            f"{what} timed out; try a smaller region or fewer records"
        )
    return GeoServiceError(f"{what} failed: {type(exc).__name__}: {exc}")


def get_json(url: str, params: dict[str, Any] | None = None, *, what: str) -> Any:
    """GET a JSON document with a size bound; ArcGIS error envelopes become errors."""
    try:
        with client() as http, http.stream("GET", url, params=params) as response:
            raise_for_status(response)
            body = bytearray()
            for chunk in response.iter_bytes(65536):
                body.extend(chunk)
                if len(body) > MAX_JSON_BYTES:
                    raise ResponseTooLarge(
                        f"{what} returned more than 64 MiB; narrow the request"
                    )
    except httpx.HTTPError as exc:
        raise http_error(exc, what) from exc
    try:
        data = json.loads(body)
    except ValueError as exc:
        raise GeoServiceError(
            f"{what} did not return JSON (starts with {bytes(body[:80])!r})"
        ) from exc
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        error = data["error"]
        details = "; ".join(str(d) for d in error.get("details") or [])
        raise GeoServiceError(
            f"{what} was refused by the service: {error.get('code', '')} "
            f"{error.get('message', '')} {details}".strip()
        )
    return data


def get_text(url: str, params: dict[str, Any] | None = None, *, what: str) -> str:
    """GET a text document (XML capabilities) with the same size bound."""
    try:
        with client() as http, http.stream("GET", url, params=params) as response:
            raise_for_status(response)
            body = bytearray()
            for chunk in response.iter_bytes(65536):
                body.extend(chunk)
                if len(body) > MAX_JSON_BYTES:
                    raise ResponseTooLarge(f"{what} returned more than 64 MiB")
    except httpx.HTTPError as exc:
        raise http_error(exc, what) from exc
    return body.decode("utf-8", errors="replace")


def artifacts_root() -> Path:
    """Writable output root: ``CLIO_KIT_ARTIFACTS`` or a per-user temp directory."""
    configured = os.environ.get("CLIO_KIT_ARTIFACTS", "").strip()
    root = (
        Path(configured).expanduser()
        if configured
        else Path(tempfile.gettempdir()) / "clio-kit-geoservices"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def output_path(candidate: str | None, default_name: str, suffix: str) -> Path:
    """Confine writes to the artifacts root, keeping only the requested file name."""
    root = artifacts_root()
    name = Path(candidate).name if candidate else default_name
    name = re.sub(r"[^\w.\-]+", "_", name).strip("._") or default_name
    if not name.lower().endswith(suffix):
        name += suffix
    path = (root / name).resolve()
    if root not in path.parents:
        raise GeoServiceError(
            "output_path must be a file name inside the artifacts directory"
        )
    return path


def slug(text: str, limit: int = 48) -> str:
    return re.sub(r"[^\w]+", "_", text).strip("_")[:limit] or "result"


# --- service-kind detection ---------------------------------------------------------

ARCGIS_RE = re.compile(
    r"/(?:arcgis/)?rest/services/.+?/(FeatureServer|MapServer|ImageServer)(?:/(\d+))?/?$",
    re.IGNORECASE,
)


def unwrap_viewer(url: str) -> str:
    """Entwine/Potree viewer links carry the point-cloud root in their ``r`` parameter."""
    parts = urlsplit(url)
    if parts.path.endswith("view.html") or "entwine.io" in parts.netloc:
        for value in parse_qs(parts.query).get("r", []):
            candidate = unquote(value).strip('"')
            if candidate.startswith(("http://", "https://")):
                return candidate.rstrip("/") + "/ept.json"
    return url


def classify(url: str) -> str:
    """Name the protocol a URL speaks, from its shape alone (no request)."""
    url = unwrap_viewer(url)
    parts = urlsplit(url)
    path = unquote(parts.path).lower()
    query = {k.lower(): v for k, v in parse_qs(parts.query).items()}
    service = (query.get("service") or [""])[0].upper()
    if ARCGIS_RE.search(parts.path.rstrip("/").split("/query")[0]):
        return "arcgis"
    if service in {"WFS", "WCS", "WMS"}:
        return service.lower()
    if re.search(r"/(wfs|ows)/?$", path) and "wcs" not in path:
        return "wfs" if path.endswith("wfs") else "ows"
    if path.endswith("/wcs"):
        return "wcs"
    if path.endswith("ept.json"):
        return "ept"
    if path.endswith(".copc.laz"):
        return "copc"
    if path.endswith((".tif", ".tiff", ".vrt", ".img", ".jp2", ".asc", ".adf")):
        return "raster"
    if "stac" in path or path.endswith(
        ("/search", "/collections", "catalog.json", "collection.json")
    ):
        return "stac"
    if path.endswith((".geojson", ".json")) or "f=geojson" in parts.query.lower():
        return "geojson"
    return "unknown"


def strip_query(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
