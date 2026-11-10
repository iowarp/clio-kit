import asyncio
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.prompts import Message
from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict

# Environment setup
load_dotenv()

# Initialize FastMCP server instance
mcp: FastMCP = FastMCP(
    "ndp",
    instructions=(
        "Discovers and explores scientific datasets from the National Data Platform "
        "(NDP) and stages their resources for analysis. Use search_datasets to find "
        "data, get_dataset_details for metadata, list_organizations for sources, and "
        "stage_resource to download an HTTP(S) or OSDF/Pelican dataset resource to a "
        "local file."
    ),
    list_page_size=10,
)

# ---------------------------------------------------------------------------
# Resource staging configuration and helpers (standalone, domain-neutral)
# ---------------------------------------------------------------------------

# Default ceiling for direct resource staging to avoid runaway downloads.
_MAX_STAGE_BYTES = 50 * 1024 * 1024
# Network timeouts (seconds) for staging HTTP resources.
_HTTP_CONNECT_TIMEOUT_S = 8.0
_HTTP_READ_TIMEOUT_S = 60.0
# Wall-clock timeout (seconds) for OSDF/Pelican CLI staging.
_PELICAN_TIMEOUT_S = 900

_SIZE_UNITS = {
    "b": 1,
    "byte": 1,
    "bytes": 1,
    "kb": 1024,
    "kib": 1024,
    "mb": 1024 * 1024,
    "mib": 1024 * 1024,
    "gb": 1024 * 1024 * 1024,
    "gib": 1024 * 1024 * 1024,
    "tb": 1024 * 1024 * 1024 * 1024,
    "tib": 1024 * 1024 * 1024 * 1024,
}


def artifacts_root(output_dir: str | Path | None = None) -> Path:
    """Return the writable root for staged resources and generated artifacts.

    Precedence: an explicit ``output_dir`` (caller-supplied destination) wins;
    otherwise the ``CLIO_KIT_ARTIFACTS`` environment variable; otherwise the
    current working directory. Falling back to the process CWD lets a caller
    that launches the server with a chosen working directory control where
    artifacts land by default. The tool writes where it is told — no destination
    is hardcoded and nothing is rerouted.
    """
    explicit = str(output_dir).strip() if output_dir not in (None, "") else ""
    configured = os.environ.get("CLIO_KIT_ARTIFACTS", "").strip()
    if explicit:
        root = Path(explicit).expanduser()
    elif configured:
        root = Path(configured).expanduser()
    else:
        root = Path.cwd()
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _validate_output_path(
    candidate: str | Path, *, default_name: str, output_dir: str | Path | None = None
) -> Path:
    """Resolve an output path, confining writes to the resolved artifacts root.

    The root is ``output_dir`` if supplied (the caller's chosen destination),
    else the configurable artifacts root. A relative path, bare filename, or any
    path outside that root is relocated under it using only its filename — a
    self-contained allowed-root check with no external dependency.
    """
    root = artifacts_root(output_dir)
    raw = Path(str(candidate)).expanduser() if candidate else Path(default_name)
    name = raw.name or default_name
    if raw.is_absolute():
        try:
            resolved = raw.resolve()
            resolved.relative_to(root)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            return resolved
        except (ValueError, OSError):
            pass
    target = (root / name).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _safe_filename(value: str, *, default: str) -> str:
    """Return a conservative filesystem name for staged resources."""
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned[:120] or default


def _clean_max_bytes(value: int | str | None) -> int:
    """Normalize an optional byte limit for resource staging."""
    if value is None or value == "":
        return _MAX_STAGE_BYTES
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return _MAX_STAGE_BYTES
    return max(1, parsed)


def _parse_resource_size_bytes(value: Any) -> int | None:
    """Parse common resource size strings such as ``1.4 GB``."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value if value >= 0 else None
    if not isinstance(value, str) or not value.strip():
        return None
    parts = value.strip().replace(",", "").split()
    if not parts:
        return None
    try:
        number = float(parts[0])
    except ValueError:
        return None
    unit = parts[1].lower() if len(parts) > 1 else "bytes"
    multiplier = _SIZE_UNITS.get(unit)
    if multiplier is None:
        return None
    return int(number * multiplier)


def _stage_pelican_resource(
    *,
    url: str,
    output_path: Path,
    max_bytes: int,
    resource_size_bytes: int | None,
) -> dict[str, Any]:
    """Stage an OSDF/Pelican resource using the local ``pelican`` CLI when present."""
    if resource_size_bytes is not None and resource_size_bytes > max_bytes:
        raise ToolError(
            f"Resource is advertised as {resource_size_bytes} bytes, exceeding the "
            f"staging limit of {max_bytes} bytes. Increase max_bytes intentionally "
            "or select a smaller object."
        )
    pelican = shutil.which("pelican")
    if pelican is None:
        raise ToolError(
            "The selected resource uses OSDF/Pelican transport, but the `pelican` "
            "CLI was not found on PATH. Install the Pelican client and retry."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [pelican, "object", "get", url, str(output_path)]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=_PELICAN_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolError(
            f"Pelican staging timed out after {_PELICAN_TIMEOUT_S}s before the "
            "resource was downloaded."
        ) from exc

    if completed.returncode != 0:
        stderr = (completed.stderr or completed.stdout or "").strip()
        raise ToolError(
            f"Pelican failed to stage the resource (exit {completed.returncode}): {stderr[-1200:]}"
        )
    if not output_path.exists():
        raise ToolError("Pelican exited successfully but the expected staged file was not found.")

    size = output_path.stat().st_size if output_path.is_file() else 0
    if size > max_bytes:
        raise ToolError(
            f"Staged file is {size} bytes, exceeding the staging limit of {max_bytes} bytes."
        )
    return {
        "ok": True,
        "local_path": str(output_path),
        "size_bytes": size,
        "content_type": None,
        "url": url,
        "method": "pelican",
        "transport": "osdf",
    }


class Dataset(BaseModel):
    """Model for dataset information from NDP API."""

    id: str
    name: str
    title: str
    owner_org: str | None = None
    notes: str | None = None
    resources: list[dict[str, Any]] = Field(default_factory=list)
    extras: dict[str, Any] | None = None


# --- Structured result shapes (drive real MCP outputSchema declarations) ----


class ToolMeta(TypedDict):
    """Per-call diagnostic metadata echoed back on every NDP tool response."""

    tool: str
    status: str


class ListOrganizationsResult(TypedDict):
    """Structured result for a successful organizations listing."""

    organizations: list[str]
    count: int
    server: str
    name_filter: str | None
    _meta: ToolMeta


class SearchParameters(TypedDict):
    """Echo of the request parameters that produced a search result set."""

    search_terms: list[str] | None
    search_keys: list[str] | None
    dataset_name: str | None
    dataset_title: str | None
    owner_org: str | None
    resource_format: str | None
    search_term: str | None
    filter_list: list[str] | None
    limit: int | str | None


class SearchDatasetsResult(TypedDict):
    """Structured result for a dataset search."""

    datasets: list[Dataset]
    count: int
    total_found: int | str
    server: str
    search_parameters: SearchParameters
    _meta: ToolMeta


class DatasetIdentifier(TypedDict):
    """Which identifier field and value resolved the returned dataset."""

    type: str
    value: str


class GetDatasetDetailsResult(TypedDict):
    """Structured result for a single dataset's detailed metadata."""

    dataset: Dataset
    identifier_used: DatasetIdentifier
    server: str
    resource_count: int
    _meta: ToolMeta


class StageResourceResult(TypedDict):
    """Structured result for a staged dataset resource (HTTP or OSDF/Pelican).

    Both staging paths populate ``ok``, ``local_path``, ``size_bytes``,
    ``content_type``, ``url``, and ``method``. ``transport`` is present only
    on the OSDF/Pelican path; ``_meta`` is present only on the direct-HTTP(S)
    path.
    """

    ok: Literal[True]
    local_path: str
    size_bytes: int
    content_type: str | None
    url: str
    method: Literal["http", "pelican"]
    transport: NotRequired[Literal["osdf"]]
    _meta: NotRequired[ToolMeta]


# Public NDP CKAN catalog; this is the catalog an NDP Endpoint API calls "global".
DEFAULT_NDP_CATALOG_URL = "https://nationaldataplatform.org/catalog"
NDP_API_URL_ENV = "NDP_API_URL"
NDP_CATALOG_URL_ENV = "NDP_CATALOG_URL"


class _NDPHttpClient:
    """Shared HTTP plumbing (retries, timeouts, actionable errors) for NDP backends."""

    label = "NDP backend"
    env_var = NDP_API_URL_ENV

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(30.0, connect=10.0)
        self.max_retries = 3
        self.retry_delay = 1.0

    def _where(self) -> str:
        return f"{self.label} at {self.base_url} (configured via {self.env_var})"

    async def _make_request(  # type: ignore[return]
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make HTTP request with retry logic."""
        url = f"{self.base_url}{endpoint}"

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    if method.upper() == "GET":
                        response = await client.get(url, params=params)
                    elif method.upper() == "POST":
                        response = await client.post(url, params=params, json=json_data)
                    else:
                        raise ValueError(f"Unsupported HTTP method: {method}")

                    response.raise_for_status()
                    return response.json()  # type: ignore[no-any-return]

            except httpx.TimeoutException:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2**attempt))
                    continue
                raise Exception(
                    f"Request timed out after {self.max_retries} attempts contacting "
                    f"{self._where()}. Check that the URL is reachable or set "
                    f"{self.env_var} to a working NDP server."
                ) from None
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2**attempt))
                    continue
                raise Exception(
                    f"HTTP {e.response.status_code} from {self._where()}: {e.response.text}"
                ) from e
            except httpx.TransportError as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2**attempt))
                    continue
                raise Exception(
                    f"Could not connect to {self._where()}: {e}. Check that the URL is "
                    f"reachable or set {self.env_var} to a working NDP server."
                ) from e
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2**attempt))
                    continue
                raise Exception(f"Request failed: {str(e)}") from e


class NDPClient(_NDPHttpClient):
    """Client for an NDP Endpoint (EP) API (``/search``, ``/organization``)."""

    label = "NDP Endpoint API"
    env_var = NDP_API_URL_ENV

    async def list_organizations(
        self, name_filter: str | None = None, server: str = "global"
    ) -> list[str]:
        """List organizations from NDP API."""
        params = {"server": server}
        if name_filter:
            params["name"] = name_filter

        result = await self._make_request("GET", "/organization", params=params)
        return result if isinstance(result, list) else []

    async def search_datasets_simple(
        self, terms: list[str], keys: list[str] | None = None, server: str = "global"
    ) -> list[Dataset]:
        """Search datasets using simple term-based search."""
        params = {"server": server}

        # Add terms as query parameters
        for term in terms:
            params.setdefault("terms", []).append(term)  # type: ignore[attr-defined, arg-type]

        # Add keys if provided
        if keys:
            for key in keys:
                params.setdefault("keys", []).append(key)  # type: ignore[attr-defined, arg-type]

        result = await self._make_request("GET", "/search", params=params)

        if isinstance(result, list):
            return [Dataset(**item) for item in result]
        return []

    async def search_datasets_advanced(
        self,
        dataset_name: str | None = None,
        dataset_title: str | None = None,
        owner_org: str | None = None,
        resource_url: str | None = None,
        resource_name: str | None = None,
        dataset_description: str | None = None,
        resource_description: str | None = None,
        resource_format: str | None = None,
        search_term: str | None = None,
        filter_list: list[str] | None = None,
        timestamp: str | None = None,
        server: str = "global",
    ) -> list[Dataset]:
        """Search datasets using advanced search with specific field filtering."""
        search_data = {"server": server}

        # Add all non-None parameters to the search
        if dataset_name:
            search_data["dataset_name"] = dataset_name
        if dataset_title:
            search_data["dataset_title"] = dataset_title
        if owner_org:
            search_data["owner_org"] = owner_org
        if resource_url:
            search_data["resource_url"] = resource_url
        if resource_name:
            search_data["resource_name"] = resource_name
        if dataset_description:
            search_data["dataset_description"] = dataset_description
        if resource_description:
            search_data["resource_description"] = resource_description
        if resource_format:
            search_data["resource_format"] = resource_format
        if search_term:
            search_data["search_term"] = search_term
        if filter_list:
            search_data["filter_list"] = filter_list  # type: ignore[assignment]
        if timestamp:
            search_data["timestamp"] = timestamp

        result = await self._make_request("POST", "/search", json_data=search_data)

        if isinstance(result, list):
            return [Dataset(**item) for item in result]
        return []


_SOLR_SPECIAL_CHARS = re.compile(r'([+\-\!\(\)\{\}\[\]\^"~\*\?:\\])')
_RESOURCE_FIELDS = ("id", "url", "name", "description", "format", "size")


def _solr_phrase(value: str) -> str:
    """Quote a term as a Solr phrase so CKAN does not split it at letter/digit boundaries."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _solr_escape(value: str) -> str:
    """Escape Solr query syntax characters."""
    return _SOLR_SPECIAL_CHARS.sub(r"\\\1", value)


def _timestamp_query(timestamp: str) -> tuple[str, int | None, str]:
    """Translate an EP-style timestamp filter (``t``, ``>t``, ``<t``, ``a/b``) to Solr."""
    parts = timestamp.split("/")
    if len(parts) > 2:
        raise ValueError("timestamp has too many range elements.")
    if len(parts) == 1:
        if timestamp.startswith("<"):
            return f"timestamp:[* TO {timestamp[1:]}]", 1, "timestamp desc"
        nearest = timestamp[1:] if timestamp.startswith(">") else timestamp
        return f"timestamp:[{nearest} TO *]", 1, "timestamp asc"
    return f"timestamp:[{parts[0] or '*'} TO {parts[1] or '*'}]", None, "timestamp asc"


def _ckan_to_dataset(row: dict[str, Any], resources: list[dict[str, Any]] | None = None) -> Dataset:
    """Map a CKAN package to the same shape an NDP Endpoint API returns."""
    organization = row.get("organization")
    extras: dict[str, Any] = {
        extra["key"]: extra.get("value")
        for extra in row.get("extras") or []
        if isinstance(extra, dict) and "key" in extra
    }
    for key in ("mapping", "processing"):
        if isinstance(extras.get(key), str):
            try:
                extras[key] = json.loads(extras[key])
            except json.JSONDecodeError:
                pass
    source = (row.get("resources") or []) if resources is None else resources
    return Dataset(
        id=row["id"],
        name=row["name"],
        title=row.get("title") or row["name"],
        owner_org=organization.get("name") if isinstance(organization, dict) else None,
        notes=row.get("notes"),
        resources=[
            {field: res[field] for field in _RESOURCE_FIELDS if res.get(field) is not None}
            for res in source
            if isinstance(res, dict)
        ],
        extras=extras,
    )


class NDPCatalogClient(_NDPHttpClient):
    """Client for the public NDP CKAN catalog (the EP API's ``global`` catalog).

    Mirrors the NDP Endpoint API's ``/search`` and ``/organization`` semantics
    for ``server="global"`` so the server works without a self-hosted endpoint.
    """

    label = "NDP catalog"
    env_var = NDP_CATALOG_URL_ENV
    page_size = 100
    max_datasets = 1000

    def _require_global(self, server: str) -> None:
        if server != "global":
            raise ValueError(
                f"The '{server}' catalog is only available through an NDP Endpoint API. "
                f"Set {NDP_API_URL_ENV} to your endpoint's base URL; the {self._where()} "
                "serves only the 'global' catalog."
            )

    async def _action(self, action: str, params: dict[str, Any]) -> Any:
        payload = await self._make_request("GET", f"/api/3/action/{action}", params=params)
        if not isinstance(payload, dict) or not payload.get("success"):
            error = payload.get("error") if isinstance(payload, dict) else payload
            raise Exception(f"{self._where()} rejected {action}: {error}")
        return payload.get("result")

    async def _collect(
        self,
        query: str,
        convert: Callable[[dict[str, Any]], Dataset | None],
        *,
        fq: str | None = None,
        sort: str | None = None,
        limit: int | None = None,
    ) -> tuple[list[Dataset], int]:
        """Page through ``package_search`` until ``limit`` datasets survive ``convert``.

        Returns the datasets and the catalog's total hit count. Pages are sized to
        what is still needed because NDP packages can be several MB each.
        """
        want = min(limit or self.max_datasets, self.max_datasets)
        found: list[Dataset] = []
        scanned = 0
        total = 0
        while len(found) < want and scanned < self.max_datasets:
            # Size the first page to the request; later pages are only needed when
            # filters dropped rows, so read them in full pages.
            rows = min(self.page_size, want if scanned == 0 else self.page_size)
            rows = min(rows, self.max_datasets - scanned)
            params: dict[str, Any] = {"q": query, "rows": rows, "start": scanned}
            if fq:
                params["fq"] = fq
            if sort:
                params["sort"] = sort
            page = await self._action("package_search", params)
            results = page.get("results", []) if isinstance(page, dict) else []
            total = int(page.get("count") or 0) if isinstance(page, dict) else 0
            scanned += len(results)
            for row in results:
                dataset = convert(row) if isinstance(row, dict) else None
                if dataset is not None:
                    found.append(dataset)
            if len(results) < rows or scanned >= total:
                break
        return found[:want], max(total, len(found))

    async def list_organizations(
        self, name_filter: str | None = None, server: str = "global"
    ) -> list[str]:
        """List organization names in the public catalog."""
        self._require_global(server)
        result = await self._action("organization_list", {"all_fields": "false", "limit": 1000})
        organizations = [org for org in result or [] if isinstance(org, str)]
        if name_filter:
            needle = name_filter.lower()
            organizations = [org for org in organizations if needle in org.lower()]
        return organizations

    async def search_datasets_simple(
        self, terms: list[str], keys: list[str] | None = None, server: str = "global"
    ) -> list[Dataset]:
        """Search by terms, optionally scoped per term to a field key."""
        datasets, _ = await self.search_datasets_simple_page(terms, keys, server)
        return datasets

    async def search_datasets_simple_page(
        self,
        terms: list[str],
        keys: list[str] | None = None,
        server: str = "global",
        limit: int | None = None,
    ) -> tuple[list[Dataset], int]:
        """Like ``search_datasets_simple`` but bounded, also returning the total hit count."""
        self._require_global(server)
        parts = []
        for index, term in enumerate(terms):
            key = keys[index] if keys and index < len(keys) else None
            phrase = _solr_phrase(term)
            if key and key.lower() != "null":
                parts.append(f"{_solr_escape(key)}:{phrase}")
            else:
                parts.append(phrase)
        lowered = [term.lower() for term in terms]

        def convert(row: dict[str, Any]) -> Dataset | None:
            text = json.dumps(row).lower()
            return _ckan_to_dataset(row) if all(term in text for term in lowered) else None

        return await self._collect(" AND ".join(parts) if parts else "*:*", convert, limit=limit)

    async def search_datasets_advanced(
        self,
        dataset_name: str | None = None,
        dataset_title: str | None = None,
        owner_org: str | None = None,
        resource_url: str | None = None,
        resource_name: str | None = None,
        dataset_description: str | None = None,
        resource_description: str | None = None,
        resource_format: str | None = None,
        search_term: str | None = None,
        filter_list: list[str] | None = None,
        timestamp: str | None = None,
        server: str = "global",
    ) -> list[Dataset]:
        """Search by dataset fields with resource-level filters, like the EP API."""
        datasets, _ = await self.search_datasets_advanced_page(
            dataset_name=dataset_name,
            dataset_title=dataset_title,
            owner_org=owner_org,
            resource_url=resource_url,
            resource_name=resource_name,
            dataset_description=dataset_description,
            resource_description=resource_description,
            resource_format=resource_format,
            search_term=search_term,
            filter_list=filter_list,
            timestamp=timestamp,
            server=server,
        )
        return datasets

    async def search_datasets_advanced_page(
        self,
        dataset_name: str | None = None,
        dataset_title: str | None = None,
        owner_org: str | None = None,
        resource_url: str | None = None,
        resource_name: str | None = None,
        dataset_description: str | None = None,
        resource_description: str | None = None,
        resource_format: str | None = None,
        search_term: str | None = None,
        filter_list: list[str] | None = None,
        timestamp: str | None = None,
        server: str = "global",
        limit: int | None = None,
    ) -> tuple[list[Dataset], int]:
        """Like ``search_datasets_advanced`` but bounded, also returning the total hit count."""
        self._require_global(server)
        params = list(filter_list or [])
        if search_term:
            params.append(search_term)
        else:
            if dataset_name:
                params.append(f"name:{dataset_name}")
            if dataset_title:
                params.append(f"title:{dataset_title}")
            if owner_org:
                params.append(f"organization:{owner_org}")
            if dataset_description:
                params.append(f"notes:{dataset_description}")
        fq = sort = None
        if timestamp:
            fq, nearest_only, sort = _timestamp_query(timestamp)
            if nearest_only:
                limit = nearest_only
        keywords = [word.strip().lower() for word in search_term.split(",")] if search_term else []

        def resource_matches(res: dict[str, Any]) -> bool:
            return (
                (not resource_url or res.get("url") == resource_url)
                and (not resource_name or res.get("name") == resource_name)
                and (not resource_description or res.get("description") == resource_description)
                and (
                    not resource_format
                    or str(res.get("format") or "").lower() == resource_format.lower()
                )
            )

        def convert(row: dict[str, Any]) -> Dataset | None:
            matching = [
                res
                for res in row.get("resources") or []
                if isinstance(res, dict) and resource_matches(res)
            ]
            if not matching:
                return None
            dataset = _ckan_to_dataset(row, matching)
            text = str(dataset.model_dump()).lower()
            return dataset if all(word in text for word in keywords) else None

        return await self._collect(
            " AND ".join(params) if params else "*:*", convert, fq=fq, sort=sort, limit=limit
        )

    async def get_dataset(self, identifier: str, server: str = "global") -> Dataset | None:
        """Fetch one dataset by ID or name, or ``None`` if the catalog has no such dataset."""
        self._require_global(server)
        try:
            result = await self._action("package_show", {"id": identifier})
        except Exception as exc:
            if "HTTP 404" in str(exc) or "HTTP 403" in str(exc):
                return None
            raise
        return _ckan_to_dataset(result) if isinstance(result, dict) else None


def create_ndp_client() -> NDPClient | NDPCatalogClient:
    """Select the NDP backend from the environment.

    ``NDP_API_URL`` selects a self-hosted NDP Endpoint API (required for the
    ``local`` and ``pre_ckan`` catalogs). Otherwise the public NDP CKAN catalog
    is used, overridable with ``NDP_CATALOG_URL``.
    """
    api_url = os.getenv(NDP_API_URL_ENV, "").strip()
    if api_url:
        return NDPClient(api_url)
    catalog_url = os.getenv(NDP_CATALOG_URL_ENV, "").strip() or DEFAULT_NDP_CATALOG_URL
    return NDPCatalogClient(catalog_url)


ndp_client = create_ndp_client()


@mcp.tool(
    name="list_organizations",
    title="List Organizations",
    description="List organizations available in the National Data Platform.",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    tags={"organizations", "catalogs"},
)
async def list_organizations(
    name_filter: Annotated[
        str | None, Field(description="Filter organizations by name substring match")
    ] = None,
    server: Annotated[
        str, Field(description="Server to query: 'local', 'global', or 'pre_ckan'")
    ] = "global",
) -> ListOrganizationsResult:
    """List organizations from the National Data Platform."""
    try:
        organizations = await ndp_client.list_organizations(name_filter, server)

        return {
            "organizations": organizations,
            "count": len(organizations),
            "server": server,
            "name_filter": name_filter,
            "_meta": {"tool": "list_organizations", "status": "success"},
        }
    except Exception as e:
        raise ToolError(str(e)) from e


@mcp.tool(
    name="search_datasets",
    title="Search Datasets",
    description="Search for datasets in the NDP using term-based or field-specific criteria.",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    tags={"datasets", "search"},
)
async def search_datasets(
    search_terms: Annotated[
        list[str] | None, Field(description="Terms for simple search across all fields")
    ] = None,
    search_keys: Annotated[
        list[str] | None, Field(description="Corresponding keys for each search term")
    ] = None,
    dataset_name: Annotated[
        str | None, Field(description="Exact or partial dataset name to match")
    ] = None,
    dataset_title: Annotated[str | None, Field(description="Dataset title to search for")] = None,
    owner_org: Annotated[
        str | None, Field(description="Organization name that owns the dataset")
    ] = None,
    resource_url: Annotated[str | None, Field(description="URL of dataset resource")] = None,
    resource_name: Annotated[str | None, Field(description="Name of dataset resource")] = None,
    dataset_description: Annotated[
        str | None, Field(description="Text to search in dataset descriptions")
    ] = None,
    resource_description: Annotated[
        str | None, Field(description="Text to search in resource descriptions")
    ] = None,
    resource_format: Annotated[
        str | None, Field(description="Resource format (e.g., CSV, JSON, NetCDF)")
    ] = None,
    search_term: Annotated[
        str | None, Field(description="Comma-separated terms to search across all fields")
    ] = None,
    filter_list: Annotated[
        list[str] | None, Field(description="Field filters in format 'key:value'")
    ] = None,
    timestamp: Annotated[str | None, Field(description="Filter by timestamp field")] = None,
    server: Annotated[str, Field(description="Server to search: 'local' or 'global'")] = "global",
    limit: Annotated[
        str | int | None, Field(description="Maximum results to return (default: 20)")
    ] = None,
) -> SearchDatasetsResult:
    """Search for datasets in the National Data Platform."""
    try:
        # Convert limit to integer if it's a string
        if isinstance(limit, str):
            try:
                limit = int(limit)
            except ValueError:
                limit = None

        # Apply limit if specified, or default limit of 20 to prevent huge responses
        effective_limit = limit if limit and limit > 0 else 20

        advanced_filters: dict[str, Any] = {
            "dataset_name": dataset_name,
            "dataset_title": dataset_title,
            "owner_org": owner_org,
            "resource_url": resource_url,
            "resource_name": resource_name,
            "dataset_description": dataset_description,
            "resource_description": resource_description,
            "resource_format": resource_format,
            "search_term": search_term,
            "filter_list": filter_list,
            "timestamp": timestamp,
            "server": server,
        }
        if isinstance(ndp_client, NDPCatalogClient):
            # Fetch only the page needed; the catalog reports the full hit count.
            if search_terms:
                datasets, total_found = await ndp_client.search_datasets_simple_page(
                    terms=search_terms, keys=search_keys, server=server, limit=effective_limit
                )
            else:
                datasets, total_found = await ndp_client.search_datasets_advanced_page(
                    **advanced_filters, limit=effective_limit
                )
        elif search_terms:
            datasets = await ndp_client.search_datasets_simple(
                terms=search_terms, keys=search_keys, server=server
            )
            total_found = len(datasets)
        else:
            datasets = await ndp_client.search_datasets_advanced(**advanced_filters)
            total_found = len(datasets)

        was_limited = total_found > effective_limit
        datasets = datasets[:effective_limit]

        # Convert datasets to dict format
        dataset_dicts = [dataset.model_dump() for dataset in datasets]

        return cast(
            SearchDatasetsResult,
            {
                "datasets": dataset_dicts,
                "count": len(dataset_dicts),
                "total_found": total_found
                if not was_limited
                else f"{len(dataset_dicts)} of {total_found}",
                "server": server,
                "search_parameters": {
                    "search_terms": search_terms,
                    "search_keys": search_keys,
                    "dataset_name": dataset_name,
                    "dataset_title": dataset_title,
                    "owner_org": owner_org,
                    "resource_format": resource_format,
                    "search_term": search_term,
                    "filter_list": filter_list,
                    "limit": limit,
                },
                "_meta": {"tool": "search_datasets", "status": "success"},
            },
        )
    except Exception as e:
        raise ToolError(str(e)) from e


@mcp.tool(
    name="get_dataset_details",
    title="Dataset Details",
    description="Retrieve detailed metadata for a specific dataset by ID or name.",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    tags={"datasets", "metadata"},
)
async def get_dataset_details(
    dataset_identifier: Annotated[
        str, Field(description="The dataset ID or name to retrieve details for")
    ],
    identifier_type: Annotated[str, Field(description="Type of identifier: 'id' or 'name'")] = "id",
    server: Annotated[str, Field(description="Server to query: 'local' or 'global'")] = "global",
) -> GetDatasetDetailsResult:
    """Get detailed information about a specific dataset."""
    try:
        # Search for the specific dataset
        if isinstance(ndp_client, NDPCatalogClient) and identifier_type in ("id", "name"):
            found = await ndp_client.get_dataset(dataset_identifier, server=server)
            matching_dataset = (
                found
                if found is not None and getattr(found, identifier_type) == dataset_identifier
                else None
            )
        elif identifier_type == "id":
            datasets = await ndp_client.search_datasets_advanced(server=server)
            matching_dataset = next((d for d in datasets if d.id == dataset_identifier), None)
        else:
            datasets = await ndp_client.search_datasets_advanced(
                dataset_name=dataset_identifier, server=server
            )
            matching_dataset = next((d for d in datasets if d.name == dataset_identifier), None)

        if not matching_dataset:
            raise ToolError(f"Dataset not found with {identifier_type}: {dataset_identifier}")

        # Return detailed dataset information
        dataset_dict = matching_dataset.model_dump()

        return cast(
            GetDatasetDetailsResult,
            {
                "dataset": dataset_dict,
                "identifier_used": {"type": identifier_type, "value": dataset_identifier},
                "server": server,
                "resource_count": len(dataset_dict.get("resources", [])),
                "_meta": {"tool": "get_dataset_details", "status": "success"},
            },
        )
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(str(e)) from e


@mcp.tool(
    name="stage_resource",
    title="Stage Resource",
    description=(
        "Download/stage an HTTP(S) or OSDF/Pelican dataset resource to a local "
        "file and return its local_path, size, and content-type."
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False},
    tags={"datasets", "staging", "download"},
)
async def stage_resource(
    url: Annotated[
        str,
        Field(description="HTTP(S) or osdf:// URL of the dataset resource to stage."),
    ],
    output_name: Annotated[
        str | None,
        Field(description="Optional filename for the staged file; derived from URL if omitted."),
    ] = None,
    output_dir: Annotated[
        str | None,
        Field(
            description=(
                "Optional destination directory for the staged file. Defaults to "
                "CLIO_KIT_ARTIFACTS or the current working directory if omitted."
            )
        ),
    ] = None,
    max_bytes: Annotated[
        int | str | None,
        Field(description="Maximum download size in bytes (default 50 MiB)."),
    ] = None,
    resource_size_bytes: Annotated[
        int | str | None,
        Field(description="Advertised resource size (bytes or '1.4 GB') for OSDF size checks."),
    ] = None,
) -> StageResourceResult:
    """Stage a dataset resource to the configurable artifacts root.

    HTTP(S) resources are streamed with a size cap. ``osdf://`` resources are
    staged via the local ``pelican`` CLI when available. Returns ``local_path``,
    ``size_bytes``, and ``content_type``.
    """
    target_url = (url or "").strip()
    if not target_url:
        raise ToolError("A non-empty resource URL is required.")
    scheme_ok = target_url.lower().startswith(("http://", "https://", "osdf://"))
    if not scheme_ok:
        raise ToolError(f"Unsupported resource URL scheme: {target_url}")

    max_stage_bytes = _clean_max_bytes(max_bytes)
    filename_source = output_name or Path(target_url.split("?", 1)[0]).name
    filename = _safe_filename(filename_source, default="ndp-resource")
    output_path = _validate_output_path(
        filename, default_name="ndp-resource", output_dir=output_dir
    )

    is_osdf = target_url.lower().startswith("osdf://")
    if is_osdf:
        return cast(
            StageResourceResult,
            await asyncio.to_thread(
                _stage_pelican_resource,
                url=target_url,
                output_path=output_path,
                max_bytes=max_stage_bytes,
                resource_size_bytes=_parse_resource_size_bytes(resource_size_bytes),
            ),
        )

    timeout = httpx.Timeout(_HTTP_READ_TIMEOUT_S, connect=_HTTP_CONNECT_TIMEOUT_S)
    content_type: str | None = None
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path = output_path.with_name(f"{output_path.name}.part")
            partial_path.unlink(missing_ok=True)
            async with client.stream("GET", target_url) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type")
                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        advertised = int(content_length)
                    except ValueError:
                        advertised = 0
                    if advertised > max_stage_bytes:
                        raise ToolError(
                            f"Resource is {advertised} bytes, exceeding the staging "
                            f"limit of {max_stage_bytes} bytes. Increase max_bytes "
                            "intentionally or select a smaller resource."
                        )
                total = 0
                with partial_path.open("wb") as handle:
                    async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > max_stage_bytes:
                            handle.close()
                            partial_path.unlink(missing_ok=True)
                            raise ToolError(
                                f"Resource exceeded the staging limit of "
                                f"{max_stage_bytes} bytes while downloading. Increase "
                                "max_bytes intentionally or select a smaller resource."
                            )
                        handle.write(chunk)
            partial_path.replace(output_path)
    except ToolError:
        raise
    except httpx.HTTPError as exc:
        raise ToolError(f"Could not download resource from {target_url}: {exc}") from exc

    size = output_path.stat().st_size
    return {
        "ok": True,
        "local_path": str(output_path),
        "size_bytes": size,
        "content_type": content_type,
        "url": target_url,
        "method": "http",
        "_meta": {"tool": "stage_resource", "status": "success"},
    }


@mcp.resource("ndp://catalogs")
def available_catalogs() -> dict[str, Any]:
    """List of available NDP dataset catalogs."""
    if isinstance(ndp_client, NDPCatalogClient):
        return {
            "catalogs": ["global"],
            "backend": {"type": "ckan", "url": ndp_client.base_url, "env": NDP_CATALOG_URL_ENV},
            "description": (
                "Public NDP catalog. Set NDP_API_URL to an NDP Endpoint API to also "
                "reach the 'local' and 'pre_ckan' catalogs."
            ),
        }
    return {
        "catalogs": ["global", "local", "pre_ckan"],
        "backend": {"type": "endpoint_api", "url": ndp_client.base_url, "env": NDP_API_URL_ENV},
        "description": "Available NDP data catalogs",
    }


@mcp.prompt()
def explore_datasets(query: str) -> list[Message]:
    """Guided workflow for discovering and exploring scientific datasets."""
    return [
        Message(
            f"I want to find datasets related to '{query}'. "
            "Search available catalogs, show me the top results, "
            "and provide details on the most relevant one."
        ),
    ]


def main() -> None:
    """Main entry point for the NDP MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="NDP MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    transport = args.transport or os.getenv("MCP_TRANSPORT", "stdio")
    if transport == "http":
        mcp.run(transport="http", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
