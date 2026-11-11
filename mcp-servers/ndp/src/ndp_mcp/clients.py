"""NDP backends: the public CKAN catalog and a self-hosted NDP Endpoint API."""

import asyncio
import json
import os
import re
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import BaseModel, Field


class Dataset(BaseModel):
    """Model for dataset information from NDP API."""

    id: str
    name: str
    title: str
    owner_org: str | None = None
    notes: str | None = None
    resources: list[dict[str, Any]] = Field(default_factory=list)
    extras: dict[str, Any] | None = None


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

    def __init__(self, base_url: str | None = None):
        base_url = base_url or os.getenv(NDP_API_URL_ENV, "").strip()
        if not base_url:
            raise ValueError(
                f"An NDP Endpoint API needs a base URL. Pass one or set {NDP_API_URL_ENV}."
            )
        super().__init__(base_url)

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
