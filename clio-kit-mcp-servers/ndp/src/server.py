import asyncio
import json
import os
import sys
from typing import Any

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import BaseModel, Field

# Path and environment setup
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

# Initialize FastMCP server instance
mcp: FastMCP = FastMCP("NDPServer")


class Dataset(BaseModel):
    """Model for dataset information from NDP API."""

    id: str
    name: str
    title: str
    owner_org: str | None = None
    notes: str | None = None
    resources: list[dict[str, Any]] = Field(default_factory=list)
    extras: dict[str, Any] | None = None


class NDPClient:
    """Client for interacting with NDP API with retry logic and error handling."""

    def __init__(self, base_url: str = "http://155.101.6.191:8003"):
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(30.0)
        self.max_retries = 3
        self.retry_delay = 1.0

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
                raise Exception(f"Request timed out after {self.max_retries} attempts") from None
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2**attempt))
                    continue
                raise Exception(f"HTTP {e.response.status_code}: {e.response.text}") from e
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2**attempt))
                    continue
                raise Exception(f"Request failed: {str(e)}") from e

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


# Initialize NDP client
ndp_client = NDPClient()


@mcp.tool(
    name="list_organizations",
    description=(  # noqa: E501
        "List organizations available in the National Data Platform. This tool should "
        "always be called before searching to verify organization names are correctly "
        "formatted. Supports filtering by organization name and selecting different "
        "servers (local, global, pre_ckan)."
    ),
)
async def list_organizations(
    name_filter: str | None = None, server: str = "global"
) -> dict[str, Any]:
    """
    List organizations from the National Data Platform.

    This tool retrieves a list of all organizations available in the NDP. It's recommended
    to call this tool before performing searches to ensure organization names are correctly
    formatted and to discover available organizations for filtering search results.

    Args:
        name_filter (str, optional): Filter organizations by name substring match
        server (str, optional): Server to query - 'local', 'global', or 'pre_ckan'
            (default: 'global')

    Returns:
        dict: Contains list of organization names and metadata about the request
    """
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
        return {
            "content": [{"text": json.dumps({"error": str(e)})}],
            "_meta": {"tool": "list_organizations", "error": type(e).__name__},
            "isError": True,
        }


@mcp.tool(
    name="search_datasets",
    description=(
        "Search for datasets in the National Data Platform using simple or advanced search "
        "criteria. Supports both term-based searches and field-specific filtering. Use this "
        "tool to discover datasets by keywords, organization, format, or other metadata. "
        "Results are automatically limited to 20 by default to prevent context overflow - use "
        "the limit parameter to adjust this."
    ),
)
async def search_datasets(
    search_terms: list[str] | None = None,
    search_keys: list[str] | None = None,
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
    limit: str | int | None = None,
) -> dict[str, Any]:
    """
    Search for datasets in the National Data Platform using various search criteria.

    This tool provides comprehensive dataset search capabilities with both simple term-based
    search and advanced field-specific filtering. When searching by organization, it's
    recommended to first call list_organizations to ensure the organization name is
    correctly formatted.

    WORKFLOW RECOMMENDATION:
    1. If searching by organization, first call list_organizations to verify organization names
    2. Use simple search (search_terms) for general keyword searches
    3. Use advanced search (specific field parameters) for precise filtering
    4. Consider using limit parameter if expecting large result sets

    Args:
        search_terms (List[str], optional): List of terms for simple search across all fields
        search_keys (List[str], optional): Corresponding keys for each search term
            (use null for global search)
        dataset_name (str, optional): Exact or partial dataset name to match
        dataset_title (str, optional): Dataset title to search for
        owner_org (str, optional): Organization name that owns the dataset
        resource_url (str, optional): URL of dataset resource
        resource_name (str, optional): Name of dataset resource
        dataset_description (str, optional): Text to search in dataset descriptions
        resource_description (str, optional): Text to search in resource descriptions
        resource_format (str, optional): Resource format (e.g., CSV, JSON, NetCDF)
        search_term (str, optional): Comma-separated terms to search across all fields
        filter_list (List[str], optional): Field filters in format 'key:value'
        timestamp (str, optional): Filter by timestamp field
        server (str, optional): Server to search - 'local' or 'global' (default: 'global')
        limit (int or str, optional): Maximum number of results to return
            (default: 20 to prevent context overflow)

    Returns:
        dict: Contains list of matching datasets with detailed metadata
    """
    try:
        # Determine which search method to use
        if search_terms:
            # Use simple search
            datasets = await ndp_client.search_datasets_simple(
                terms=search_terms, keys=search_keys, server=server
            )
        else:
            # Use advanced search
            datasets = await ndp_client.search_datasets_advanced(
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

        # Store total count before limiting
        total_found = len(datasets)

        # Convert limit to integer if it's a string
        if isinstance(limit, str):
            try:
                limit = int(limit)
            except ValueError:
                limit = None

        # Apply limit if specified, or default limit of 20 to prevent huge responses
        effective_limit = limit if limit and limit > 0 else 20
        was_limited = len(datasets) > effective_limit

        if len(datasets) > effective_limit:
            datasets = datasets[:effective_limit]

        # Convert datasets to dict format
        dataset_dicts = [dataset.model_dump() for dataset in datasets]

        return {
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
        }
    except Exception as e:
        return {
            "content": [{"text": json.dumps({"error": str(e)})}],
            "_meta": {"tool": "search_datasets", "error": type(e).__name__},
            "isError": True,
        }


@mcp.tool(
    name="get_dataset_details",
    description=(
        "Retrieve detailed information about a specific dataset using its ID or name. Returns "
        "comprehensive metadata including all resources, descriptions, and additional fields. "
        "Use this after finding datasets with search_datasets to get complete information."
    ),
)
async def get_dataset_details(
    dataset_identifier: str, identifier_type: str = "id", server: str = "global"
) -> dict[str, Any]:
    """
    Get detailed information about a specific dataset.

    This tool retrieves comprehensive information about a dataset using either its
    unique ID or name. Use this tool after finding datasets with search_datasets
    to get complete details including all resources, metadata, and additional fields.

    Args:
        dataset_identifier (str): The dataset ID or name to retrieve details for
        identifier_type (str, optional): Type of identifier - 'id' or 'name' (default: 'id')
        server (str, optional): Server to query - 'local' or 'global' (default: 'global')

    Returns:
        dict: Detailed dataset information including all metadata and resources
    """
    try:
        # Search for the specific dataset
        if identifier_type == "id":
            # Search by dataset ID - this would typically be a more specific API call
            # For now, we'll use the search functionality
            datasets = await ndp_client.search_datasets_advanced(server=server)
            matching_dataset = next((d for d in datasets if d.id == dataset_identifier), None)
        else:
            # Search by dataset name
            datasets = await ndp_client.search_datasets_advanced(
                dataset_name=dataset_identifier, server=server
            )
            matching_dataset = next((d for d in datasets if d.name == dataset_identifier), None)

        if not matching_dataset:
            return {
                "content": [
                    {
                        "text": json.dumps(
                            {
                                "error": (
                                    f"Dataset not found with {identifier_type}: "
                                    f"{dataset_identifier}"
                                )
                            }
                        )
                    }
                ],
                "_meta": {"tool": "get_dataset_details", "error": "NotFound"},
                "isError": True,
            }

        # Return detailed dataset information
        dataset_dict = matching_dataset.model_dump()

        return {
            "dataset": dataset_dict,
            "identifier_used": {"type": identifier_type, "value": dataset_identifier},
            "server": server,
            "resource_count": len(dataset_dict.get("resources", [])),
            "_meta": {"tool": "get_dataset_details", "status": "success"},
        }
    except Exception as e:
        return {
            "content": [{"text": json.dumps({"error": str(e)})}],
            "_meta": {"tool": "get_dataset_details", "error": type(e).__name__},
            "isError": True,
        }


def main() -> None:
    """Main entry point with transport selection."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--fastapi":
        # FastAPI mode for development/testing
        mcp.run(transport="fastapi", host="localhost", port=8000)  # type: ignore[arg-type]
    else:
        # Standard stdio mode for production
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
