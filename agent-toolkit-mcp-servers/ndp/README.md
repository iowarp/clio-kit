# Ndp MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/ndp-mcp.svg)](https://pypi.org/project/ndp-mcp/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

**Part of [Agent Toolkit](https://iowarp.github.io/agent-toolkit/) - Gnosis Research Center**

The National Data Platform (NDP) MCP server provides comprehensive access to search and discover datasets across multiple CKAN instances within the National Data Platform ecosystem. This server enables seamless interaction with the NDP API to find scientific datasets, explore organizations, and r...

## Quick Start

```bash
uvx iowarp-agent-toolkit mcp-server ndp
```

## Documentation

- **Full Documentation**: [Agent Toolkit Website](https://iowarp.github.io/agent-toolkit/)
- **Installation Guide**: See [INSTALLATION.md](../../../CLAUDE.md#setup--installation)
- **Contributing**: See [Contribution Guide](https://github.com/iowarp/agent-toolkit/wiki/Contribution)

---

## 🛠️ Installation

### Requirements
- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended)
- Linux/macOS environment (Windows supported)

<details>
<summary><b>Install in Cursor</b></summary>

Go to: `Settings` -> `Cursor Settings` -> `MCP` -> `Add new global MCP server`

Pasting the following configuration into your Cursor `~/.cursor/mcp.json` file:

```json
{
  "mcpServers": {
    "ndp-mcp": {
      "command": "uvx",
      "args": ["iowarp-agent-toolkit", "mcp-server", "ndp"]
    }
  }
}
```

</details>

<details>
<summary><b>Install in VS Code</b></summary>

Add the following to your VS Code MCP configuration:

```json
{
  "mcpServers": {
    "ndp-mcp": {
      "command": "uvx",
      "args": ["iowarp-agent-toolkit", "mcp-server", "ndp"]
    }
  }
}
```

</details>

<details>
<summary><b>Install in Claude Code</b></summary>

Run the following command in your terminal:

```bash
uvx iowarp-agent-toolkit mcp-server ndp
```

</details>

<details>
<summary><b>Install in Claude Desktop</b></summary>

Add the following to your `claude_desktop_config.json` file:

```json
{
  "mcpServers": {
    "ndp-mcp": {
      "command": "uvx",
      "args": ["iowarp-agent-toolkit", "mcp-server", "ndp"]
    }
  }
}
```

</details>

<details>
<summary><b>Manual Setup</b></summary>

1. Clone the repository:
```bash
git clone https://github.com/iowarp/agent-toolkit.git
cd agent-toolkit/agent-toolkit-mcp-servers/ndp
```

2. Install dependencies using uv:
```bash
uv sync --all-extras --dev
```

3. Run the server:
```bash
uv run python src/server.py
```

</details>

## Capabilities

### `list_organizations`
**Description**: List organizations from the National Data Platform.

**Parameters**:
- `name_filter` (str, optional): Filter organizations by name substring match
- `server` (str, optional): Server to query - 'local', 'global', or 'pre_ckan' (default: 'global')

**Returns**: dict: Contains list of organization names and metadata about the request

### `search_datasets`
**Description**: Search for datasets in the National Data Platform using various search criteria.

**Parameters**:
- `search_terms` (List[str], optional): List of terms for simple search across all fields
- `search_keys` (List[str], optional): Corresponding keys for each search term (use null for global search)
- `dataset_name` (str, optional): Exact or partial dataset name to match
- `dataset_title` (str, optional): Dataset title to search for
- `owner_org` (str, optional): Organization name that owns the dataset
- `resource_url` (str, optional): URL of dataset resource
- `resource_name` (str, optional): Name of dataset resource
- `dataset_description` (str, optional): Text to search in dataset descriptions
- `resource_description` (str, optional): Text to search in resource descriptions
- `resource_format` (str, optional): Resource format (e.g., CSV, JSON, NetCDF)
- `search_term` (str, optional): Comma-separated terms to search across all fields
- `filter_list` (List[str], optional): Field filters in format 'key:value'
- `timestamp` (str, optional): Filter by timestamp field
- `server` (str, optional): Server to search - 'local' or 'global' (default: 'global')
- `limit` (int or str, optional): Maximum number of results to return (default: 20 to prevent context overflow)

**Returns**: dict: Contains list of matching datasets with detailed metadata

### `get_dataset_details`
**Description**: Get detailed information about a specific dataset.

**Parameters**:
- `dataset_identifier` (str): The dataset ID or name to retrieve details for
- `identifier_type` (str, optional): Type of identifier - 'id' or 'name' (default: 'id')
- `server` (str, optional): Server to query - 'local' or 'global' (default: 'global')

**Returns**: dict: Detailed dataset information including all metadata and resources
## Examples

### 1. Discover Available Organizations
```
List all organizations in the National Data Platform to see what data is available
```

**Tools called:**
- `list_organizations` - Retrieves all available organizations from the global server

This prompt will:
- Return a comprehensive list of organizations contributing data to NDP
- Show the total count of organizations available
- Provide foundation for targeted dataset searches

### 2. Search for Climate Data from NOAA
```
I want to find climate datasets from NOAA. First show me organizations that contain "noaa" and then search for climate-related datasets from that organization.
```

**Tools called:**
- `list_organizations` - Filters organizations containing "noaa" to verify correct name formatting
- `search_datasets` - Searches for datasets with climate terms from the verified NOAA organization

This prompt will:
- Verify the correct NOAA organization name format
- Find all climate-related datasets published by NOAA
- Return dataset metadata including titles, descriptions, and resource information

### 3. Find CSV Datasets about Temperature Monitoring
```
Find datasets that contain temperature sensor data in CSV format, limit to 10 results
```

**Tools called:**
- `search_datasets` - Searches with advanced parameters for temperature data in CSV format

This prompt will:
- Search across all fields for temperature-related terms
- Filter results to only CSV format resources
- Limit results to 10 datasets to manage response size
- Return detailed metadata for each matching dataset

### 4. Get Complete Information About a Specific Dataset
```
I found a dataset with ID "dataset-12345-climate-temp" in my search. Give me all the details about this dataset including all its resources and metadata.
```

**Tools called:**
- `get_dataset_details` - Retrieves comprehensive information for the specified dataset ID

This prompt will:
- Fetch complete dataset metadata using the provided ID
- Return all associated resources with download URLs and formats
- Provide additional metadata fields and processing information
- Show resource count and detailed descriptions

### 5. Multi-Server Search Workflow
```
Search for oceanographic datasets on both global and local servers, focusing on those from research institutions
```

**Tools called:**
- `list_organizations` - First on global server, then on local server to compare available organizations
- `search_datasets` - Search global server for oceanographic data
- `search_datasets` - Search local server for oceanographic data

This prompt will:
- Compare organization availability across different NDP servers
- Search multiple server instances for comprehensive coverage
- Filter results by research institution organizations
- Provide comparative analysis of dataset availability

### 6. Advanced Filtering for Specific Research Needs
```
Find datasets that have "satellite imagery" in their description, are in NetCDF format, and were published after 2020. Also show me organizations that might have earth observation data.
```

**Tools called:**
- `list_organizations` - Filter organizations that might contain earth observation data
- `search_datasets` - Advanced search with description, format, and timestamp filtering

This prompt will:
- Identify organizations likely to have earth observation datasets
- Use advanced field-specific search parameters
- Filter by resource format and temporal constraints
- Return highly targeted results matching specific research criteria