#!/usr/bin/env python3
"""
NDP MCP Server for National Data Platform integration.
Provides tools for dataset discovery, details, downloading, and geospatial analysis.
"""

import os
import sys
import json
import asyncio
import tempfile
import logging
from typing import Dict, List, Optional, Any
from fastmcp import FastMCP
from dotenv import load_dotenv

# NDP Client class definition
class NDPClient:
    """Client for interacting with the NDP API."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        import aiohttp
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_organizations(self) -> List[Dict]:
        """Fetch all organizations from the NDP API."""
        # For EarthScope, return a mock organization since the API structure is different
        return [{"id": "earthscope", "name": "EarthScope", "description": "EarthScope GNSS data"}]
    
    async def search_datasets(self, query: str, organization: Optional[str] = None, limit: int = 10) -> Dict:
        """Search for datasets in the NDP catalog."""
        try:
            # Try to discover EarthScope datasets by querying available endpoints
            discovered_datasets = []
            
            # Common EarthScope station patterns
            station_patterns = ["rhcl", "p041", "p042", "p043", "p044", "p045"]
            years = ["20", "21", "22", "23", "24"]  # Recent years
            
            for station in station_patterns:
                for year in years:
                    # Try to discover GeoJSON metadata
                    geojson_url = f"{self.base_url}/Earthscope_api/geojson/{station}.geojson"
                    csv_url = f"{self.base_url}/Earthscope_api/{station.upper()}.CI.LY_.{year}.csv"
                    
                    # Check if files exist by making HEAD requests
                    try:
                        async with self.session.head(geojson_url) as response:
                            if response.status == 200:
                                # File exists, create dataset entry
                                dataset_id = f"earthscope_{station}_{year}"
                                discovered_datasets.append({
                                    "id": dataset_id,
                                    "title": f"EarthScope GNSS Data - {station.upper()} Station ({year})",
                                    "organization": {"name": "EarthScope"},
                                    "created": f"20{year}-01-01T00:00:00Z",
                                    "modified": f"20{year}-12-31T23:59:59Z",
                                    "resources": [
                                        {
                                            "name": f"{station.upper()} Station Metadata",
                                            "format": "geojson",
                                            "url": geojson_url
                                        },
                                        {
                                            "name": f"{station.upper()} GNSS Time Series",
                                            "format": "csv",
                                            "url": csv_url
                                        }
                                    ]
                                })
                    except:
                        continue
            
            # If no datasets found, return empty result
            if not discovered_datasets:
                return {"datasets": []}
            
            # Sort by modification date (newest first) and limit results
            discovered_datasets.sort(key=lambda x: x.get("modified", ""), reverse=True)
            return {"datasets": discovered_datasets[:limit]}
            
        except Exception as e:
            logger.error(f"Error searching EarthScope datasets: {e}")
            return {"datasets": []}
    
    async def get_dataset_details(self, dataset_id: str) -> Dict:
        """Get complete metadata for a specific dataset."""
        try:
            # Parse dataset ID to extract station and year
            if dataset_id.startswith("earthscope_"):
                parts = dataset_id.split("_")
                if len(parts) >= 3:
                    station = parts[1]
                    year = parts[2]
                    
                    # Construct URLs
                    geojson_url = f"{self.base_url}/Earthscope_api/geojson/{station}.geojson"
                    csv_url = f"{self.base_url}/Earthscope_api/{station.upper()}.CI.LY_.{year}.csv"
                    
                    # Verify files exist
                    geojson_exists = False
                    csv_exists = False
                    
                    try:
                        async with self.session.head(geojson_url) as response:
                            geojson_exists = response.status == 200
                    except:
                        pass
                    
                    try:
                        async with self.session.head(csv_url) as response:
                            csv_exists = response.status == 200
                    except:
                        pass
                    
                    if geojson_exists or csv_exists:
                        resources = []
                        if geojson_exists:
                            resources.append({
                                "name": f"{station.upper()} Station Metadata",
                                "format": "geojson",
                                "url": geojson_url,
                                "description": "Station location and metadata in GeoJSON format"
                            })
                        
                        if csv_exists:
                            resources.append({
                                "name": f"{station.upper()} GNSS Time Series",
                                "format": "csv",
                                "url": csv_url,
                                "description": "GNSS time series data in CSV format"
                            })
                        
                        return {
                            "id": dataset_id,
                            "title": f"EarthScope GNSS Data - {station.upper()} Station ({year})",
                            "name": dataset_id,
                            "owner_org": "EarthScope",
                            "metadata_created": f"20{year}-01-01T00:00:00Z",
                            "metadata_modified": f"20{year}-12-31T23:59:59Z",
                            "notes": f"GNSS time series data from EarthScope {station.upper()} station for year 20{year}",
                            "resources": resources
                        }
            
            # If dataset not found or invalid format, return empty
            return {}
            
        except Exception as e:
            logger.error(f"Error getting dataset details: {e}")
            return {}
    
    async def download_file(self, url: str) -> bytes:
        """Download a file from a URL."""
        async with self.session.get(url) as response:
            response.raise_for_status()
            return await response.read()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp: FastMCP = FastMCP("NDPServer")

# NDP API configuration
NDP_BASE_URL = os.getenv("NDP_BASE_URL", "https://ds2.datacollaboratory.org")

# Search cache for chunking
search_cache = {}
CHUNK_THRESHOLD = 50

# Tool definitions
@mcp.tool(
    name="list_organizations",
    description="List all available organizations from the NDP API"
)
async def list_organizations_tool() -> dict:
    """List all available organizations from the NDP API."""
    return await _list_organizations()

@mcp.tool(
    name="search_datasets",
    description="Search for datasets across the NDP catalog with intelligent chunking"
)
async def search_datasets_tool(
    query: str, 
    organization: Optional[str] = None, 
    limit: int = 10
) -> dict:
    """Search for datasets across the NDP catalog."""
    return await _search_datasets(query, organization, limit)

@mcp.tool(
    name="get_dataset_details",
    description="Retrieve complete metadata for a specific dataset"
)
async def get_dataset_details_tool(dataset_id: str) -> dict:
    """Retrieve complete metadata for a specific dataset."""
    return await _get_dataset_details(dataset_id)

@mcp.tool(
    name="download_dataset_resources",
    description="Download dataset files from dataset resources"
)
async def download_dataset_resources_tool(
    dataset_id: str, 
    resource_types: Optional[List[str]] = None
) -> dict:
    """Download dataset files from dataset resources."""
    return await _download_dataset_resources(dataset_id, resource_types)

@mcp.tool(
    name="download_file_from_url",
    description="Download a file directly from a URL and save it to a specified location"
)
async def download_file_from_url_tool(url: str, output_path: str) -> dict:
    """Download a file directly from a URL and save it to a specified location."""
    return await _download_file_from_url(url, output_path)

@mcp.tool(
    name="analyze_geospatial_data",
    description="Analyze geospatial data from a dataset with comprehensive insights"
)
async def analyze_geospatial_data_tool(dataset_id: str) -> dict:
    """Analyze geospatial data from a dataset."""
    return await _analyze_geospatial_data(dataset_id)

@mcp.tool(
    name="create_multi_series_plot",
    description="Create a multi-series line plot with multiple y-columns against a single x-column, perfect for time series data like GNSS measurements"
)
async def create_multi_series_plot_tool(
    file_path: str,
    x_column: str,
    y_columns: List[str],
    title: str = "Multi-Series Line Plot",
    output_path: str = "multi_series_plot.png",
    figure_size: str = "15x12",
    dpi: int = 300
) -> dict:
    """Create a multi-series line plot with multiple y-columns against a single x-column."""
    return await _create_multi_series_plot(file_path, x_column, y_columns, title, output_path, figure_size, dpi)

@mcp.tool(
    name="create_multi_panel_plot",
    description="Create a multi-panel plot with separate subplots for each y-column, perfect for GNSS time series with East, North, Up components"
)
async def create_multi_panel_plot_tool(
    file_path: str,
    x_column: str,
    y_columns: List[str],
    title: str = "Multi-Panel Time Series Plot",
    output_path: str = "multi_panel_plot.png",
    figure_size: str = "15x12",
    dpi: int = 300,
    layout: str = "vertical"
) -> dict:
    """Create a multi-panel plot with separate subplots for each y-column."""
    return await _create_multi_panel_plot(file_path, x_column, y_columns, title, output_path, figure_size, dpi, layout)

@mcp.tool(
    name="discover_latest_earthscope_datasets",
    description="Automatically discover the latest EarthScope datasets and extract their GeoJSON and CSV file URLs"
)
async def discover_latest_earthscope_datasets_tool() -> dict:
    """Automatically discover the latest EarthScope datasets and extract their GeoJSON and CSV file URLs."""
    return await _discover_latest_earthscope_datasets()

# Helper functions
async def _list_organizations() -> dict:
    """List all available organizations from the NDP API."""
    try:
        async with NDPClient(NDP_BASE_URL) as client:
            organizations = await client.get_organizations()
            
            result_text = "Available NDP Organizations:\n\n"
            for org in organizations:
                result_text += f"- **{org.get('name', 'Unknown')}** (ID: {org.get('id', 'N/A')})\n"
                if org.get('description'):
                    result_text += f"  Description: {org['description']}\n"
                result_text += "\n"
            
        return {"status": "success", "result": result_text, "organizations": organizations}
    except Exception as e:
        logger.error(f"Error listing organizations: {e}")
        return {"status": "error", "error": str(e)}
    
async def _search_datasets(query: str, organization: Optional[str] = None, limit: int = 10) -> dict:
    """Search for datasets across the NDP catalog with intelligent chunking."""
    try:
        # Check if search exists in cache
        cache_key = f"{query}_{organization}_{limit}"
        if cache_key in search_cache:
            chunk_id = search_cache[cache_key].get('next_chunk', 1)
            return await _get_chunk(cache_key, chunk_id)
        
        async with NDPClient(NDP_BASE_URL) as client:
            search_result = await client.search_datasets(query, organization, limit)
            
            results = search_result.get('result', {}).get('results', [])
            
            if len(results) > CHUNK_THRESHOLD:
                # Implement chunking
                search_cache[cache_key] = {
                    'results': results,
                    'next_chunk': 1,
                    'total_chunks': (len(results) + 9) // 10  # 10 results per chunk
                }
                return await _get_chunk(cache_key, 1)
            else:
                return _format_search_results(results, query, organization)
    
    except Exception as e:
        logger.error(f"Error searching datasets: {e}")
        return {"status": "error", "error": str(e)}
    
async def _get_chunk(cache_key: str, chunk_id: int) -> dict:
    """Get a specific chunk of search results."""
    if cache_key not in search_cache:
        raise Exception("Search results not found in cache")
    
    cache_data = search_cache[cache_key]
    results = cache_data['results']
    chunk_size = 10
    start_idx = (chunk_id - 1) * chunk_size
    end_idx = start_idx + chunk_size
    
    chunk_results = results[start_idx:end_idx]
    
    result_text = _format_search_results_text(chunk_results, cache_key, chunk_id, cache_data['total_chunks'])
    
    return {"status": "success", "result": result_text, "chunk_id": chunk_id, "total_chunks": cache_data['total_chunks']}

def _format_search_results(results: List[Dict], query: str, organization: Optional[str] = None) -> dict:
    """Format search results for display."""
    result_text = _format_search_results_text(results, None, 1, 1)
    return {"status": "success", "result": result_text, "results": results}

def _format_search_results_text(results: List[Dict], cache_key: Optional[str], chunk_id: int, total_chunks: int) -> str:
    """Format search results as text."""
    result_text = f"Found {len(results)} datasets"
    if total_chunks > 1:
        result_text += f" (Chunk {chunk_id} of {total_chunks})"
    result_text += ":\n\n"
    
    for i, dataset in enumerate(results, 1):
        result_text += f"{i}. **{dataset.get('title', dataset.get('name', 'Untitled'))}**\n"
        result_text += f"   - ID: `{dataset.get('id', 'N/A')}`\n"
        result_text += f"   - Organization: {dataset.get('owner_org', 'Unknown')}\n"
        if dataset.get('notes'):
            result_text += f"   - Description: {dataset['notes'][:100]}...\n"
        result_text += "\n"
    
    if total_chunks > 1 and chunk_id < total_chunks:
        result_text += f"\nTo get the next chunk, use the search_datasets tool with the same parameters.\n"
    
    return result_text
    
async def _get_dataset_details(dataset_id: str) -> dict:
    """Retrieve complete metadata for a specific dataset."""
    try:
        async with NDPClient(NDP_BASE_URL) as client:
            dataset = await client.get_dataset_details(dataset_id)
            
            if not dataset:
                return {"status": "error", "error": f"Dataset with ID '{dataset_id}' not found."}
            
            result_text = f"**Dataset Details: {dataset.get('title', dataset.get('name', 'Untitled'))}**\n\n"
            result_text += f"- **ID**: `{dataset.get('id', 'N/A')}`\n"
            result_text += f"- **Name**: {dataset.get('name', 'N/A')}\n"
            result_text += f"- **Organization**: {dataset.get('owner_org', 'N/A')}\n"
            result_text += f"- **Created**: {dataset.get('metadata_created', 'N/A')}\n"
            result_text += f"- **Modified**: {dataset.get('metadata_modified', 'N/A')}\n"
            
            if dataset.get('notes'):
                result_text += f"- **Description**: {dataset['notes']}\n"
            
            if dataset.get('tags'):
                result_text += f"- **Tags**: {', '.join([tag['name'] for tag in dataset['tags']])}\n"
            
            # Resources
            resources = dataset.get('resources', [])
            if resources:
                result_text += f"\n**Resources ({len(resources)}):**\n"
                for i, resource in enumerate(resources, 1):
                    result_text += f"{i}. **{resource.get('name', 'Unnamed Resource')}**\n"
                    result_text += f"   - Format: {resource.get('format', 'Unknown')}\n"
                    result_text += f"   - URL: {resource.get('url', 'N/A')}\n"
                    if resource.get('description'):
                        result_text += f"   - Description: {resource['description']}\n"
                    result_text += "\n"
            
            # Extras
            extras = dataset.get('extras', [])
            if extras:
                result_text += "**Additional Metadata:**\n"
                for extra in extras:
                    result_text += f"- {extra.get('key', 'Unknown')}: {extra.get('value', 'N/A')}\n"
            
        return {"status": "success", "result": result_text, "dataset": dataset}
        
    except Exception as e:
        logger.error(f"Error getting dataset details: {e}")
        return {"status": "error", "error": str(e)}

async def _download_file_from_url(url: str, output_path: str) -> dict:
    """Download a file directly from a URL and save it to a specified location."""
    try:
        import aiohttp
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    
                    with open(output_path, 'wb') as f:
                        f.write(content)
                    
                    file_size = len(content)
                    result_text = f"Successfully downloaded file from {url}\n"
                    result_text += f"Saved to: {output_path}\n"
                    result_text += f"File size: {file_size} bytes"
                    
                    return {
                        "status": "success",
                        "result": result_text,
                        "file_path": output_path,
                        "file_size": file_size,
                        "url": url
                    }
                else:
                    error_msg = f"Failed to download file. HTTP status: {response.status}"
                    return {"status": "error", "error": error_msg}
                    
    except Exception as e:
        logger.error(f"Error downloading file from URL: {e}")
        return {"status": "error", "error": str(e)}

async def _download_dataset_resources(dataset_id: str, resource_types: Optional[List[str]] = None) -> dict:
    """Download dataset files from dataset resources."""
    try:
        async with NDPClient(NDP_BASE_URL) as client:
            # Get dataset details first
            dataset = await client.get_dataset_details(dataset_id)
            if not dataset:
                return {"status": "error", "error": f"Dataset with ID '{dataset_id}' not found."}
            
            resources = dataset.get('resources', [])
            if not resources:
                return {"status": "error", "error": "No resources found for this dataset."}
            
            # Filter resources by type if specified
            if resource_types:
                resources = [r for r in resources if r.get('format', '').upper() in [rt.upper() for rt in resource_types]]
            
            result_text = f"**Downloading Resources for Dataset: {dataset.get('title', dataset.get('name', 'Untitled'))}**\n\n"
            
            downloaded_files = []
            for i, resource in enumerate(resources, 1):
                resource_url = resource.get('url')
                if not resource_url:
                    continue
                
                # Create filename from resource name or URL
                resource_name = resource.get('name', f'resource_{i}')
                file_extension = resource.get('format', '').lower()
                if not file_extension:
                    # Try to extract from URL
                    if '.' in resource_url.split('/')[-1]:
                        file_extension = resource_url.split('.')[-1]
                    else:
                        file_extension = 'dat'
                
                filename = f"{resource_name}.{file_extension}"
                
                try:
                    # Download the file
                    content = await client.download_file(resource_url)
                    
                    # Save to file
                    with open(filename, 'wb') as f:
                        f.write(content)
                    
                    downloaded_files.append({
                        'name': resource_name,
                        'filename': filename,
                        'size': len(content),
                        'format': resource.get('format', 'Unknown'),
                        'url': resource_url
                    })
                    
                    result_text += f"✅ **{resource_name}** - Downloaded successfully\n"
                    result_text += f"   - File: {filename}\n"
                    result_text += f"   - Size: {len(content)} bytes\n"
                    result_text += f"   - Format: {resource.get('format', 'Unknown')}\n\n"
                    
                except Exception as e:
                    result_text += f"❌ **{resource_name}** - Failed to download: {str(e)}\n\n"
            
            if not downloaded_files:
                return {"status": "error", "error": "No files were successfully downloaded."}
            
            return {
                "status": "success",
                "result": result_text,
                "downloaded_files": downloaded_files,
                "total_files": len(downloaded_files)
            }
            
    except Exception as e:
        logger.error(f"Error downloading dataset resources: {e}")
        return {"status": "error", "error": str(e)}

async def _analyze_geospatial_data(dataset_id: str) -> dict:
    """Analyze geospatial data from a dataset with comprehensive insights."""
    try:
        async with NDPClient(NDP_BASE_URL) as client:
            # Get dataset details
            dataset = await client.get_dataset_details(dataset_id)
            if not dataset:
                return {"status": "error", "error": f"Dataset with ID '{dataset_id}' not found."}
            
            result_text = f"**Geospatial Analysis: {dataset.get('title', dataset.get('name', 'Untitled'))}**\n\n"
            
            # Basic dataset info
            result_text += f"**Dataset Information:**\n"
            result_text += f"- ID: {dataset.get('id', 'N/A')}\n"
            result_text += f"- Organization: {dataset.get('owner_org', 'N/A')}\n"
            result_text += f"- Created: {dataset.get('metadata_created', 'N/A')}\n"
            result_text += f"- Modified: {dataset.get('metadata_modified', 'N/A')}\n\n"
            
            # Resource analysis
            resources = dataset.get('resources', [])
            if resources:
                result_text += f"**Resource Analysis ({len(resources)} resources):**\n"
                
                geojson_count = 0
                csv_count = 0
                other_count = 0
                
                for resource in resources:
                    format_type = resource.get('format', '').lower()
                    if 'geojson' in format_type:
                        geojson_count += 1
                    elif 'csv' in format_type:
                        csv_count += 1
                    else:
                        other_count += 1
                
                result_text += f"- GeoJSON files: {geojson_count}\n"
                result_text += f"- CSV files: {csv_count}\n"
                result_text += f"- Other formats: {other_count}\n\n"
                
                # Detailed resource info
                result_text += "**Resource Details:**\n"
                for i, resource in enumerate(resources, 1):
                    result_text += f"{i}. **{resource.get('name', 'Unnamed')}**\n"
                    result_text += f"   - Format: {resource.get('format', 'Unknown')}\n"
                    result_text += f"   - URL: {resource.get('url', 'N/A')}\n"
                    if resource.get('description'):
                        result_text += f"   - Description: {resource['description']}\n"
                    result_text += "\n"
            
            # Spatial extent analysis (if available)
            extras = dataset.get('extras', [])
            spatial_info = {}
            for extra in extras:
                key = extra.get('key', '').lower()
                if 'spatial' in key or 'bbox' in key or 'extent' in key:
                    spatial_info[key] = extra.get('value')
            
            if spatial_info:
                result_text += "**Spatial Information:**\n"
                for key, value in spatial_info.items():
                    result_text += f"- {key}: {value}\n"
                result_text += "\n"
            
            return {
                "status": "success",
                "result": result_text,
                "dataset_id": dataset_id,
                "total_resources": len(resources),
                "geojson_count": geojson_count if 'geojson_count' in locals() else 0,
                "csv_count": csv_count if 'csv_count' in locals() else 0,
                "other_count": other_count if 'other_count' in locals() else 0
            }
            
    except Exception as e:
        logger.error(f"Error analyzing geospatial data: {e}")
        return {"status": "error", "error": str(e)}

async def _create_multi_series_plot(file_path: str, x_column: str, y_columns: List[str], title: str, output_path: str, figure_size: str, dpi: int) -> dict:
    """Create a multi-series line plot with multiple y-columns against a single x-column."""
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import os
        
        # Parse figure size
        width, height = map(float, figure_size.split('x'))
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Convert x_column to datetime if it's numeric (Unix timestamp)
        if df[x_column].dtype in ['int64', 'float64']:
            # Try different timestamp units to handle various formats
            try:
                # First try seconds
                df[x_column] = pd.to_datetime(df[x_column], unit='s')
            except (ValueError, pd.errors.OutOfBoundsDatetime):
                try:
                    # Try milliseconds
                    df[x_column] = pd.to_datetime(df[x_column], unit='ms')
                except (ValueError, pd.errors.OutOfBoundsDatetime):
                    try:
                        # Try microseconds
                        df[x_column] = pd.to_datetime(df[x_column], unit='us')
                    except (ValueError, pd.errors.OutOfBoundsDatetime):
                        try:
                            # Try nanoseconds
                            df[x_column] = pd.to_datetime(df[x_column], unit='ns')
                        except (ValueError, pd.errors.OutOfBoundsDatetime):
                            # If all fail, treat as Julian day or create simple numeric range
                            logger.warning(f"Could not convert {x_column} to datetime, using numeric values")
                            pass
        
        x_data = df[x_column]
        
        # Create the plot
        plt.figure(figsize=(width, height))
        
        # Define colors for different series
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Plot each y-column
        for i, y_col in enumerate(y_columns):
            color = colors[i % len(colors)]
            plt.plot(x_data, df[y_col], color=color, linewidth=1.2, alpha=0.8, label=y_col)
        
        # Customize the plot
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis for datetime
        if hasattr(x_data, 'dt'):
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
            plt.xticks(rotation=45, ha='right')
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        # Generate summary statistics
        stats = {}
        for y_col in y_columns:
            stats[y_col] = {
                'mean': float(df[y_col].mean()),
                'std': float(df[y_col].std()),
                'min': float(df[y_col].min()),
                'max': float(df[y_col].max())
            }
        
        result_text = f"✅ Successfully created multi-series plot\n"
        result_text += f"📁 Saved to: {output_path}\n"
        result_text += f"📊 Figure size: {width}x{height} inches\n"
        result_text += f"🖼️ DPI: {dpi}\n"
        result_text += f"📈 Series: {', '.join(y_columns)}\n"
        result_text += f"📅 Time range: {x_data.min()} to {x_data.max()}\n"
        result_text += f"📊 Data points: {len(df)}"
        
        return {
            "status": "success",
            "result": result_text,
            "output_path": output_path,
            "figure_size": f"{width}x{height}",
            "dpi": dpi,
            "series": y_columns,
            "data_points": len(df),
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Error creating multi-series plot: {e}")
        return {"status": "error", "error": str(e)}

async def _create_multi_panel_plot(file_path: str, x_column: str, y_columns: List[str], title: str, output_path: str, figure_size: str, dpi: int, layout: str) -> dict:
    """Create a multi-panel plot with separate subplots for each y-column."""
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import os
        
        # Parse figure size
        width, height = map(float, figure_size.split('x'))
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Convert x_column to datetime if it's numeric (Unix timestamp)
        if df[x_column].dtype in ['int64', 'float64']:
            # Try different timestamp units to handle various formats
            try:
                # First try seconds
                df[x_column] = pd.to_datetime(df[x_column], unit='s')
            except (ValueError, pd.errors.OutOfBoundsDatetime):
                try:
                    # Try milliseconds
                    df[x_column] = pd.to_datetime(df[x_column], unit='ms')
                except (ValueError, pd.errors.OutOfBoundsDatetime):
                    try:
                        # Try microseconds
                        df[x_column] = pd.to_datetime(df[x_column], unit='us')
                    except (ValueError, pd.errors.OutOfBoundsDatetime):
                        try:
                            # Try nanoseconds
                            df[x_column] = pd.to_datetime(df[x_column], unit='ns')
                        except (ValueError, pd.errors.OutOfBoundsDatetime):
                            # If all fail, treat as Julian day or create simple numeric range
                            logger.warning(f"Could not convert {x_column} to datetime, using numeric values")
                            pass
        
        x_data = df[x_column]
        n_panels = len(y_columns)
        
        # Create subplots based on layout
        if layout.lower() == "horizontal":
            fig, axes = plt.subplots(1, n_panels, figsize=(width, height))
            if n_panels == 1:
                axes = [axes]
        else:  # vertical layout (default)
            fig, axes = plt.subplots(n_panels, 1, figsize=(width, height))
            if n_panels == 1:
                axes = [axes]
        
        # Define colors for different components
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green for East, North, Up
        
        # Plot each component in its own subplot
        for i, (y_col, ax) in enumerate(zip(y_columns, axes)):
            color = colors[i % len(colors)]
            
            # Plot the data
            ax.plot(x_data, df[y_col], color=color, linewidth=0.8, alpha=0.8)
            
            # Customize subplot
            ax.set_title(f'{y_col.upper()} Component', fontsize=14, fontweight='bold')
            ax.set_ylabel('Position (meters)', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Format x-axis for datetime
            if hasattr(x_data, 'dt'):
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Only show x-label for bottom subplot (vertical) or all subplots (horizontal)
            if layout.lower() == "vertical" and i < n_panels - 1:
                ax.set_xlabel('')
            else:
                ax.set_xlabel('Time', fontsize=12)
        
        # Add main title
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Make room for suptitle
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        # Save the plot
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        # Generate summary statistics
        stats = {}
        for y_col in y_columns:
            stats[y_col] = {
                'mean': float(df[y_col].mean()),
                'std': float(df[y_col].std()),
                'min': float(df[y_col].min()),
                'max': float(df[y_col].max())
            }
        
        result_text = f"✅ Successfully created multi-panel plot\n"
        result_text += f"📁 Saved to: {output_path}\n"
        result_text += f"📊 Figure size: {width}x{height} inches\n"
        result_text += f"🖼️ DPI: {dpi}\n"
        result_text += f"📈 Panels: {n_panels} ({', '.join(y_columns)})\n"
        result_text += f"📐 Layout: {layout}\n"
        result_text += f"📅 Time range: {x_data.min()} to {x_data.max()}\n"
        result_text += f"📊 Data points: {len(df)}"
        
        return {
            "status": "success",
            "result": result_text,
            "output_path": output_path,
            "figure_size": f"{width}x{height}",
            "dpi": dpi,
            "panels": y_columns,
            "layout": layout,
            "data_points": len(df),
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Error creating multi-panel plot: {e}")
        return {"status": "error", "error": str(e)}

async def _discover_latest_earthscope_datasets() -> dict:
    """Automatically discover the latest EarthScope datasets and extract their GeoJSON and CSV file URLs."""
    try:
        async with NDPClient(NDP_BASE_URL) as client:
            # Step 1: Search for EarthScope datasets
            logger.info("Searching for EarthScope datasets...")
            search_result = await client.search_datasets("EarthScope", "earthscope", limit=20)
            
            if not search_result.get("datasets"):
                return {"status": "error", "error": "No EarthScope datasets found"}
            
            # Step 2: Find the most recent dataset
            datasets = search_result["datasets"]
            latest_dataset = None
            latest_date = None
            
            for dataset in datasets:
                # Look for creation date or modification date
                created = dataset.get("created")
                modified = dataset.get("modified")
                
                if created or modified:
                    date_str = modified if modified else created
                    try:
                        # Parse date (assuming ISO format)
                        from datetime import datetime
                        date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        
                        if latest_date is None or date_obj > latest_date:
                            latest_date = date_obj
                            latest_dataset = dataset
                    except:
                        continue
            
            if not latest_dataset:
                # If no date found, use the first dataset
                latest_dataset = datasets[0]
            
            # Step 3: Get detailed information about the latest dataset
            dataset_id = latest_dataset.get("id")
            if not dataset_id:
                return {"status": "error", "error": "No dataset ID found"}
            
            logger.info(f"Getting details for dataset: {dataset_id}")
            dataset_details = await client.get_dataset_details(dataset_id)
            
            # Step 4: Extract GeoJSON and CSV URLs from resources
            geojson_url = None
            csv_url = None
            
            resources = dataset_details.get("resources", [])
            for resource in resources:
                resource_url = resource.get("url", "")
                resource_format = resource.get("format", "").lower()
                resource_name = resource.get("name", "").lower()
                
                # Look for GeoJSON files
                if (resource_format == "geojson" or 
                    resource_url.endswith(".geojson") or 
                    "geojson" in resource_name):
                    geojson_url = resource_url
                
                # Look for CSV files
                elif (resource_format == "csv" or 
                      resource_url.endswith(".csv") or 
                      "csv" in resource_name):
                    csv_url = resource_url
            
            # Step 5: Prepare result
            result_text = f"🔍 **Latest EarthScope Dataset Found**\n\n"
            result_text += f"**Dataset ID:** {dataset_id}\n"
            result_text += f"**Title:** {latest_dataset.get('title', 'N/A')}\n"
            result_text += f"**Organization:** {latest_dataset.get('organization', {}).get('name', 'N/A')}\n"
            result_text += f"**Created:** {latest_dataset.get('created', 'N/A')}\n"
            result_text += f"**Modified:** {latest_dataset.get('modified', 'N/A')}\n\n"
            
            if geojson_url:
                result_text += f"✅ **GeoJSON URL:** {geojson_url}\n"
            else:
                result_text += f"❌ **GeoJSON URL:** Not found\n"
            
            if csv_url:
                result_text += f"✅ **CSV URL:** {csv_url}\n"
            else:
                result_text += f"❌ **CSV URL:** Not found\n"
            
            return {
                "status": "success",
                "result": result_text,
                "dataset_id": dataset_id,
                "dataset_title": latest_dataset.get("title"),
                "organization": latest_dataset.get("organization", {}).get("name"),
                "created": latest_dataset.get("created"),
                "modified": latest_dataset.get("modified"),
                "geojson_url": geojson_url,
                "csv_url": csv_url,
                "total_datasets_found": len(datasets)
            }
            
    except Exception as e:
        logger.error(f"Error discovering EarthScope datasets: {e}")
        return {"status": "error", "error": str(e)}

def main():
    """
    Main entry point to start the FastMCP server using the specified transport.
    Chooses between stdio and SSE based on MCP_TRANSPORT environment variable.
    """
    transport = os.getenv("MCP_TRANSPORT", "stdio").lower()
    if transport == "sse":
        host = os.getenv("MCP_SSE_HOST", "0.0.0.0")
        port = int(os.getenv("MCP_SSE_PORT", "8000"))
        print(f"Starting SSE on {host}:{port}", file=sys.stderr)
        mcp.run(transport="sse", host=host, port=port)
    else:
        print("Starting stdio transport", file=sys.stderr)
        mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
