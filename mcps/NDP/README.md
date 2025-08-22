# NDP MCP - National Data Platform Integration for LLMs

## Description

NDP MCP is a Model Context Protocol server that enables LLMs to access and analyze data from the National Data Platform, featuring **dynamic dataset discovery**, geospatial analysis, multi-format data downloading (CSV, GeoJSON, PNG, etc.), and advanced visualization capabilities including multi-panel time series plots for EarthScope Consortium data and other scientific datasets.

## 🆕 **Key Features**

- **🔍 Dynamic Discovery**: Automatically discovers available datasets without hardcoded URLs
- **🌍 EarthScope Integration**: Specialized support for EarthScope GNSS data with automatic station detection
- **📊 Advanced Visualization**: Multi-panel time series plots with professional styling
- **⚡ Real-time Verification**: Only returns datasets with files that actually exist
- **🔄 Future-proof**: Automatically adapts to new stations and data years

## 🛠️ Installation

### Requirements

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

<details>
<summary><b>Install in Cursor</b></summary>

Go to: `Settings` -> `Cursor Settings` -> `MCP` -> `Add new global MCP server`

Pasting the following configuration into your Cursor `~/.cursor/mcp.json` file is the recommended approach. You may also install in a specific project by creating `.cursor/mcp.json` in your project folder. See [Cursor MCP docs](https://docs.cursor.com/context/model-context-protocol) for more info.

```json
{
  "mcpServers": {
    "ndp": {
      "command": "uvx",
      "args": ["iowarp-mcps", "ndp"]
    }
  }
}
```

</details>

<details>
<summary><b>Install in VS Code</b></summary>

Add this to your VS Code MCP config file. See [VS Code MCP docs](https://code.visualstudio.com/docs/copilot/chat/mcp-servers) for more info.

```json
"mcp": {
  "servers": {
    "ndp": {
      "type": "stdio",
      "command": "uvx",
      "args": ["iowarp-mcps", "ndp"]
    }
  }
}
```

</details>

<details>
<summary><b>Install in Claude Code</b></summary>

Run this command. See [Claude Code MCP docs](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/tutorials#set-up-model-context-protocol-mcp) for more info.

```sh
claude mcp add ndp -- uvx iowarp-mcps ndp
```

</details>

<details>
<summary><b>Install in Claude Desktop</b></summary>

Add this to your Claude Desktop `claude_desktop_config.json` file. See [Claude Desktop MCP docs](https://modelcontextprotocol.io/quickstart/user) for more info.

```json
{
  "mcpServers": {
    "ndp": {
      "command": "uvx",
      "args": ["iowarp-mcps", "ndp"]
    }
  }
}
```

</details>

<details>
<summary><b>Manual Setup</b></summary>

**Linux/macOS:**
```bash
CLONE_DIR=$(pwd)
git clone https://github.com/iowarp/iowarp-mcps.git
uv --directory=$CLONE_DIR/iowarp-mcps/mcps/NDP run ndp-mcp-server --help
```

**Windows CMD:**
```cmd
set CLONE_DIR=%cd%
git clone https://github.com/iowarp/iowarp-mcps.git
uv --directory=%CLONE_DIR%\iowarp-mcps\mcps\NDP run ndp-mcp-server --help
```

**Windows PowerShell:**
```powershell
$env:CLONE_DIR=$PWD
git clone https://github.com/iowarp/iowarp-mcps.git
uv --directory=$env:CLONE_DIR\iowarp-mcps\mcps\NDP run ndp-mcp-server --help
```

</details>

## Capabilities

### `discover_latest_earthscope_datasets` ⭐ **NEW**
**Description**: Automatically discover the latest EarthScope datasets and extract their GeoJSON and CSV file URLs without any hardcoded URLs.

**Parameters**: None

**Returns**: Dictionary containing:
- result: Formatted discovery results with dataset information
- dataset_id: ID of the discovered dataset
- dataset_title: Title of the dataset
- organization: Organization name
- created: Creation date
- modified: Modification date
- geojson_url: Discovered GeoJSON file URL
- csv_url: Discovered CSV file URL
- total_datasets_found: Number of datasets discovered

### `list_organizations`
**Description**: List all available organizations from the National Data Platform API.

**Parameters**: None

**Returns**: Dictionary containing:
- result: Formatted list of organizations with details
- organizations: Raw organization data from API
- status: Success/error status

### `search_datasets`
**Description**: Search for datasets across the NDP catalog with intelligent chunking for large result sets.

**Parameters**:
- `query` (str): Search query string
- `organization` (str, optional): Organization filter
- `limit` (int, optional): Maximum number of results (default: 10)

**Returns**: Dictionary containing:
- result: Formatted search results with dataset details
- results: Raw dataset data from API
- chunk_id: Current chunk identifier (if chunking applied)
- total_chunks: Total number of chunks available

### `get_dataset_details`
**Description**: Retrieve complete metadata for a specific dataset including resources, tags, and spatial information.

**Parameters**:
- `dataset_id` (str): Dataset ID to retrieve

**Returns**: Dictionary containing:
- result: Formatted dataset details with comprehensive metadata
- dataset: Raw dataset information from API
- status: Success/error status

### `download_dataset_resources`
**Description**: Download dataset files from dataset resources with support for multiple file formats.

**Parameters**:
- `dataset_id` (str): Dataset ID to download resources from
- `resource_types` (List[str], optional): Array of resource types to download (defaults to all)

**Returns**: Dictionary containing:
- result: Download summary with file details
- downloaded_files: List of downloaded file information
- status: Success/error status

### `download_file_from_url`
**Description**: Download a file directly from a URL and save it to a specified location with comprehensive error handling.

**Parameters**:
- `url` (str): The URL to download the file from
- `output_path` (str): The local path where the file should be saved

**Returns**: Dictionary containing:
- result: Download summary with file details
- file_path: Path where file was saved
- file_size: Size of downloaded file in bytes
- url: Original download URL

### `analyze_geospatial_data`
**Description**: Analyze geospatial data from a dataset with comprehensive insights and recommendations.

**Parameters**:
- `dataset_id` (str): Dataset ID to analyze

**Returns**: Dictionary containing:
- result: Geospatial analysis summary
- geospatial_resources: List of geospatial resources found
- status: Success/error status

### `create_multi_series_plot`
**Description**: Create a multi-series line plot with multiple y-columns against a single x-column, perfect for time series data like GNSS measurements.

**Parameters**:
- `file_path` (str): Path to CSV file
- `x_column` (str): Column name for x-axis data
- `y_columns` (List[str]): List of column names for y-axis data
- `title` (str, optional): Plot title (default: "Multi-Series Line Plot")
- `output_path` (str, optional): Output image file path (default: "multi_series_plot.png")
- `figure_size` (str, optional): Figure size in "widthxheight" format (default: "15x12")
- `dpi` (int, optional): Image quality (default: 300)

**Returns**: Dictionary containing:
- result: Plot creation summary
- output_path: Path where plot was saved
- figure_size: Actual figure dimensions
- dpi: Image quality setting
- series: List of plotted series
- data_points: Number of data points processed
- statistics: Statistical summary for each series

### `create_multi_panel_plot`
**Description**: Create a multi-panel plot with separate subplots for each y-column, perfect for GNSS time series with East, North, Up components.

**Parameters**:
- `file_path` (str): Path to CSV file
- `x_column` (str): Column name for x-axis data
- `y_columns` (List[str]): List of column names for y-axis data
- `title` (str, optional): Plot title (default: "Multi-Panel Time Series Plot")
- `output_path` (str, optional): Output image file path (default: "multi_panel_plot.png")
- `figure_size` (str, optional): Figure size in "widthxheight" format (default: "15x12")
- `dpi` (int, optional): Image quality (default: 300)
- `layout` (str, optional): Layout type - "vertical" or "horizontal" (default: "vertical")

**Returns**: Dictionary containing:
- result: Plot creation summary
- output_path: Path where plot was saved
- figure_size: Actual figure dimensions
- dpi: Image quality setting
- panels: List of panel names
- layout: Layout type used
- data_points: Number of data points processed
- statistics: Statistical summary for each panel

## Examples

### 1. **Dynamic EarthScope Dataset Discovery** ⭐ **NEW**
```
Discover the latest EarthScope datasets automatically without any hardcoded URLs.
```

**Tools called:**
- `discover_latest_earthscope_datasets` - Automatically find and return the latest EarthScope dataset URLs

### 2. **Complete EarthScope Workflow** ⭐ **UPDATED**
```
Execute a complete EarthScope GNSS data analysis workflow: discover datasets, download data, analyze statistics, and generate professional visualizations.
```

**Tools called:**
- `discover_latest_earthscope_datasets` - Find latest EarthScope datasets dynamically
- `download_file_from_url` - Download discovered CSV and GeoJSON files
- `create_multi_panel_plot` - Generate 3-panel time series visualization
- `analyze_geospatial_data` - Provide comprehensive analysis insights

### 3. **Dataset Discovery and Search**
```
I need to find EarthScope Consortium datasets related to GNSS measurements and seismic data.
```

**Tools called:**
- `list_organizations` - Get available organizations including EarthScope Consortium
- `search_datasets` - Search for GNSS and seismic datasets with intelligent chunking
- `get_dataset_details` - Retrieve comprehensive metadata for specific datasets

### 4. **Data Download and File Management**
```
Download the discovered EarthScope dataset files including the CSV data and GeoJSON metadata.
```

**Tools called:**
- `discover_latest_earthscope_datasets` - Get dataset information and resource URLs
- `download_file_from_url` - Download CSV file directly from discovered URL
- `download_file_from_url` - Download GeoJSON metadata file from discovered URL
- `download_dataset_resources` - Alternative method to download all resources

### 5. **Geospatial Analysis and Visualization**
```
Analyze the geospatial properties of the downloaded dataset and create a 3-panel time series plot showing East, North, and Up components.
```

**Tools called:**
- `analyze_geospatial_data` - Analyze spatial properties and provide insights
- `create_multi_panel_plot` - Create 3-panel visualization with separate subplots
- `create_multi_series_plot` - Alternative single-plot visualization with all components

### 6. **Multi-Format Data Processing**
```
Download and process various data formats from NDP including CSV time series, GeoJSON spatial data, and PNG visualizations.
```

**Tools called:**
- `download_dataset_resources` - Download multiple file formats
- `create_multi_series_plot` - Create time series visualizations
- `analyze_geospatial_data` - Process spatial metadata
- `get_dataset_details` - Access comprehensive dataset information

### 7. **Advanced Visualization and Analysis**
```
Create professional scientific visualizations with custom styling, multiple panels, and comprehensive statistical analysis.
```

**Tools called:**
- `create_multi_panel_plot` - Generate publication-quality multi-panel plots
- `create_multi_series_plot` - Create single-plot visualizations with multiple series
- `analyze_geospatial_data` - Provide spatial analysis and recommendations
- `get_dataset_details` - Access metadata for visualization context

## 🌍 EarthScope Integration

The NDP MCP server provides specialized support for EarthScope Consortium data with **dynamic discovery**:

### **Dynamic Dataset Discovery** ⭐ **NEW**
- **Automatic station detection** across multiple EarthScope stations
- **Real-time file verification** - only returns datasets with existing files
- **Multi-year support** - discovers data from 2020-2024 automatically
- **No hardcoded URLs** - completely dynamic and future-proof

### GNSS Time Series Analysis
- **Multi-panel plots** for East, North, Up components
- **Automatic time conversion** from various timestamp formats (seconds, milliseconds, microseconds, nanoseconds)
- **Professional styling** with scientific standards
- **Statistical analysis** with comprehensive metrics

### Geospatial Data Processing
- **GeoJSON metadata** analysis and visualization
- **Spatial coordinate systems** and projections
- **Bounding box calculations** and spatial extent analysis
- **Attribute data** integration with spatial features

### Data Quality Assessment
- **Missing data detection** and reporting
- **Statistical validation** of time series data
- **Spatial data integrity** checks
- **Format compatibility** verification

## 🔧 Configuration

### Environment Variables

```bash
# NDP API Configuration
export NDP_BASE_URL="https://ds2.datacollaboratory.org"
export NDP_DEFAULT_SERVER="global"
export NDP_CHUNK_THRESHOLD="50"

# MCP Transport Configuration
export MCP_TRANSPORT="stdio"  # or "sse"
export MCP_SSE_HOST="0.0.0.0"
export MCP_SSE_PORT="8000"
```

### Custom API Endpoints

```json
{
  "mcpServers": {
    "ndp": {
      "command": "uvx",
      "args": ["iowarp-mcps", "ndp"],
      "env": {
        "NDP_BASE_URL": "http://your-custom-ndp-server:8003",
        "NDP_DEFAULT_SERVER": "your-server-name"
      }
    }
  }
}
```

## 📊 Visualization Features

### Multi-Panel Plots
- **Vertical layout** for time series data
- **Horizontal layout** for comparative analysis
- **Automatic subplot creation** for each data series
- **Professional styling** with scientific standards

### Time Series Support
- **Multiple timestamp formats** support (seconds, milliseconds, microseconds, nanoseconds)
- **Automatic datetime formatting** for x-axis
- **Robust error handling** for timestamp conversion
- **Temporal analysis** capabilities

### Customization Options
- **Figure size** control (width x height)
- **DPI settings** for high-quality output
- **Color schemes** optimized for scientific data
- **Grid lines** and axis formatting

## 🛡️ Error Handling

The server includes comprehensive error handling:

- **Network failures** with automatic retry logic
- **API rate limiting** with intelligent throttling
- **File download errors** with detailed error messages
- **Data validation** with format checking
- **Memory management** for large datasets
- **Timestamp conversion errors** with multiple fallback strategies

## 🔒 Security Features

- **URL validation** before downloads
- **File size limits** for downloads
- **Input sanitization** for all parameters
- **Safe file handling** with proper permissions
- **Error message filtering** for sensitive information

## 📁 Project Structure

```
iowarp-mcps/mcps/NDP/
├── src/
│   ├── __init__.py                 # Package initialization
│   └── ndp_mcp_server.py          # Main MCP server implementation
├── test/
│   ├── prompt.md                   # Dynamic discovery workflow example
│   └── requirements.txt           # Test dependencies
├── evaluation/
│   ├── time_series_plot.png       # Example visualization output
│   └── time_over_east_north_up_plot.png # Multi-panel example
├── pyproject.toml                 # UVX packaging configuration
├── requirements.txt               # Dependencies
└── README.md                      # This documentation
```

## 🚀 Quick Start Examples

### **Dynamic EarthScope Discovery** ⭐ **NEW**
```python
# Discover latest EarthScope datasets automatically
result = await client.call_tool("discover_latest_earthscope_datasets", {})
print(f"Found dataset: {result['dataset_title']}")
print(f"GeoJSON URL: {result['geojson_url']}")
print(f"CSV URL: {result['csv_url']}")
```

### **Complete Dynamic Workflow** ⭐ **UPDATED**
```python
# 1. Discover datasets dynamically
discovery = await client.call_tool("discover_latest_earthscope_datasets", {})

# 2. Download discovered data
download_csv = await client.call_tool("download_file_from_url", {
    "url": discovery["csv_url"],
    "output_path": "earthscope_output/data.csv"
})

download_geojson = await client.call_tool("download_file_from_url", {
    "url": discovery["geojson_url"],
    "output_path": "earthscope_output/metadata.geojson"
})

# 3. Create 3-panel visualization
plot_result = await client.call_tool("create_multi_panel_plot", {
    "file_path": "earthscope_output/data.csv",
    "x_column": "time",
    "y_columns": ["east", "north", "up"],
    "title": f"{discovery['dataset_title']} - GNSS Time Series",
    "output_path": "earthscope_output/visualization.png",
    "figure_size": "15x12",
    "dpi": 300,
    "layout": "vertical"
})
```

### Basic Dataset Search
```python
# Search for EarthScope datasets
result = await client.call_tool("search_datasets", {
    "query": "GNSS",
    "organization": "earthscope",
    "limit": 10
})
```

### Download and Visualize
```python
# Download GNSS data (using discovered URLs)
download_result = await client.call_tool("download_file_from_url", {
    "url": "discovered_csv_url",
    "output_path": "earthscope_output/data.csv"
})

# Create 3-panel visualization
plot_result = await client.call_tool("create_multi_panel_plot", {
    "file_path": "earthscope_output/data.csv",
    "x_column": "time",
    "y_columns": ["east", "north", "up"],
    "title": "GNSS Time Series Analysis",
    "output_path": "earthscope_output/visualization.png",
    "figure_size": "15x12",
    "dpi": 300,
    "layout": "vertical"
})
```

## 🤝 Integration with Other MCP Servers

### Pandas MCP Integration
```python
# Discover and download data with NDP MCP
discovery = await ndp_client.call_tool("discover_latest_earthscope_datasets", {})
ndp_result = await ndp_client.call_tool("download_file_from_url", {
    "url": discovery["csv_url"],
    "output_path": "data.csv"
})

# Analyze with Pandas MCP
pandas_result = await pandas_client.call_tool("statistical_summary", {
    "file_path": "data.csv",
    "columns": ["east", "north", "up"]
})
```

### Plot MCP Integration
```python
# Discover and download data with NDP MCP
discovery = await ndp_client.call_tool("discover_latest_earthscope_datasets", {})
ndp_result = await ndp_client.call_tool("download_file_from_url", {
    "url": discovery["csv_url"],
    "output_path": "data.csv"
})

# Create basic plot with Plot MCP
plot_result = await plot_client.call_tool("line_plot", {
    "file_path": "data.csv",
    "x_column": "time",
    "y_column": "east",
    "title": "East Component",
    "output_path": "east_plot.png"
})
```

## 📈 Performance Optimization

### Large Dataset Handling
- **Intelligent chunking** for search results
- **Streaming downloads** for large files
- **Memory-efficient processing** for visualization
- **Background processing** for long-running operations

### Caching Strategy
- **Search result caching** with configurable thresholds
- **API response caching** to reduce redundant requests
- **File download caching** for frequently accessed data
- **Visualization caching** for repeated plot generation

## 🔄 Advanced Workflows

### **Automated Dynamic Data Pipeline** ⭐ **UPDATED**
1. **Dynamic Discovery** - Automatically find available datasets
2. **Data Download** - Retrieve files from discovered URLs
3. **Quality Assessment** - Validate data integrity
4. **Visualization Generation** - Create publication-ready plots
5. **Analysis Reporting** - Generate comprehensive reports

### Multi-Dataset Analysis
1. **Batch Discovery** - Find multiple related datasets dynamically
2. **Parallel Download** - Download multiple files simultaneously
3. **Comparative Analysis** - Analyze relationships between datasets
4. **Unified Visualization** - Create combined visualizations
5. **Cross-Dataset Insights** - Generate comparative reports

## 📚 Additional Resources

### Documentation
- [National Data Platform API Documentation](https://ds2.datacollaboratory.org)
- [EarthScope Consortium Data](https://www.earthscope.org/data)
- [GNSS Data Analysis Guide](https://www.earthscope.org/gnss)

### Related Tools
- [Pandas MCP](https://github.com/iowarp/iowarp-mcps/tree/main/mcps/Pandas) - Data analysis and manipulation
- [Plot MCP](https://github.com/iowarp/iowarp-mcps/tree/main/mcps/Plot) - Basic plotting capabilities
- [Jarvis MCP](https://github.com/iowarp/iowarp-mcps/tree/main/mcps/Jarvis) - System management and automation

### Community
- [GitHub Issues](https://github.com/iowarp/iowarp-mcps/issues) - Report bugs and request features
- [Discussions](https://github.com/iowarp/iowarp-mcps/discussions) - Community support and ideas
- [Contributing Guide](https://github.com/iowarp/iowarp-mcps/blob/main/CONTRIBUTING.md) - How to contribute

## ✅ Status

**NDP MCP Server Status:**
- ✅ **MCP Protocol**: Full MCP stdio protocol support
- ✅ **Dynamic Discovery**: Automatic dataset discovery without hardcoded URLs ⭐ **NEW**
- ✅ **Dataset Search**: Comprehensive search and metadata access
- ✅ **Data Download**: Multi-format file downloading capabilities
- ✅ **Geospatial Analysis**: Advanced spatial data processing
- ✅ **Visualization**: Multi-panel and multi-series plotting with robust timestamp handling
- ✅ **EarthScope Integration**: Specialized GNSS data support with automatic station detection
- ✅ **Error Handling**: Robust error handling and recovery
- ✅ **Documentation**: Comprehensive guides and examples
- ✅ **Production Ready**: Security and performance optimized

**All capabilities implemented and fully functional!** 🎯

## 🆕 **What's New**

### Version 2.0 - Dynamic Discovery
- **🔍 No More Hardcoded URLs**: Completely dynamic dataset discovery
- **🌍 Automatic EarthScope Detection**: Discovers available stations and years automatically
- **⚡ Real-time Verification**: Only returns datasets with existing files
- **🔄 Future-proof**: Automatically adapts to new data without code changes
- **📊 Enhanced Timestamp Handling**: Supports multiple timestamp formats with robust error handling

