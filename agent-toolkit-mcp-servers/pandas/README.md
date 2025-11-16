# Pandas MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/pandas-mcp.svg)](https://pypi.org/project/pandas-mcp/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

**Part of [Agent Toolkit](https://iowarp.github.io/agent-toolkit/) - Gnosis Research Center**

Pandas MCP is a Model Context Protocol server that enables LLMs to perform advanced data analysis and manipulation using the powerful Pandas library, featuring comprehensive statistical analysis, data cleaning and transformation, time series operations, multi-format data I/O (CSV, Excel, JSON, Pa...

## Quick Start

```bash
uvx iowarp-agent-toolkit mcp-server pandas
```

## Documentation

- **Full Documentation**: [Agent Toolkit Website](https://iowarp.github.io/agent-toolkit/)
- **Installation Guide**: See [INSTALLATION.md](../../../CLAUDE.md#setup--installation)
- **Contributing**: See [Contribution Guide](https://github.com/iowarp/agent-toolkit/wiki/Contribution)

---

## Description

Pandas MCP is a Model Context Protocol server that enables LLMs to perform advanced data analysis and manipulation using the powerful Pandas library, featuring comprehensive statistical analysis, data cleaning and transformation, time series operations, multi-format data I/O (CSV, Excel, JSON, Parquet, HDF5), and intelligent data quality assessment for seamless data science workflows.



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
    "pandas-mcp": {
      "command": "uvx",
      "args": ["iowarp-agent-toolkit", "mcp-server", "pandas"]
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
    "pandas-mcp": {
      "type": "stdio",
      "command": "uvx",
      "args": ["iowarp-agent-toolkit", "mcp-server", "pandas"]
    }
  }
}
```

</details>

<details>
<summary><b>Install in Claude Code</b></summary>

Run this command. See [Claude Code MCP docs](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/tutorials#set-up-model-context-protocol-mcp) for more info.

```sh
claude mcp add pandas-mcp -- uvx iowarp-agent-toolkit mcp-server pandas
```

</details>

<details>
<summary><b>Install in Claude Desktop</b></summary>

Add this to your Claude Desktop `claude_desktop_config.json` file. See [Claude Desktop MCP docs](https://modelcontextprotocol.io/quickstart/user) for more info.

```json
{
  "mcpServers": {
    "pandas-mcp": {
      "command": "uvx",
      "args": ["iowarp-agent-toolkit", "mcp-server", "pandas"]
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
git clone https://github.com/iowarp/agent-toolkit.git
uv --directory=$CLONE_DIR/agent-toolkit/agent-toolkit-mcp-servers/pandas run pandas-mcp --help
```

**Windows CMD:**
```cmd
set CLONE_DIR=%cd%
git clone https://github.com/iowarp/agent-toolkit.git
uv --directory=%CLONE_DIR%\agent-toolkit\agent-toolkit-mcp-servers\pandas run pandas-mcp --help
```

**Windows PowerShell:**
```powershell
$env:CLONE_DIR=$PWD
git clone https://github.com/iowarp/agent-toolkit.git
uv --directory=$env:CLONE_DIR\agent-toolkit\agent-toolkit-mcp-servers\pandas run pandas-mcp --help
```

</details>

## Capabilities

### `load_data`
**Description**: Load data from various file formats with comprehensive parsing options.

**Parameters**:
- `file_path` (str): Parameter for file_path
- `file_format` (Any, optional): Parameter for file_format
- `sheet_name` (Any, optional): Parameter for sheet_name
- `encoding` (Any, optional): Parameter for encoding
- `columns` (Any, optional): Parameter for columns
- `nrows` (Any, optional): Parameter for nrows

**Returns**: Dictionary containing: - data: Loaded dataset in structured format - metadata: File information, data types, and loading statistics - data_info: Shape, columns, and data quality metrics - loading_stats: Performance metrics and parsing information

### `save_data`
**Description**: Save data to various file formats with comprehensive export options.

**Parameters**:
- `data` (dict): Parameter for data
- `file_path` (str): Parameter for file_path
- `file_format` (Any, optional): Parameter for file_format
- `index` (bool, optional): Parameter for index (default: True)

**Returns**: Dictionary containing: - save_info: File save details including size and format - compression_stats: Space savings and compression metrics - export_stats: Performance metrics and data integrity checks - file_details: Output file specifications and validation

### `statistical_summary`
**Description**: Generate comprehensive statistical summary with advanced analytics.

**Parameters**:
- `file_path` (str): Parameter for file_path
- `columns` (Any, optional): Parameter for columns
- `include_distributions` (bool, optional): Parameter for include_distributions (default: False)

**Returns**: Dictionary containing: - descriptive_stats: Mean, median, mode, standard deviation, and percentiles - distribution_analysis: Skewness, kurtosis, and normality test results - data_profiling: Data types, missing values, and unique value counts - outlier_detection: Outlier identification and statistical anomalies

### `correlation_analysis`
**Description**: Perform comprehensive correlation analysis with statistical significance testing.

**Parameters**:
- `file_path` (str): Parameter for file_path
- `method` (str, optional): Parameter for method (default: pearson)
- `columns` (Any, optional): Parameter for columns

**Returns**: Dictionary containing: - correlation_matrix: Full correlation matrix with coefficient values - significance_tests: P-values and statistical significance indicators - correlation_insights: Strong correlations and dependency patterns - visualization_data: Data formatted for correlation heatmaps and plots

### `hypothesis_testing`
**Description**: Perform comprehensive statistical hypothesis testing with multiple test types and advanced analysis.

**Parameters**:
- `file_path` (str): Parameter for file_path
- `test_type` (str): Parameter for test_type
- `column1` (str): Parameter for column1
- `column2` (Any, optional): Parameter for column2
- `alpha` (float, optional): Parameter for alpha (default: 0.05)

**Returns**: Dictionary containing: - test_results: Statistical test results including test statistic and p-value - effect_size: Effect size measures and practical significance assessment - confidence_intervals: Confidence intervals for parameters and differences - interpretation: Statistical interpretation and practical conclusions

### `handle_missing_data`
**Description**: Handle missing data with comprehensive strategies and statistical methods.

**Parameters**:
- `file_path` (str): Parameter for file_path
- `strategy` (str, optional): Parameter for strategy (default: detect)
- `method` (Any, optional): Parameter for method
- `columns` (Any, optional): Parameter for columns

**Returns**: Dictionary containing: - missing_data_report: Detailed analysis of missing data patterns - imputation_results: Results of imputation with quality metrics - data_completeness: Before/after comparison of data completeness - strategy_recommendations: Suggested approaches for optimal data handling

### `clean_data`
**Description**: Perform comprehensive data cleaning with advanced quality improvement techniques.

**Parameters**:
- `file_path` (str): Parameter for file_path
- `remove_duplicates` (bool, optional): Parameter for remove_duplicates (default: False)
- `detect_outliers` (bool, optional): Parameter for detect_outliers (default: False)
- `convert_types` (bool, optional): Parameter for convert_types (default: False)

**Returns**: Dictionary containing: - cleaning_report: Detailed summary of cleaning operations performed - data_quality_metrics: Before/after data quality comparison - outlier_analysis: Outlier detection results and recommendations - type_conversion_log: Data type changes and optimization results

### `groupby_operations`
**Description**: Perform sophisticated groupby operations with comprehensive aggregation options.

**Parameters**:
- `file_path` (str): Parameter for file_path
- `group_by` (Any): Parameter for group_by
- `operations` (Any): Parameter for operations
- `filter_condition` (Any, optional): Parameter for filter_condition

**Returns**: Dictionary containing: - grouped_results: Results of groupby operations with aggregated data - group_statistics: Statistics about group sizes and distributions - aggregation_summary: Summary of all aggregation operations performed - performance_metrics: Groupby operation performance and optimization insights

### `merge_datasets`
**Description**: Merge and join datasets with comprehensive integration capabilities.

**Parameters**:
- `left_file` (str): Parameter for left_file
- `right_file` (str): Parameter for right_file
- `join_type` (str, optional): Parameter for join_type (default: inner)
- `left_on` (Any, optional): Parameter for left_on
- `right_on` (Any, optional): Parameter for right_on
- `on` (Any, optional): Parameter for on

**Returns**: Dictionary containing: - merged_data: Results of the merge operation - merge_statistics: Statistics about the merge operation and data overlap - data_quality_report: Quality assessment of the merged dataset - relationship_analysis: Analysis of data relationships and join effectiveness

### `pivot_table`
**Description**: Create sophisticated pivot tables with comprehensive aggregation options.

**Parameters**:
- `file_path` (str): Parameter for file_path
- `index` (Any): Parameter for index
- `columns` (Any, optional): Parameter for columns
- `values` (Any, optional): Parameter for values
- `aggfunc` (str, optional): Parameter for aggfunc (default: mean)

**Returns**: Dictionary containing: - pivot_results: The pivot table with aggregated data - summary_statistics: Statistical summary of the pivot operation - data_insights: Key insights and patterns from the pivot analysis - visualization_data: Data formatted for pivot table visualization

### `time_series_operations`
**Description**: Perform comprehensive time series operations with advanced temporal analysis.

**Parameters**:
- `file_path` (str): Parameter for file_path
- `date_column` (str): Parameter for date_column
- `operation` (str): Parameter for operation
- `window_size` (Any, optional): Parameter for window_size
- `frequency` (Any, optional): Parameter for frequency

**Returns**: Dictionary containing: - time_series_results: Results of the time series operation - temporal_analysis: Trend and seasonality analysis - statistical_summary: Time series statistical properties - forecasting_insights: Patterns and insights for forecasting applications

### `validate_data`
**Description**: Perform comprehensive data validation with advanced constraint checking and quality assessment.

**Parameters**:
- `file_path` (str): Parameter for file_path
- `validation_rules` (Any): Parameter for validation_rules

**Returns**: Dictionary containing: - validation_results: Detailed validation results for each column and rule - data_quality_score: Overall data quality score and assessment - violation_summary: Summary of validation violations and error patterns - recommendations: Suggested actions for data quality improvement

### `filter_data`
**Description**: Perform advanced data filtering with sophisticated boolean indexing and conditional expressions.

**Parameters**:
- `file_path` (str): Parameter for file_path
- `filter_conditions` (Any): Parameter for filter_conditions
- `output_file` (Any, optional): Parameter for output_file

**Returns**: Dictionary containing: - filtered_data: Results of filtering operation with matching records - filter_statistics: Summary of filtering results including row counts - data_quality_report: Quality assessment of filtered dataset - performance_metrics: Filtering operation performance and efficiency

### `optimize_memory`
**Description**: Perform advanced memory optimization for large datasets with intelligent strategies.

**Parameters**:
- `file_path` (str): Parameter for file_path
- `optimize_dtypes` (bool, optional): Parameter for optimize_dtypes (default: True)
- `chunk_size` (Any, optional): Parameter for chunk_size

**Returns**: Dictionary containing: - memory_optimization_results: Before/after memory usage comparison - dtype_optimization_log: Details of data type changes and memory savings - chunking_strategy: Optimal chunking recommendations for large datasets - performance_metrics: Speed and efficiency improvements achieved

### `profile_data`
**Description**: Perform comprehensive data profiling with detailed statistical analysis and quality assessment.

**Parameters**:
- `file_path` (str): Parameter for file_path
- `include_correlations` (bool, optional): Parameter for include_correlations (default: False)
- `sample_size` (Any, optional): Parameter for sample_size

**Returns**: Dictionary containing: - data_profile: Comprehensive dataset overview including shape, types, and statistics - column_analysis: Detailed analysis of each column including distributions - data_quality_metrics: Missing values, duplicates, and data quality indicators - correlation_matrix: Variable correlations (if include_correlations is True)
## Examples

### 1. Data Loading and Profiling
```
I have a large CSV file with sales data that I need to load and get a comprehensive profile including data types, missing values, and basic statistics.
```

**Tools called:**
- `load_data` - Load CSV file with intelligent format detection
- `profile_data` - Get comprehensive data profile and quality metrics
- `statistical_summary` - Generate descriptive statistics and distributions

### 2. Data Cleaning and Quality Assessment
```
My dataset has missing values and outliers that need to be handled. I also want to remove duplicates and validate the data quality.
```

**Tools called:**
- `handle_missing_data` - Impute missing values with appropriate strategies
- `clean_data` - Remove outliers, duplicates, and optimize data types
- `validate_data` - Apply business rules and data quality checks

### 3. Statistical Analysis and Correlation
```
Analyze the relationships between different variables in my dataset and perform hypothesis testing to validate my assumptions.
```

**Tools called:**
- `correlation_analysis` - Calculate correlation matrices with different methods
- `hypothesis_testing` - Perform t-tests, ANOVA, and normality tests
- `statistical_summary` - Generate comprehensive statistical insights

### 4. Data Transformation and Aggregation
```
I need to group my sales data by region and product category, then create pivot tables for cross-analysis and merge with customer data.
```

**Tools called:**
- `groupby_operations` - Group data and perform multiple aggregations
- `pivot_table` - Create pivot tables with multi-level indexing
- `merge_datasets` - Join datasets using different merge strategies

### 5. Time Series Analysis and Filtering
```
Analyze my time series data by resampling to different frequencies, calculating rolling averages, and filtering specific date ranges.
```

**Tools called:**
- `time_series_operations` - Resample, rolling windows, and lag features
- `filter_data` - Apply complex time-based filtering conditions
- `statistical_summary` - Analyze time series patterns and trends

### 6. Data Export and Memory Optimization
```
Optimize memory usage of my large dataset and export the cleaned data to multiple formats for different teams.
```

**Tools called:**
- `optimize_memory` - Reduce memory usage with dtype optimization
- `save_data` - Export to CSV, Excel, Parquet, and JSON formats
- `profile_data` - Verify optimization results and final data quality

