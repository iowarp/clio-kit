# EarthScope GNSS Data Analysis Workflow - Gemini CLI Prompt

## Objective
Execute a complete EarthScope GNSS data analysis workflow using Gemini CLI with the NDP MCP server and Pandas MCP to:
1. **Download data** using NDP MCP from specific URLs
2. **Parse and analyze** using Pandas MCP 
3. **Generate visualization** that matches the reference PNG

The goal is to replicate the EarthScope workflow by downloading, processing, and visualizing GNSS station data.

## Data Sources
- **GeoJSON URL**: https://ds2.datacollaboratory.org/Earthscope_api/geojson/rhcl.geojson
- **CSV URL**: https://ds2.datacollaboratory.org/Earthscope_api/RHCL.CI.LY_.20.csv  
- **Reference PNG**: https://ds2.datacollaboratory.org/Earthscope_api/RHCL.CI.LY_.20.png

## MCP Tools Required
- **NDP MCP**: For downloading the data files from the URLs above
  - Tools: `download_file_from_url` (for downloading from direct URLs), `create_multi_panel_plot` (for creating 3-panel time series plots)
- **Pandas MCP**: For parsing CSV data and performing analysis
  - Tools: `load_data`, `statistical_summary`, `correlation_analysis`

## IMPORTANT: File Generation Restrictions
- ❌ DO NOT create any Python scripts (.py files)
- ❌ DO NOT create any temporary files beyond the final outputs
- ❌ DO NOT create any intermediate processing files
- ✅ Use NDP MCP for downloading data and creating visualizations
- ✅ Use Pandas MCP for data parsing and analysis
- ✅ Generate final PNG visualization using NDP MCP tools
- ✅ Save all outputs to earthscope_output/ directory

## Workflow Overview
This workflow will perform the following steps:
1. **Setup**: Create the earthscope_output directory
2. **Download Data**: Use NDP MCP to download CSV and GeoJSON from specific URLs
3. **Parse CSV Data**: Use Pandas MCP to load and analyze the GNSS time series data  
4. **Parse GeoJSON**: Load and analyze the geospatial metadata
5. **Generate Visualization**: Create the 3-panel time series plot using NDP MCP
6. **Verify Output**: Ensure all files match expected structure and content

## Detailed Instructions

### Step 1: Setup
```
1. Create the output directory:
   mkdir -p earthscope_output

2. Verify the target URLs are accessible:
   - GeoJSON: https://ds2.datacollaboratory.org/Earthscope_api/geojson/rhcl.geojson
   - CSV: https://ds2.datacollaboratory.org/Earthscope_api/RHCL.CI.LY_.20.csv
   - Reference PNG: https://ds2.datacollaboratory.org/Earthscope_api/RHCL.CI.LY_.20.png
```

### Step 2: Download Data Using NDP MCP
```
1. Use NDP MCP `download_file_from_url` tool to download the GeoJSON file:
   - URL: https://ds2.datacollaboratory.org/Earthscope_api/geojson/rhcl.geojson
   - Save as: earthscope_output/rhcl.geojson
   - Expected size: ~811 bytes

2. Use NDP MCP `download_file_from_url` tool to download the CSV file:
   - URL: https://ds2.datacollaboratory.org/Earthscope_api/RHCL.CI.LY_.20.csv
   - Save as: earthscope_output/RHCL.CI.LY_.20.csv
   - Expected size: ~49MB (816,588 rows, 8 columns)

3. Verify both downloads completed successfully:
   - Check file sizes match expectations
   - Confirm files are readable and not corrupted
   - Display download summary
```

### Step 3: Parse CSV Data Using Pandas MCP
```
1. Use Pandas MCP `load_data` tool to load the CSV file:
   - Read: earthscope_output/RHCL.CI.LY_.20.csv
   - Expected columns: ['time', 'east', 'north', 'up', 'sigEE', 'sigNN', 'sigUU', 'qChannel']
   - Expected shape: (816588, 8)
   - Data types: time (int64), east/north/up (float64), sigEE/sigNN/sigUU (float64), qChannel (int64)

2. Use Pandas MCP `statistical_summary` tool to analyze the data:
   - Calculate basic statistics for east, north, up components
   - Calculate standard deviations: East (~0.18m), North (~0.08m), Up (~0.22m)
   - Verify data completeness: ~94.5% (816,588 / 864,000 expected points)
   - Check for missing values (should be none)
   - Note: Time column remains as int64 (nanoseconds since epoch) for MCP compatibility

3. Prepare data for visualization:
   - Extract time series for east, north, up components
   - Note: Time conversion to datetime will be handled during visualization step
   - Calculate temporal coverage (2024-12-03 to 2024-12-12) from time range
```

### Step 4: Generate PNG Visualization Using NDP MCP
```
1. Use NDP MCP `create_multi_panel_plot` tool to create the 3-panel time series plot:
   - File: earthscope_output/RHCL.CI.LY
   _.20.csv
   - X-column: 'time' (will be automatically converted from nanoseconds to datetime)
   - Y-columns: ['east', 'north', 'up']
   - Title: "RHCL.CI.LY_.20 - GNSS Time Series (1Hz, 816,588 points, ~94.5% complete)"
   - Output: earthscope_output/RHCL.CI.LY_.20.png
   - Figure size: "15x12"
   - DPI: 300
   - Layout: "vertical" (3 panels stacked vertically)

2. The tool will automatically:
   - Convert time from nanoseconds to proper datetime format
   - Create 3 separate panels (East, North, Up) stacked vertically
   - Use appropriate colors for each component (Blue, Orange, Green)
   - Format the x-axis with proper date labels
   - Add grid lines and professional styling
   - Generate high-quality PNG output

3. Expected result:
   - 3-panel plot with separate subplots for East, North, and Up components
   - Vertical layout with proper spacing
   - Proper time axis formatting (2024-12-03 to 2024-12-12)
   - Y-axis in meters for each component
   - Professional styling matching scientific standards
   - File saved as: earthscope_output/RHCL.CI.LY_.20.png
```

### Step 5: Parse GeoJSON Metadata
```
1. Load the GeoJSON file:
   - Read: earthscope_output/rhcl.geojson
   - Expected: 2 features (Point and Polygon geometries)
   - CRS: EPSG:4326 (WGS84)
   - Contains station location and boundary information

2. Extract geospatial information:
   - Station coordinates
   - Bounding box
   - Geometry types and properties
   - Coordinate reference system details
```

### Step 6: Final Output Verification
```
1. Verify all files are created in earthscope_output/:
   - RHCL.CI.LY_.20.csv (~49MB) - Downloaded GNSS time series data  
   - rhcl.geojson (~811 bytes) - Downloaded geospatial metadata
   - RHCL.CI.LY_.20.png (~1-2MB) - Generated visualization matching reference

2. Quality checks:
   - CSV: 816,588 rows, 8 columns, no missing values
   - GeoJSON: 2 features, EPSG:4326 CRS
   - PNG: 3-panel plot matching reference exactly

3. Display file sizes, creation timestamps, and verification summary
```

## Expected Results

### File Structure  
```
earthscope_output/
├── RHCL.CI.LY_.20.csv          (~49MB, downloaded GNSS time series)
├── rhcl.geojson               (~811 bytes, downloaded geospatial metadata)  
└── RHCL.CI.LY_.20.png         (~1-2MB, generated visualization matching reference)
```

### Key Metrics
- **Data Records**: 816,588 time series points
- **Data Completeness**: ~94.5%
- **Standard Deviations**: East (~0.18m), North (~0.08m), Up (~0.22m)
- **Geospatial Features**: 2 (Point and Polygon)
- **Coordinate System**: WGS84 (EPSG:4326)
- **Temporal Coverage**: 2024-12-03 to 2024-12-12
- **Sampling Rate**: 1Hz

### Visualization Features
- Three-panel time series plot
- Professional styling with seaborn
- Clear component separation (East, North, Up)
- Comprehensive metadata in title
- High-resolution output (300 DPI)

## Technical Requirements

### Required MCP Tools
1. **NDP MCP** - For downloading files from URLs:
   - Download CSV from: https://ds2.datacollaboratory.org/Earthscope_api/RHCL.CI.LY_.20.csv
   - Download GeoJSON from: https://ds2.datacollaboratory.org/Earthscope_api/geojson/rhcl.geojson

2. **Pandas MCP** - For data parsing and analysis:
   - Load and analyze CSV data
   - Convert time columns to datetime
   - Calculate statistics and prepare data for visualization

3. **Visualization capability** - For PNG generation:
   - Create 3-panel time series plot
   - Match reference PNG: https://ds2.datacollaboratory.org/Earthscope_api/RHCL.CI.LY_.20.png

### IMPORTANT: Use Only MCP Tools
- ✅ Use NDP MCP for downloading data
- ✅ Use Pandas MCP for data parsing and analysis
- ✅ Use available visualization tools for PNG generation
- ❌ DO NOT create Python scripts or use libraries directly
- ❌ DO NOT create intermediate files beyond the 3 final outputs

### Output Quality Standards
- All files must be properly formatted and readable
- Visualization must be publication-quality
- Documentation must be comprehensive and professional
- Data integrity must be maintained throughout the process
- NO intermediate files or scripts should be created

## Success Criteria
1. ✅ All three output files created successfully in earthscope_output/
2. ✅ CSV and GeoJSON downloaded correctly using NDP MCP
3. ✅ CSV data parsed and analyzed correctly using Pandas MCP  
4. ✅ PNG visualization generated matching the reference exactly
5. ✅ File sizes and content match expected specifications
6. ✅ All statistics and metrics calculated accurately
7. ✅ NO intermediate files, scripts, or temporary files created
8. ✅ All processing done through MCP tools only

## Execution Notes
- Execute workflow step-by-step with clear progress indicators
- Use NDP MCP for downloading both data files from the specified URLs
- Use Pandas MCP for all CSV parsing and analysis operations
- Generate PNG that visually matches the reference as closely as possible
- Verify each step before proceeding to the next
- Display file sizes, timestamps, and quality metrics for validation
- CRITICAL: Only create the final 3 files in earthscope_output/ directory
- CRITICAL: Use specified MCP tools, no direct library usage or script creation
