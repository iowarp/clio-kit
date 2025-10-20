# HDF5 MCP Examples

Demonstration scripts for teaching HDF5 MCP v2.0 capabilities.

## Quick Start

### 1. Create Demo Data

```bash
cd iowarp_mcp_servers/hdf5/examples
python create_demo_data.py
```

Creates `demo_data.h5` with realistic climate simulation structure:
- Temperature field (100 timesteps × 50 locations)
- Pressure field (100 timesteps × 50 locations)
- Metadata (timestamps, lat/lon coordinates)
- Analysis results (mean, std)

### 2. Run Demo Script

```bash
python demo_script.py
```

Demonstrates 8 core HDF5 MCP operations:
1. Opening files
2. Exploring structure
3. Navigating to datasets
4. Reading metadata (shape, type, size)
5. Reading attributes
6. Reading data subsets
7. Checking performance features (chunking)
8. Closing files

## Expected Output

```
======================================================================
HDF5 MCP v2.0 - Demonstration
======================================================================

1. Opening HDF5 file...
   Opened file: .../demo_data.h5 in mode 'r'

2. Exploring file structure...
   Root groups: ['analysis', 'metadata', 'simulation']

3. Inspecting dataset...
   Dataset: /simulation/temperature, shape: (100, 50), dtype: float64

4. Dataset metadata:
   Shape: (100, 50)
   Type: float64
   Size: 5000 elements

5. Reading attributes...
   Attributes: ['long_name', 'units', 'valid_max', 'valid_min']
   Units: Kelvin

6. Reading data subset...
   Data (5x5 subset): [[...], [...], ...]

7. Checking file statistics...
   Chunking: (10, 50) with gzip compression

8. Closing file...
   Closed file: .../demo_data.h5

======================================================================
Demo Complete - All HDF5 MCP core features demonstrated
======================================================================
```

## Learning Objectives

Students should understand:
- HDF5 hierarchical structure (groups, datasets)
- Metadata vs data (attributes vs datasets)
- Efficient data access (partial reads, chunking)
- MCP tool patterns (open → operate → close)
- Real scientific data patterns

## Use Cases

**For AI agents:**
- Load demo_data.h5
- Use MCP tools to explore
- Answer questions about the data
- Practice surgical data operations

**For students:**
- See realistic HDF5 structure
- Learn MCP tool invocation
- Understand scientific data workflows
- Practice with safe demo data (not production)

## Files

- `create_demo_data.py` - Generate demo HDF5 file
- `demo_script.py` - Complete demonstration
- `demo_data.h5` - Generated sample file (not in git)
- `README.md` - This file
