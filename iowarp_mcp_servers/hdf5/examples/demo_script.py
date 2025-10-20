#!/usr/bin/env python3
"""
HDF5 MCP Demo - Educational Preview

Demonstrates what HDF5 MCP does using direct h5py calls.
Shows the operations an AI agent would perform via MCP tools.

Run: uv run python examples/demo_script.py
"""

import h5py
import numpy as np
from pathlib import Path


def demo():
    """Demonstrate HDF5 operations that MCP enables."""
    demo_file = Path(__file__).parent / "demo_data.h5"

    if not demo_file.exists():
        print("ERROR: demo_data.h5 not found")
        print("Run: uv run python examples/create_demo_data.py")
        return 1

    print("=" * 70)
    print("HDF5 MCP v2.0 - Operation Demonstration")
    print("What an AI agent does via MCP tools")
    print("=" * 70)
    print()

    with h5py.File(demo_file, 'r') as f:
        # 1. Explore structure (via list_keys, visit tools)
        print("1. File Structure Exploration (MCP: list_keys, visit)")
        print(f"   Root groups: {list(f.keys())}")
        print(f"   Simulation datasets: {list(f['simulation'].keys())}")
        print()

        # 2. Dataset metadata (via get_shape, get_dtype, get_size tools)
        print("2. Dataset Metadata (MCP: get_shape, get_dtype, get_size)")
        temp = f['simulation/temperature']
        print(f"   Temperature shape: {temp.shape}")
        print(f"   Temperature type: {temp.dtype}")
        print(f"   Temperature size: {temp.size} elements")
        print()

        # 3. Read attributes (via read_attribute, list_attributes tools)
        print("3. Metadata Access (MCP: read_attribute, list_attributes)")
        print(f"   File attributes: {dict(f.attrs)}")
        print(f"   Temperature attributes: {dict(temp.attrs)}")
        print(f"   Units: {temp.attrs['units']}")
        print()

        # 4. Partial read (via read_partial_dataset tool)
        print("4. Efficient Data Reading (MCP: read_partial_dataset)")
        subset = temp[0:5, 0:5]
        print(f"   5x5 subset from (100,50) dataset:")
        print(f"   Mean: {subset.mean():.2f} K")
        print(f"   Std: {subset.std():.2f} K")
        print()

        # 5. Performance features (via get_chunks tool)
        print("5. Performance Inspection (MCP: get_chunks)")
        print(f"   Chunking: {temp.chunks}")
        print(f"   Compression: {temp.compression}")
        print(f"   → Enables efficient partial reads & streaming")
        print()

        # 6. Multiple datasets (via batch_read tool)
        print("6. Multi-Dataset Operations (MCP: hdf5_batch_read)")
        pressure = f['simulation/pressure']
        print(f"   Temperature mean: {temp[:].mean():.2f} K")
        print(f"   Pressure mean: {pressure[:].mean():.2f} Pa")
        print(f"   → Parallel batch reading available")
        print()

        # 7. Analysis results (via get_by_path, read_full_dataset)
        print("7. Analysis Data Access (MCP: get_by_path, read_full_dataset)")
        mean_temp = f['analysis/mean_temp']
        print(f"   Temporal mean shape: {mean_temp.shape}")
        print(f"   Min mean temp: {mean_temp[:].min():.2f} K")
        print(f"   Max mean temp: {mean_temp[:].max():.2f} K")
        print()

    print("=" * 70)
    print("Demo Complete")
    print()
    print("What this shows:")
    print("  ✓ Hierarchical data navigation")
    print("  ✓ Metadata and attribute access")
    print("  ✓ Efficient partial reading")
    print("  ✓ Performance features (chunking, compression)")
    print("  ✓ Multi-dataset workflows")
    print()
    print("Via MCP, AI agents perform these operations with natural language:")
    print('  "What datasets are in this file?"')
    print('  "Read temperature data for timestep 10"')
    print('  "What are the units for pressure?"')
    print('  "Compare temperature and pressure fields"')
    print("=" * 70)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(demo())
