#!/usr/bin/env python3
"""
Create demo HDF5 file for teaching/testing.

Generates realistic scientific data structure.
"""

import h5py
import numpy as np
from pathlib import Path


def create_demo_data():
    """Create demo HDF5 file with realistic structure."""
    output_file = Path(__file__).parent / "demo_data.h5"

    print(f"Creating demo HDF5 file: {output_file}")

    with h5py.File(output_file, "w") as f:
        # Root metadata
        f.attrs["experiment"] = "Climate Simulation Demo"
        f.attrs["institution"] = "Gnosis Research Center"
        f.attrs["date"] = "2025-01-15"
        f.attrs["version"] = "2.0"

        # Create simulation group
        sim = f.create_group("simulation")
        sim.attrs["description"] = "Climate model output"
        sim.attrs["timesteps"] = 100

        # Temperature data (100 timesteps x 50 locations)
        temp_data = 273.15 + 20 * np.random.randn(100, 50)  # ~20°C variation
        temp_ds = sim.create_dataset(
            "temperature", data=temp_data, chunks=(10, 50), compression="gzip"
        )
        temp_ds.attrs["units"] = "Kelvin"
        temp_ds.attrs["long_name"] = "Surface Temperature"
        temp_ds.attrs["valid_min"] = 200.0
        temp_ds.attrs["valid_max"] = 350.0

        # Pressure data
        pressure_data = 101325 + 5000 * np.random.randn(100, 50)  # ~1 atm
        pressure_ds = sim.create_dataset(
            "pressure", data=pressure_data, chunks=(10, 50), compression="gzip"
        )
        pressure_ds.attrs["units"] = "Pascal"
        pressure_ds.attrs["long_name"] = "Sea Level Pressure"

        # Metadata group
        meta = f.create_group("metadata")
        meta.create_dataset("timestamps", data=np.arange(100))
        meta.create_dataset("latitude", data=np.linspace(-90, 90, 50))
        meta.create_dataset("longitude", data=np.linspace(-180, 180, 50))

        # Analysis group
        analysis = f.create_group("analysis")
        analysis.create_dataset("mean_temp", data=temp_data.mean(axis=1))
        analysis.create_dataset("std_temp", data=temp_data.std(axis=1))

    print(f"✓ Created demo file: {output_file}")
    print(f"  - Temperature: {temp_data.shape} float64")
    print(f"  - Pressure: {pressure_data.shape} float64")
    print("  - Metadata: timestamps, lat/lon")
    print("  - Analysis: mean, std")

    return output_file


if __name__ == "__main__":
    create_demo_data()
