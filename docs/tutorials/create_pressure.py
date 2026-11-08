import h5py
import numpy as np

with h5py.File("pressure.h5", "x") as f:
    dataset = f.create_dataset(
        "pressure", data=100 + np.arange(100000, dtype=float) / 1000
    )
    dataset.attrs["units"] = "kPa"
print("Created pressure.h5: 100000 float64 pressures in kPa")
