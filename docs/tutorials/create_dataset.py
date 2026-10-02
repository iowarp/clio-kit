import h5py
import numpy as np

with h5py.File("simulation.h5", "x") as file:
    file.attrs["experiment"] = "Heat diffusion tutorial fixture"
    time = file.create_dataset("time", data=np.arange(5, dtype=float))
    time.attrs["units"] = "s"
    field = file.create_dataset(
        "temperature", data=273.15 + np.arange(60, dtype=float).reshape(5, 4, 3)
    )
    field.attrs["units"] = "K"
    field.attrs["axes"] = "time, y, x"
print("Created simulation.h5: time (5,), temperature (5, 4, 3); units s and K")
