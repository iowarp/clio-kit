import numpy as np
import adios2
from typing import Any, List, Optional

from .selection import check_selection, elements, parse_shape

DEFAULT_MAX_ELEMENTS = 10000


def read_variable_at_step(
    filename: str,
    variable_name: str,
    target_step: int,
    start: Optional[List[int]] = None,
    count: Optional[List[int]] = None,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
):
    """
    Read a single variable, or a start/count slice of it, from one step of a BP5 file.

    Args:
      filename: Path to the .bp directory (basename.bp)
      variable_name: Name of the variable to read
      target_step: The integer step index to fetch
      start, count: Optional hyperslab, one value per dimension
      max_elements: Refuse reads larger than this instead of returning them

    Returns:
      {"value": Python scalar or flattened row-major (C order) list, "shape": list of
      dimensions ([] for a scalar)}, plus "start" when a slice was read.

    Raises:
      ValueError: if the step or variable is not found, the selection is invalid,
      or the read is larger than max_elements.
    """
    with adios2.Stream(filename, "r") as s:
        for _ in s.steps():
            current = s.current_step()
            if current == target_step:
                # make sure the variable exists
                avail = s.available_variables()
                if variable_name not in avail:
                    raise ValueError(
                        f"Variable '{variable_name}' not in step {current}"
                    )
                info = avail[variable_name]
                selected: Optional[List[int]] = None
                if "Shape" in info:
                    shape = parse_shape(info["Shape"])
                    selected = check_selection(shape, start, count)
                    size = elements(selected)
                    if size > max_elements:
                        raise ValueError(
                            f"'{variable_name}' at step {current} selects {size} "
                            f"elements (shape {selected}), more than max_elements "
                            f"{max_elements}. Read a slice with start/count, raise "
                            "max_elements, or use variable_stats for min/max/mean "
                            "and the location of extremes."
                        )
                if start is None and count is None:
                    arr = s.read(variable_name)
                else:
                    shape = parse_shape(info.get("Shape", ""))
                    offset = start if start is not None else [0] * len(shape)
                    arr = s.read(variable_name, offset, selected)
                # convert NumPy types → native Python
                if isinstance(arr, np.generic) or (
                    hasattr(arr, "shape") and arr.shape == ()
                ):
                    return {"value": np.array(arr).item(), "shape": []}
                result: dict[str, Any] = {
                    "value": np.array(arr).flatten().tolist(),
                    "shape": list(np.shape(arr)),
                }
                if start is not None or count is not None:
                    result["start"] = (
                        start if start is not None else [0] * len(result["shape"])
                    )
                return result
    raise ValueError(f"Step {target_step} not found in file '{filename}'")
