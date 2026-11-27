"""Per-step statistics of one variable, so a time scan needs one call."""

from typing import Any, Dict, List, Optional

import numpy as np
from adios2 import FileReader

from .selection import check_selection, parse_shape

MAX_STEPS = 1000


def _index(flat: int, shape: List[int], start: List[int]) -> List[int]:
    local = np.unravel_index(flat, shape) if shape else ()
    return [int(i) + offset for i, offset in zip(local, start)]


def variable_stats(
    filename: str,
    variable_name: str,
    step_start: int = 0,
    step_count: Optional[int] = None,
    start: Optional[List[int]] = None,
    count: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Min, max, mean and std of a variable for each step in a range, with the global
    (i, j, k, ...) index of each step's min and max, over the whole variable or a
    start/count slice. NaNs are ignored.
    """
    with FileReader(filename) as reader:
        variables = reader.available_variables()
        if variable_name not in variables:
            raise ValueError(
                f"Variable '{variable_name}' not found in file. "
                f"Available variables: {sorted(variables)}"
            )
        info = variables[variable_name]
        total_steps = int(info.get("AvailableStepsCount", "1"))
        shape = parse_shape(info.get("Shape", ""))
        selected = check_selection(shape, start, count)
        offset = start if start is not None else [0] * len(shape)
        if step_start < 0 or step_start >= total_steps:
            raise ValueError(
                f"step_start {step_start} is outside 0..{total_steps - 1} "
                f"({total_steps} steps)"
            )
        last = total_steps if step_count is None else step_start + step_count
        if step_count is not None and step_count <= 0:
            raise ValueError("step_count must be positive")
        last = min(last, total_steps, step_start + MAX_STEPS)

        steps: List[Dict[str, Any]] = []
        for step in range(step_start, last):
            if shape:
                data = reader.read(
                    variable_name, offset, selected, step_selection=[step, 1]
                )
            else:
                data = reader.read(variable_name, step_selection=[step, 1])
            values = np.asarray(data, dtype=np.float64).reshape(selected or [1])
            if np.isnan(values).all():
                steps.append({"step": step, "min": None, "max": None, "nan": True})
                continue
            row: Dict[str, Any] = {
                "step": step,
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
            }
            if not shape:
                row["value"] = row["min"]
            else:
                row["mean"] = float(np.nanmean(values))
                row["std"] = float(np.nanstd(values))
                row["argmin"] = _index(int(np.nanargmin(values)), selected, offset)
                row["argmax"] = _index(int(np.nanargmax(values)), selected, offset)
            steps.append(row)

        valid = [row for row in steps if row.get("max") is not None]
        result: Dict[str, Any] = {
            "variable_name": variable_name,
            "shape": shape,
            "steps_total": total_steps,
            "steps": steps,
        }
        if shape:
            result["index_order"] = "row-major (C), global indices"
        if start is not None or count is not None:
            result["selection"] = {"start": offset, "count": selected}
        if last < (total_steps if step_count is None else step_start + step_count):
            result["steps_truncated"] = True
        if valid:
            peak = max(valid, key=lambda row: row["max"])
            low = min(valid, key=lambda row: row["min"])
            result["overall"] = {
                "max": peak["max"],
                "max_step": peak["step"],
                "min": low["min"],
                "min_step": low["step"],
            }
            if shape:
                result["overall"]["max_index"] = peak["argmax"]
                result["overall"]["min_index"] = low["argmin"]
        return result
