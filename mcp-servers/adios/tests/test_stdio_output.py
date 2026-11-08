"""Native ADIOS diagnostics must not corrupt the MCP stdout channel."""

import adios2
import numpy as np
from adios_mcp.implementation.bp5_inspect_variables_at_step import (
    inspect_variables_at_step,
)


def test_step_inspection_keeps_stdout_clean(tmp_path, capfd):
    path = str(tmp_path / "values.bp")
    with adios2.Stream(path, "w") as writer:
        writer.write("temperature", np.array([1.0, 2.0, 3.0]))
    capfd.readouterr()
    result = inspect_variables_at_step(path, "temperature", 0)
    assert result["Shape"] == "3"
    assert capfd.readouterr().out == ""
