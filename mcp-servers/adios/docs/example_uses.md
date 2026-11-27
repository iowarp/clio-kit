# Inspecting ADIOS BP5 data

Configure the server with the [CLIO Kit setup guide](../../../setup.md).
Use files accessible to the MCP process, and inspect metadata before reading
variable values. The current server exposes five tools:

| Tool | Inputs | Use |
| --- | --- | --- |
| `list_bp5` | `directory` | Find BP5 files and their metadata |
| `inspect_variables` | `filename`, optional `variable_name` | Inspect names, types, shapes and available steps |
| `inspect_variables_at_step` | `filename`, `variable_name`, `step` | Inspect one variable at a selected step |
| `inspect_attributes` | `filename`, optional `variable_name` | Read global or variable attributes |
| `read_variable_at_step` | `filename`, `variable_name`, `target_step` | Read one variable's values at a selected step |

For example, ask the agent to find BP5 data under your data directory, inspect
the variables in a selected file, and report the shape and attributes of one
variable. Choose an available step from that metadata before reading it.
Inspect its units and dimensions before comparing values across files.

`step` and `target_step` are different parameter names on different tools;
copy the live input schema for the tool being called. Read-variable calls can
return substantial arrays, so choose a small known variable for verification.
The current tool surface does not expose the old read-all, variable-addition
or min/max operations.

See the [server reference](../README.md#capabilities) for tool descriptions and
[ADIOS setup](adios_setup.md) for native development prerequisites.
