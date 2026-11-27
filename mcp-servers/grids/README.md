# Grids MCP

Install the launcher using the [CLIO Kit setup guide](../../setup.md) before
using the commands below. See [agent integrations](../../README.md#agent-integrations)
for MCP and skill configuration.

Hand edits to 2-D gridded fields: model bathymetry, land masks and
topography. `grid_open` shows a field in an interactive editor (MCP Apps,
`ui://grids/editor`) where the person drags a box or picks cells and chooses an
edit. Every change is a `grid_edit` call, so the transcript records it.

## Edits

An edit list is ordered; each edit is `{op, region?, note?, ...}`:

| op | Arguments | Effect |
| --- | --- | --- |
| `set` | `value` | Set the covered cells to `value` (close a channel, open a strait). |
| `clip` | `min` and/or `max`, optional `wet` | Clip the covered cells, or only their wet cells, to a range (a minimum depth). |
| `fill_enclosed` | `wet`, `land_value` | Turn wet cells not connected to the largest wet area into land (remove lakes). |

`wet` is `positive`, `negative` or `nonzero`. A region is
`{box: [west, south, east, north]}` (a box may cross the dateline) or
`{points: [[lon, lat], ...]}` (the nearest cell each); without one an edit
covers the whole grid.

## Logs and replay

`grid_edit` applies the whole list to the base grid, which it never changes,
and writes the result with `<output>.edits.json`: the base path, variable and
content hash, each edit with the number of cells it changed, and the result
hash. The result depends only on the base and the list, so undo is the list
without its last edit. Edits are located by longitude and latitude, so
`grid_replay` applies a log to a regenerated grid and reports whether the base
is the logged one and, edit by edit, whether the same number of cells changed.

## Tools

| Tool | Purpose |
| --- | --- |
| `grid_open` | Open a field in the editor; shape, extent, value range, hash |
| `grid_edit` | Apply an ordered edit list; write the result and its log |
| `grid_replay` | Apply a saved log to a (regenerated) grid |
| `grid_window` | Strided values for the editor (app-only) |

Errors start with a reason code: `bad_input` or `internal_error`.
