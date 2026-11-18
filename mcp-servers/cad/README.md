# CAD MCP server

Model solid parts with [build123d](https://github.com/gumyr/build123d) code (an
OpenCascade kernel, pip-installable), export them as STL and STEP, and rotate them
in an interactive 3D view. The agent writes the part as a short Python script; the
server runs it in its own process, so a bad script cannot take the server down.

## Tools

| Tool | What it does |
| --- | --- |
| `build_part` | Runs a build123d script that assigns the finished shape to `result`, writes `part.py`, `part.stl`, `part.step` and `build.json` to `<out_dir>/<name>`, returns size, bounds, volume, area, solid count and validity, and opens the part view. |
| `show_part` | Opens a built part in the view again. |
| `list_parts` | The parts in a folder, with size and volume. |
| `part_examples` | Worked scripts: `naca_wing_section`, `ahmed_car_body`, `cylinder`, `pipe_bend`. |

The view (`ui://cad/part`, MCP Apps) draws the part with three.js 0.170.0 from
jsDelivr, shows the flow direction (+x) and the axes, and has buttons to look from
the front, top, side or an iso angle and to draw the triangle edges. It reports
the part and camera direction with `ui/update-model-context` and offers the app
tool `look_from`. The view calls `part_mesh` and `remember_part_view`, which are
visible only to it.

Parts are drawn in the frame the flow servers use: x along the flow, y up, z
across, in body lengths. A wing section is sketched in x-y and extruded along z.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `CAD_PARTS_DIR` | `~/.clio/cad/parts` | Where parts go when `out_dir` is empty. |
| `CAD_BUILD_TIMEOUT` | `600` | Seconds a script may run before it is stopped. |

## Run

```bash
uv run cad-mcp
```

## Development

```bash
uv sync --frozen --dev
uv run pytest -q
uv run ruff check src tests
```
