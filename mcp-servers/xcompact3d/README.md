# Xcompact3d MCP

Set up [Xcompact3d](https://github.com/xcompact3d/Incompact3d) flow simulations
around a CAD body, follow them while they run and read the forces back.

The server turns a closed STL surface into a ready-to-run case: an immersed
boundary body mask, a grid and time step sized from the body and the Reynolds
number, ADIOS2 snapshot output and a `run.sh`. It does not start the solver.
Submit `run.sh`, or the `relay_job` job spec it returns, with any batch launcher
(for example clio-relay's `run`). Open the `view` descriptor in a viewer that
follows a growing source (for example the ParaView MCP's `open_view`) to watch
the wake form step by step.

## Tools

| Tool | What it does |
| --- | --- |
| `prepare_case` | Writes a run directory for the flow past an STL body and returns the plan, a job spec and a view descriptor. |
| `check_case` | Reports the step, simulated time, snapshots written, speed and time left, and whether the run failed or diverged. |
| `read_forces` | Drag and lift coefficients (mean, rms, extremes, L/D), the lift frequency and the Strouhal number, with a time series. |
| `build_instructions` | The build guide and the two source patches the cases need. |

Resources: `xcompact3d://build`, `xcompact3d://patches/{name}` and
`xcompact3d://capabilities`.

## The case

The case is Xcompact3d's cylinder wake flow configuration (`itype = 5`). A
uniform inflow U runs along +x into a box with a convective outflow and
periodic sides. Lengths are scaled so the body's reference length L is 1, so
`Re = U L / nu` and time is in L/U.

- **Body.** The STL is scaled, turned nose-up by `angle_deg` about +z and
  voxelised into a solid mask on the solver grid. The geometry patch makes
  Xcompact3d read that mask (`clio-geometry.txt` and `.bin`) in place of its
  built-in cylinder.
- **Span.** `periodic` treats an extruded section (a wing section, a
  cylinder) as spanning the periodic z direction, so coefficients are per unit
  span. `free` puts a 3D body (a car, a finite wing) in the middle of the box;
  coefficients use the frontal area. `auto` picks from the shape.
- **Resolution.** `points_per_length` sets an isotropic spacing. About 6
  points across the thinnest part of the body is the least that holds the
  immersed boundary; the plan warns below that. Cost grows with its cube.
- **Time step.** It follows the CFL target on the inflow speed, with a
  viscous check, rounded to a short decimal.
- **Output.** Snapshots of `ux uy uz pp vort critq ep` go to `data.bp5`
  through ADIOS2, as VTK ImageData that ParaView's `ADIOS2VTXReader` opens
  while it grows. `ep` is the body mask, so a 0.5 contour shows the body.
- **Forces.** A control volume around the body writes drag and lift to
  `forces.dat` every step.

## Building the solver

The cases need Xcompact3d v5.0 built with ADIOS2 and two source patches: the
VTX patch, which adds the VTK schema to the ADIOS2 output, and the geometry
patch, which reads the body mask. `build_instructions` returns the full guide
and writes the patch scripts. The executable is found from `XCOMPACT3D_BIN`, or
at `~/.clio/xcompact3d/current/bin/xcompact3d`. When an
`xcompact3d-env.sh` sits next to the executable, `run.sh` sources it first.

## Running

```bash
uv run xcompact3d-mcp
```

The server is stdio. It reads and writes the paths it is given, so run it on a
machine that sees the solver's filesystem.

## Development

```bash
uv sync --frozen --dev
uv run --frozen pytest -q
uv run --frozen ruff check src tests
```
