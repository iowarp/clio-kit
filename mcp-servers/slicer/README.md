# Slicer MCP

Slice 3D models (STL, 3MF, STEP, OBJ) for a 3D printer with the slicer already on this
machine, and see the result layer by layer in an interactive view. Slicing and viewing
only: the server never talks to a printer.

## The slicer is yours, not downloaded

The server runs **PrusaSlicer** or **OrcaSlicer** that is already installed, as a
separate, unmodified program. It never downloads, bundles or updates a slicer. It looks
for one in this order, and the first answer is final:

1. `SLICER_BIN`: the full path of `prusa-slicer`/`PrusaSlicer` or `orca-slicer`/`OrcaSlicer`.
   If it does not name a working slicer, the server says so; it does not fall back to
   another one.
2. `PATH`: `prusa-slicer`, `PrusaSlicer`, then `orca-slicer`, `OrcaSlicer`.
3. The official installers' folders on macOS and Windows.

With none found, every tool answers the typed error `slicer_missing`, which says in
words which slicers are supported and how to point at one. The presets come from the
slicer's own vendor folder (`resources/profiles`); `SLICER_PROFILES` overrides it.
The slicer runs with `--datadir` set to this server's own folder, so the person's
slicer settings are never read or changed.

## Tools

| Tool | What it does |
| --- | --- |
| `list_printers(vendor?, printer?)` | The installed slicer's vendors; a vendor's printers; or one printer's quality and material presets. |
| `slice(model, printer, material?, quality?, overrides?, rotate?)` | Slices a model. Printer, material and quality are words ("Prusa MK4", "PLA", "0.2 mm") or exact preset names; the presets chosen, and how, are reported. Returns a slice id, print time, filament (g and m), layer count and the slicer's warnings in words. |
| `show_slice(slice_id, layer?)` | Opens the layer view (MCP Apps, `ui://slicer/layers`). |
| `layer_stats(slice_id, first_layer?, last_layer?)` | Time and filament by feature type over a layer range, the slowest layer, and one row per layer. |
| `export(slice_id, format, path?)` | Writes the plain-text G-code or a per-layer CSV (for a chart). |
| `compare_slices(slice_ids)` | Time, filament, presets and where the time goes, against the first slice. |

Presets are chosen with the vendor's own compatibility rules (`compatible_printers`,
`compatible_printers_condition`), evaluated by the server; a condition it cannot read
never matches. G-code is forced to plain text (`binary_gcode = 0`).

## The view

Toolpaths coloured by feature type (outer and inner walls, overhangs, sparse and solid
infill, top surfaces, bridges, gap fill, skirt and brim, support, travel), each named in
the legend with its share of print time and a toggle; a layer slider and a "from" slider;
print time, filament, layer count and height; the slicer's warnings in words; orbit,
pan and zoom; the host's theme. A click picks a toolpath: its type, layer, Z, ends,
length and G-code line.

- **State** (`ai.iowarp.clio/state`): `slice_id`, `layer` (set with `go_to_layer`),
  `layer_count`, `z_mm`, `print_time_s`, `filament_g`, `printer`, `material`,
  `visible_features` (set with `set_feature_visibility`) and `selection` (set with
  `pick_segment`), so a dashboard chart of time per layer can follow and move the view.
- **App tools** (the view's own): `go_to_layer(layer | z_mm)`,
  `set_feature_visibility(visible_features | features)`, `pick_segment(x_mm, y_mm, layer?)`.
- **Server tools only the view calls**: `slice_layers` (toolpaths in chunks) and
  `remember_slice_view` (the shown layer and types, so a reopen shows them).
- **Reopen** (`ai.iowarp.clio/reopen`): `{title, slice_id}`; slices are saved in
  clio-kit's data folder (`SLICER_MCP_HOME` overrides it).
- **Paths** (`ai.iowarp.clio/paths`): `slice.model` is read on this machine;
  `export.path` is written on this machine.

**Large G-code.** The file is read once, line by line; only the current layer is held in
memory, and toolpaths are stored compactly (`float32` points) beside the slice. The view
draws at most 1.5 million points below the shown layer; past that it draws every n-th
layer there, always the shown layer whole, and says so in words. Every number uses every
move. Per-layer and per-type times are a feedrate estimate scaled to the slicer's own
total.

three.js 0.170.0 (MIT) is loaded by the view from jsDelivr, declared in the resource's
CSP.

## Licences

PrusaSlicer and OrcaSlicer are AGPL-3.0. This server (BSD-3-Clause) runs the unmodified
program as a separate process and reads the files it writes; it does not link to,
include or redistribute any slicer code, binary or vendor profile.

## Run

```sh
clio-kit mcp-server slicer       # via the clio-kit launcher
slicer-mcp                       # direct entry point
```

## Test

```sh
uv run --frozen pytest
```

The G-code reader, preset selection, `slicer_missing`, the view's payload and the
OrcaSlicer 3MF reading run without a slicer. `tests/test_slicer_live.py` slices the
fixture tube when PrusaSlicer is installed and is skipped, with its reason, otherwise.
`tests/fixtures/make_fixture.py` regenerates the fixtures.

The view has a browser check in a stand-in MCP Apps host (headless Chromium, software
WebGL): it slices a model, opens the view, moves the slider, calls the view's tools,
picks a toolpath, orbits and zooms, and switches to the dark theme. It needs PrusaSlicer,
node with Playwright and network access to `cdn.jsdelivr.net`:

```sh
uv run --frozen python tests/browser/run_standin.py --model part.step --out /tmp/slicer-view [--width 390 --height 844]
```
