# Seismology MCP

Install the launcher using the [CLIO Kit setup guide](../../setup.md) before
using the commands below. See [agent integrations](../../README.md#agent-integrations)
for MCP and skill configuration.

Analyzes SAC seismic-waveform files that already exist on disk. Point it at a
single `.sac` file or a `.tar` / `.tar.gz` / `.tgz` archive of SAC files and it
will inspect members, compute per-trace statistics, and plot traces. It does
**not** retrieve or download any data.

SAC binary headers are parsed with pure stdlib (`struct`, `tarfile`);
endianness is auto-detected. Statistics use NumPy and plotting uses Matplotlib.

## Tools

### `inspect_archive`

List the SAC members of a file/archive: count, sample member names and sizes,
and inferred stations and phases. Read-only.

```jsonc
{ "filepath": "events.tar.gz", "member_filter": "BHZ", "max_members": 12 }
```

Returns `{status, filepath, sac_trace_count, sample_members, sample_sizes_bytes,
phases, stations, members_truncated}`.

### `compute_trace_statistics`

Per-trace `min`, `max`, `mean`, `std`, `peak_abs` plus header metadata
(`npts`, `delta_s`, `begin_s`, `end_s`). Read-only.

```jsonc
{ "filepath": "events.tar.gz", "member_filter": "ANMO", "max_traces": 6 }
```

Returns `{status, filepath, sac_trace_count, traces_analyzed, traces,
traces_truncated}`.

### `plot_traces`

Render amplitude-normalized, vertically offset traces to a PNG.

```jsonc
{ "filepath": "events.tar.gz", "max_traces": 3, "output_path": "traces.png" }
```

Returns `{status, filepath, output_path, sac_trace_count, traces_plotted,
members, duration_ms}`.

### Earthquake catalogues

`analyze_sequence` computes descriptive statistics from saved GeoJSON/JSON or
CSV event catalogues, including completeness magnitude, b-value uncertainty
and event-rate decay. It does not classify an earthquake sequence.
`plot_sequence` writes an epicenter map, magnitude-frequency distribution and
cumulative event plot. These tools take `catalog_path`; raw SAC waveforms are
not an event catalogue. Neither tool downloads data.

## Run

```sh
clio-kit mcp-server seismology          # via the clio-kit launcher
seismology-mcp            # direct entry point
```

## Test

```sh
uv run --frozen pytest
```

## Capabilities

### `inspect_archive`
**Description**: Inspect a staged SAC file or TAR archive and summarize its SAC waveform members: count, a sample of member names and sizes, and the inferred stations and phases. Read-only; a good first step before computing statistics or plotting.
**Hints**: read-only, idempotent
**Tags**: inspect, sac, seismic, waveform

### `compute_trace_statistics`
**Description**: Compute per-trace amplitude statistics (min, max, mean, std, peak_abs) plus header metadata (npts, delta_s, begin_s, end_s) for SAC traces in a file or archive. Read-only; bounded by max_traces.
**Hints**: read-only, idempotent
**Tags**: sac, seismic, statistics, waveform

### `plot_traces`
**Description**: Plot selected SAC traces from a file or archive to a PNG artifact. Traces are amplitude-normalized and vertically offset. Writes a file; returns the output path, plotted member names, and render duration.
**Hints**: destructive, idempotent
**Tags**: plot, sac, seismic, visualization, waveform

### `analyze_sequence`
**Description**: Compute the descriptive statistics of a saved earthquake catalog: completeness magnitude (Mc), the Gutenberg-Richter b-value with uncertainty, the largest event, the Bath-law magnitude gap to the second-largest, the share of events before vs after the largest, the spatial extent, and the Omori post-event rate decay. Returns statistics ONLY - it does not classify the sequence.
**Hints**: read-only, idempotent
**Tags**: earthquake, gutenberg-richter, omori, seismic, statistics

### `plot_sequence`
**Description**: Render the three-panel earthquake-sequence figure from a saved catalog: (1) an epicenter map sized by magnitude and coloured by time, (2) the Gutenberg-Richter magnitude-frequency distribution with an optional b-value fit line, and (3) the cumulative count over time. Writes a PNG and returns its path; pass mc/b_value to draw the G-R fit line.
**Hints**: destructive, idempotent
**Tags**: earthquake, gutenberg-richter, plot, seismic, visualization

### Resources

- `seismology://capabilities` - What this server can do and the inputs it accepts.

### Prompts

- **analyze_sac_archive**: Guided workflow for inspecting and analyzing a SAC file or archive.
- **characterize_sequence**: Guided workflow for characterizing a saved earthquake catalog.
## Claude Code

```bash
claude mcp add clio-seismology -- clio-kit mcp-server seismology
```

Or install via the CLIO Kit plugin marketplace:

```
/plugin marketplace add iowarp/clio-kit
/plugin install clio-seismology@clio-kit
```
## Claude Desktop

Add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "clio-seismology": {
      "command": "clio-kit",
      "args": [
        "mcp-server",
        "seismology"
      ]
    }
  }
}
```