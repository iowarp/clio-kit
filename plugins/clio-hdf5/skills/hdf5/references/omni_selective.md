# Producing OMNI YAML Files for Selective HDF5 Data Assimilation

## Purpose

Generate IOWarp OMNI YAML files that selectively ingest **only the HDF5 datasets needed** for a figure/paper/analysis — avoiding costly transfer of unused data.

**Context**: IOWarp's Context Assimilation Engine (CAE) uses OMNI YAML files to describe data transfers. For HDF5 files, the CAE recursively discovers datasets and loads them into CTE (Context Transfer Engine) tags. Without filtering, it loads *everything*. For large scientific files (GB+), this wastes bandwidth and memory.

## OMNI YAML Structure for HDF5

### Minimal (all datasets)

```yaml
transfers:
  - src: "hdf5::/path/to/file.h5"
    dst: "iowarp::my_data"
    format: "hdf5"
```

### Selective (filtered datasets)

```yaml
transfers:
  - src: "hdf5::/path/to/file.h5"
    dst: "iowarp::my_data"
    format: "hdf5"
    dataset_filter:
      include_patterns:
        - "/temperature/surface"
        - "*/Geolocation/*"
      exclude_patterns:
        - "*/Calibration/*"
        - "*_backup"
```

### Required fields per transfer

| Field | Value | Notes |
|-------|-------|-------|
| `src` | `hdf5::<file_path>` | Double-colon separator, absolute path |
| `dst` | `iowarp::<tag_name>` | Base tag; each dataset appended as `<tag>/<dataset>` |
| `format` | `"hdf5"` | Must be exactly `"hdf5"` |

### Optional fields

| Field | Default | Notes |
|-------|---------|-------|
| `dataset_filter` | (none — all datasets) | See filtering below |
| `depends_on` | `""` | Name of a prior transfer to wait for |
| `range_off` | `0` | Not typically used for HDF5 |
| `range_size` | `0` | Not typically used for HDF5 |
| `name` | — | Human label for the transfer |
| `description` | — | Human description |

## Dataset Filtering

Filtering uses glob patterns via `fnmatch`. This is the critical mechanism for selective download.

### Rules

1. **Exclude patterns are checked first.** If any exclude pattern matches, the dataset is skipped.
2. **If include patterns are specified**, only datasets matching at least one are kept.
3. **If no include patterns are specified**, all non-excluded datasets are included.
4. Dataset paths have a leading `/` (e.g., `/group/subgroup/dataset`).

### Glob syntax

| Pattern | Meaning | Example |
|---------|---------|---------|
| `*` | Any characters (including `/`) | `*/temperature` matches `/data/temperature` |
| `?` | Single character | `/data?` matches `/data1` |
| `[abc]` | Character class | `/data[12]` matches `/data1`, `/data2` |

**Note**: `fnmatch` is called without `FNM_PATHNAME`, so `*` matches across `/` separators.

### Common filtering strategies

**Select specific datasets by exact path:**
```yaml
dataset_filter:
  include_patterns:
    - "/temperature/surface"
    - "/pressure/sea_level"
```

**Select an entire group and its children:**
```yaml
dataset_filter:
  include_patterns:
    - "/Geolocation/*"
```

**Select across groups by dataset name pattern:**
```yaml
dataset_filter:
  include_patterns:
    - "*/radiance*"
```

**Exclude large auxiliary data:**
```yaml
dataset_filter:
  exclude_patterns:
    - "*/Calibration/*"
    - "*/Quality/*"
    - "*_raw"
```

## Workflow: From Figure/Paper to OMNI File

When a user has a figure or paper and wants to know which datasets to include:

### Step 1: Inspect the HDF5 file structure

```python
import h5py

def list_datasets(h5_path):
    datasets = []
    with h5py.File(h5_path, 'r') as f:
        f.visititems(lambda name, obj: datasets.append(f"/{name} {obj.dtype} {obj.shape}") if isinstance(obj, h5py.Dataset) else None)
    for d in sorted(datasets):
        print(d)

list_datasets("/path/to/file.h5")
```

Or with CLI: `h5ls -r /path/to/file.h5`

### Step 2: Identify required datasets

From the figure/paper, determine:
- **What variables are plotted** (e.g., temperature, radiance, pressure)
- **What coordinate/geolocation data is needed** (lat, lon, time, altitude)
- **What ancillary data is needed** (quality flags, land/sea masks)

Ignore: raw counts, calibration coefficients, engineering data, redundant backup datasets.

### Step 3: Write the OMNI YAML with tight include patterns

Prefer **explicit include patterns** over broad includes + excludes. List exactly what you need:

```yaml
transfers:
  - src: "hdf5::s3://bucket/ASTER_L2T.h5"
    dst: "iowarp::aster_fig3"
    format: "hdf5"
    dataset_filter:
      include_patterns:
        - "/ASTER/Temperature/TIR_Surface_Temperature"
        - "/ASTER/Geolocation/Latitude"
        - "/ASTER/Geolocation/Longitude"
```

### Step 4: Validate dataset paths

The include patterns must match the actual dataset paths in the file (with leading `/`). Verify with:

```bash
h5ls -r file.h5 | grep -i "temperature\|latitude\|longitude"
```

## Multiple Files for One Analysis

Use multiple transfers for multi-file analyses:

```yaml
transfers:
  - src: "hdf5::/data/simulation_t0.h5"
    dst: "iowarp::timeseries"
    format: "hdf5"
    dataset_filter:
      include_patterns:
        - "/fields/temperature"
        - "/mesh/coordinates"

  - src: "hdf5::/data/simulation_t1.h5"
    dst: "iowarp::timeseries_t1"
    format: "hdf5"
    depends_on: "timeseries"
    dataset_filter:
      include_patterns:
        - "/fields/temperature"
        - "/mesh/coordinates"
```

## What Happens at Runtime

For reference (not needed to produce OMNI files):

1. CAE parses the OMNI YAML into `AssimilationCtx` objects
2. HDF5 assimilator opens the file, discovers all datasets via `H5Lvisit`
3. Filters datasets against include/exclude patterns
4. Each surviving dataset becomes a CTE tag: `<dst>/<dataset_path>`
5. Tensor metadata stored as description (e.g., `tensor<float64, 100, 200>`)
6. Data transferred in ~1.5 MB chunks across available nodes
7. Run via: `wrp_cae_omni <omni_file.yaml>`
