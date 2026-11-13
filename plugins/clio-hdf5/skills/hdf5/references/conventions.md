# HDF5 Conventions

## Purpose

Most HDF5 files follow a community convention layered on top of HDF5. The
convention says where the primary data is and how to read it, so identify it
before exploring or advising. Use this when a user asks what a file is, which
dataset matters, or what to plot, and before reading data from an unfamiliar
file.

**Related:** `references/cf_compliance.md` for CF metadata questions,
`references/dimension_scales.md` for netCDF-4 coordinates.

## How to detect

Detection uses attributes and object names only; never read dataset values
to decide. Start with the root group's attributes, then the names of its
members (with h5py: `dict(f.attrs)` and `list(f)`). A few conventions need
one more step, noted below.

Report the evidence: name the attribute or group you matched ("root
attribute `Conventions = CF-1.8`"). Never name a convention from the file
extension alone.

A file can match several conventions: CF sits on netCDF-4, pandas sits on
PyTables, and HDF-EOS5 files often carry CF attributes as well. Report each
one you find.

| Convention | Signature | Primary data |
|---|---|---|
| NeXus | A top-level group with attribute `NX_class` (usually `NXentry`). The root `default` attribute is optional | Follow root `default` to the entry, then the entry's `default` to an `NXdata` group. Its `signal` attribute names the dataset and `axes` names the coordinates. With no `default`, use the first `NXdata` group |
| netCDF-4 | Root attribute `_NCProperties` (netCDF 4.4 and later). Older files: datasets with `CLASS = DIMENSION_SCALE` and `_Netcdf4Dimid` | Variables that carry `DIMENSION_LIST`; dimension scales are coordinates |
| CF (on netCDF-4) | Root attribute `Conventions` starting with `CF-` | Data variables with `standard_name`; coordinates from dimension scales and each variable's `coordinates` attribute |
| HDF-EOS5 | Group `/HDFEOS INFORMATION` with attribute `HDFEOSVersion` | `/HDFEOS/GRIDS/<grid>/Data Fields` or `/HDFEOS/SWATHS/<swath>/Data Fields`; geolocation in `Geolocation Fields` |
| NWB | Root attributes `neurodata_type = NWBFile` and `nwb_version` | `/acquisition` (raw recordings) and `/processing` (derived data); each object's type is its `neurodata_type`. The schema is embedded under `/specifications` |
| AnnData (`.h5ad`) | Root attribute `encoding-type = anndata` | `X` is the main matrix; `obs` and `var` are tables of row and column annotations |
| openPMD | Root attributes `openPMD` (a version) and `basePath` | `basePath` (for example `/data/<iteration>/`) plus `meshesPath` and `particlesPath`. With file-based iteration, each file holds one timestep |
| VTKHDF | Group `/VTKHDF` with attributes `Version` and `Type` | Under `/VTKHDF`, laid out by `Type` (`ImageData`, `UnstructuredGrid`, …); fields in `PointData` and `CellData` |
| PyTables | Root attribute `PYTABLES_FORMAT_VERSION` | Nodes whose `CLASS` is `TABLE`, `ARRAY` or `EARRAY` |
| pandas HDFStore | Groups with `pandas_type` (for example `frame_table`) and `pandas_version` | Each `pandas_type` group is one DataFrame |
| MATLAB v7.3 | Datasets and groups with attribute `MATLAB_class`; the file begins with a 512-byte `MATLAB 7.3 MAT-file` header | Each top-level object is one MATLAB variable |
| Keras (legacy `.h5`) | Root attributes `keras_version` and `model_config` | Weights under `model_weights/<layer>/` |

XDMF keeps its structure in a separate `.xmf` XML file, so an HDF5 file
written for XDMF looks like plain HDF5. If the user mentions XDMF, ask for
the `.xmf` file.

## After detecting

- Lead with the primary data the convention points to, not with a full
  walk of the file.
- Use the convention's own metadata for units and coordinates: NeXus
  `axes` and `units`, CF `units` and `standard_name`, NWB `unit`.
- For a tabular primary object (AnnData `obs`/`var`, a pandas group), say it
  is a table and give its columns; don't treat it as one array.
- When nothing matches, say the file follows no recognized convention and
  explore it as plain HDF5.
