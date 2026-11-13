---
name: hdf5
description: Use when working with HDF5 or formats built on it (netCDF-4, NeXus, NWB, .h5ad) on conventions, h5py, chunking, cloud access, parallel I/O or clio-core ingest. Triggers on "NeXus file", "h5py", "chunk cache", "ROS3", "SWMR", "parallel HDF5", "virtual dataset". Not for choosing between file formats; use storage-format.
metadata:
  bundle: clio-scientific-io
  servers: clio-hdf5
  provenance: designed
  eval-status: scenarios-recorded
---

# HDF5

This skill routes HDF5 questions to the reference guide that answers them.
The guides are in `references/`; read the ones a question needs. In CLIO,
load them with `load_skill`, for example
`load_skill("hdf5", file="references/chunking.md")`, or several at once with
`files=[...]`.

## References to objects in a file

In CLIO, a user can point at an object inside a file by typing `@`, which
inserts a reference: the file, a colon, and the object's HDF5 path.

```
@water_224.h5:/entry_0000/4_azimuthal_integration/results/I
```

- The path is the native HDF5 object path. Dots are part of names, not
  separators.
- When a message contains a reference, CLIO has already added a description
  of that object: kind, shape, dtype, attributes, storage, a small preview,
  and the convention it detected. Start from it instead of reading the same
  facts again.
- If you have the generic structured-file tools (`sf_list`, `sf_describe`,
  `sf_read`), pass them the file id and path from that description.
  `sf_read` takes a selection with one slice per dimension, such as `0:100`
  or `5, ::2`. Keep selections small; the read is capped and comes back
  marked truncated when it hits the cap.
- When your answer names an object, write it as a reference in the same form,
  so the user can open it in the viewer.

## How to work

- **Inspect before advising.** When a file is available, look at it first;
  never invent file facts. Report the paths, shapes, dtypes, chunking and
  attributes you actually saw.
- **Use the HDF5 tools if you have them** (`open_file`, `list_keys`,
  `get_by_path`, `list_attributes`, `read_partial_dataset`, `close_file`, …).
  Otherwise use h5py through code execution: open read-only with
  `h5py.File(path, "r")`, list with `list(f)` and `f.attrs`, and sample
  values with slices such as `dset[:10]`. Never read a whole large dataset
  to look at it.
- **Start with the convention.** For an unfamiliar file, identify the
  community convention it follows before walking it:
  `references/conventions.md`. It tells you where the primary data is.
- **Check versions before version-specific advice.** The guides describe
  HDF5 2.2 and h5py 3.16. Many behaviors and defaults changed in HDF5 2.0;
  in h5py, check `h5py.version.hdf5_version` and `h5py.version.version`.
- **Advice is not measurement.** Present tuning advice as recommendations
  unless you measured it.

## Which guide

**What is in a file**

| Guide | Use for |
|---|---|
| `references/conventions.md` | Which community convention a file follows and where its main data is |
| `references/datatypes.md` | Compound, string, enum, variable-length and other datatypes; h5py type mapping |
| `references/dimension_scales.md` | Dimension scales and coordinate variables |
| `references/cf_compliance.md` | Whether a netCDF-4/HDF5 file meets the CF conventions |
| `references/region_references.md` | Object and region references |
| `references/map_objects.md` | HDF5 map objects (key-value stores) |

**Layout and performance**

| Guide | Use for |
|---|---|
| `references/chunking.md` | Storage layout, chunk shape and the chunk cache |
| `references/filters.md` | Compression and other filters, and choosing among them |
| `references/io.md` | General I/O tuning: buffers, caches, access patterns |
| `references/file_space.md` | File size, free space, page allocation, repacking |

**Remote and cloud.** Choose by how the data is reached:
reading an existing file directly from S3 or HTTP is ROS3; going through a
service layer is HSDS; changing how a file is *written* so remote reads are
fast is the cloud-optimized guide.

| Guide | Use for |
|---|---|
| `references/ros3_vfd.md` | Reading HDF5 files directly from S3 or S3-compatible storage |
| `references/hsds.md` | HDF5 through the HSDS REST service (S3, Azure, POSIX backends) |
| `references/cloud_optimized.md` | Writing or rewriting files for efficient remote access |

**Concurrency and parallel I/O.** One writer with live readers is SWMR;
many MPI processes writing one file is parallel HDF5; at very large scale,
subfiling splits that file across subfiles.

| Guide | Use for |
|---|---|
| `references/swmr.md` | Single-writer/multiple-reader access to a file being written |
| `references/parallel.md` | MPI parallel HDF5 and collective I/O |
| `references/subfiling_vfd.md` | The subfiling driver for large-scale parallel writes |

**Drivers, connectors and virtual data**

| Guide | Use for |
|---|---|
| `references/core_vfd.md` | In-memory files with the core driver |
| `references/onion_vfd.md` | Revision history with the onion driver |
| `references/vds.md` | Virtual datasets that combine data from several files |
| `references/vol_usage.md` | VOL connectors such as async, cache, DAOS and REST |

**CLIO**

| Guide | Use for |
|---|---|
| `references/clio_core_ingest.md` | Whether to ingest an HDF5 file into clio-core or read it with h5py |
| `references/omni_selective.md` | OMNI configurations that load only the datasets a task needs |

Questions often span guides: a slow remote read may need `ros3_vfd.md` and
`chunking.md`; compression choices depend on `chunking.md` and
`filters.md`. Load each guide whose topic the question touches.
