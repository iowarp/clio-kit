# HDF5 Onion VFD (File-Level Revision Control)

## Purpose

The Onion Virtual File Driver (VFD) enables file-level revision tracking across open/close cycles. By default the revision history is kept in a companion `.onion` history file next to the original file, which stays unchanged and readable without Onion. Instead of rewriting full files for each version, it stores layered revisions using copy-on-write semantics, allowing access to historical file states, creation of new revisions from prior ones, and provenance tracking.

**Related**: `references/swmr.md` for concurrent read/write access, `references/core_vfd.md` for in-memory file operations.

## What Problem Onion VFD Solves

Scientific workflows often need to track how an HDF5 file evolves over time: what data was present at a given stage, what changed between runs, or the ability to roll back to a prior state. Without Onion VFD, this requires external version control (which handles binary files poorly) or manually maintaining multiple file copies. Onion VFD provides internal, layered versioning at the VFD level, below the logical object model.

## What Onion VFD Is Not

- **Not a dataset-level version control system** -- revisions occur at the file level, not per-dataset
- **Not a branching/merging system** -- revisions are linear; there is no branching, merging or diffing (each revision does carry an optional comment and a timestamp)
- **Not a replacement for distributed version control** -- use Git/DVC for project-level history

## Conceptual Model

- **Base file** = initial state of the HDF5 file
- **Each open/write/close cycle** = a new revision layer
- **Revisions stored as deltas** (copy-on-write), not full copies
- **Reading an old revision** = reconstructing state from layered deltas
- Revisions are linear unless specifically manipulated

## C API Usage

### Enabling Onion VFD

```c
H5FD_onion_fapl_info_t onion_config;
/* Configure onion_config fields */

hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
H5Pset_fapl_onion(fapl, &onion_config);
hid_t file = H5Fcreate("data.h5", H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

/* ... write data ... */

H5Fclose(file);
H5Pclose(fapl);
```

### Querying Revision Count

```c
uint64_t rev_count;
H5FDonion_get_revision_count("data.h5", fapl, &rev_count);
/* Always validate revision ID bounds before accessing */
```

### Opening a Prior Revision

Set the desired revision index in the configuration structure before opening. Writing from a prior revision creates a new revision layer -- it does not modify the old one.

## Key Configuration (H5FD_onion_fapl_info_t)

The Onion VFD uses a configuration structure passed to `H5Pset_fapl_onion`. Understand each field before enabling:

- **Revision index**: Which revision to open (0-based). Validate against `H5FDonion_get_revision_count` before use.
- **Creation flags**: Control behavior for file creation vs. opening existing revisions.

The structure's fields:

- `version`: structure version; set it to the current Onion FAPL info version constant
- `backing_fapl_id`: FAPL for the underlying (backing) driver that does the real file I/O
- `page_size`: revision page size in bytes (must be a power of two)
- `store_target`: where the revision history is stored (the default `.onion` history file)
- `revision_num`: which revision to open
- `force_write_open`: allow opening for write when the history file carries a write-lock marker (recovery use)
- `creation_flags`: options applied when the history file is created (see the `H5FD_ONION_FAPL_INFO_CREATE_FLAG_*` constants)
- `comment`: user comment recorded with the new revision

## Best Practices

- **Always explicitly set the VFD** -- Never rely on defaults. The FAPL must explicitly configure Onion via `H5Pset_fapl_onion`.
- **Minimize open/close cycles** -- Each close may create a revision layer. Batch writes within a single session where possible.
- **Flush before close** -- Ensure data integrity and clean revision commits with `H5Fflush` before `H5Fclose`.
- **Monitor revision count** -- Periodically check revision growth with `H5FDonion_get_revision_count`.
- **Plan a retention strategy** -- Onion has no automatic garbage collection. Revisions accumulate indefinitely.
- **Validate revision indices** -- Always bounds-check revision IDs before attempting access.
- **Benchmark before production** -- Test performance characteristics with representative workloads.

## Performance Characteristics

**Overhead**:
- Increased metadata overhead per revision
- Slower random access when many revisions exist (delta reconstruction)
- Larger history file over time due to accumulated revision layers
- Potential fragmentation within the file

**Mitigation**:
- Periodic archival of the file (flatten revisions)
- Controlled revision retention policies
- Benchmarking with production-scale data before deployment

## Pitfalls to Avoid

**Treating Onion as Git**: There is no branching, merging or diffing, and the per-revision comment is only an annotation. Onion provides linear, file-level revision layers only.

**Assuming dataset-level versioning**: Revisions capture the entire file state, not individual datasets. You cannot version one dataset independently of others.

**Ignoring HDF5 version compatibility**: Onion VFD is available since HDF5 1.14.2 and in 2.x; no Onion changes are listed in the 2.0 through 2.2 changelogs. APIs may change across releases. Always confirm your HDF5 version supports the Onion VFD functions you need.

**Mixing with unsupported VFD stacks**: Ensure compatibility when stacking drivers. Not all VFD combinations are valid.

**Unbounded revision growth**: Without explicit cleanup, revision layers accumulate and file size grows indefinitely. This is the most common production issue.

## When to Use Onion VFD

- Tracking provenance of an HDF5 file across processing stages
- Rolling back to a known-good file state after a failed operation
- Auditing what data was present at a specific point in time
- Lightweight versioning where external VCS is impractical for large binary files

## When NOT to Use Onion VFD

- Per-dataset version tracking (Onion is file-level only)
- Concurrent read/write access (use SWMR instead)
- Branching or non-linear history (use external VCS)
- High-frequency write cycles where revision overhead is unacceptable
- Project-level version control (use Git, DVC, or similar)

## Differences from Related HDF5 Features

**Onion VFD vs. SWMR**:
- Onion provides version history across open/close cycles
- SWMR provides concurrent read/write safety within a single session
- These solve different problems and are not interchangeable

**Onion VFD vs. External Version Control**:
- Onion stores revision layers in a companion `.onion` history file beside the original file
- Git/DVC store file snapshots externally with branching and merging
- Onion is better for in-file provenance; external VCS is better for project-level collaboration

## Version Requirements

- HDF5 >= 1.14.2 (first stable release with the Onion VFD; developed in the 1.13 series), and HDF5 2.x
- Check release notes for API stability and changes in newer versions

## Citations and References

- [HDF Group: Virtual File Layer Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html)
- [HDF Group: Registered Virtual File Drivers](https://support.hdfgroup.org/documentation/hdf5-docs/registered_virtual_file_drivers_vfds.html)
- [HDF Group: File Access Properties](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html)
