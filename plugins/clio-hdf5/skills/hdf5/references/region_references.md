# HDF5 Region References

## Purpose

Region references (`H5R_DATASET_REGION`, or `H5R_DATASET_REGION2` in the 1.12+ API) store persistent, typed pointers to a specific hyperslab or point-selection within a dataset. They enable annotation of data subsets — ROIs, detected features, calibration regions — without copying data.

**Related**: `references/vds.md` for logical dataset composition across files, `references/swmr.md` (region references are restricted under SWMR).

## What Region References Solve

Annotating a subset normally requires copying data or storing raw offsets/shapes as plain integers. Region references encode the selection in HDF5's type system — they survive file reorganization better than hardcoded offsets and are readable by any HDF5-aware tool without custom parsing.

**Typical use cases**: ROI annotation in imaging/spectroscopy data, linking derived results to exact source regions, building index datasets where each entry references a time-series segment.

## Architecture

A region reference is a fixed-size opaque blob encoding:
- A token identifying the referenced dataset within the file
- The selection type (hyperslab or point list)
- The selection geometry (start/stride/count/block or point coordinates)

Size: legacy region references (what `h5py.regionref_dtype` stores) are 12 bytes; the `H5R_ref_t` of the HDF5 1.12+ reference API is a 64-byte opaque struct. Never hard-code either size.

Legacy region references and h5py references are **same-file only** — the token is an internal identifier, not a portable path. The 1.12+ `H5R_ref_t` API can reference objects in other files (the file name is stored; see `H5Rget_file_name`).

## Reference Types

| Type | Constant | Points To |
|---|---|---|
| Object reference | `H5R_OBJECT` (legacy), `H5R_OBJECT2` (1.12+) | An HDF5 object (dataset, group) |
| Region reference | `H5R_DATASET_REGION` (legacy), `H5R_DATASET_REGION2` (1.12+) | A selection within a dataset |
| Attribute reference | `H5R_ATTR` | A specific attribute (HDF5 1.12+) |

## h5py API

**Creating a hyperslab region reference**:
`ref = dataset.regionref[start:stop, ...]` — uses numpy-style slice indexing

**Storing references**: Store in a dataset with `dtype=h5py.regionref_dtype`

**Dereferencing**:
- `src_dset = f[ref]` — returns the referenced dataset object (not the data)
- `src_dset[ref]` — the selected data via the dataset

h5py 3.16: storing region references in a dataset of object references raises `TypeError` (previously the region was silently dropped); use a `regionref_dtype` dataset for region references. A list of reference objects can now be used to create a dataset.

**Inspecting the selection geometry** (low-level): `h5py.h5r.get_region(ref, dset.id)` returns an `h5py.h5s` dataspace with the selection applied

**Point selections**: h5py's `dataset.regionref[...]` syntax only supports hyperslab (slice) syntax. Point selections require the low-level `h5py.h5s.select_elements()` API.

## Do / Do Not

**Do**:
- Store region references in a dedicated dataset with `dtype=h5py.regionref_dtype`
- Keep references and their target datasets in the same file — legacy and h5py references are same-file only
- Document what each reference dataset annotates via attributes (`purpose`, `label_scheme`)
- Use hyperslab references for contiguous rectangular regions; point selections for scattered coordinates

**Do not**:
- Attempt cross-file region references with legacy/h5py references — they encode internal tokens, not portable paths
- Dereference a region reference against a file other than the one it was created in
- Delete a referenced dataset without first removing all references pointing to it — no automatic invalidation
- Store region references in attributes on the referenced dataset itself — circular reference semantics confuse tooling
- Use region references in SWMR mode (restricted operation)
- Hardcode the 12-byte size assumption — use `h5py.regionref_dtype` instead

## Limitations

- **Same-file only**: No cross-file region references for legacy references or the h5py high-level API (the 1.12+ `H5R_ref_t` API can reference other files)
- **File copy caveat**: h5repack rewrites legacy object and region references to the new file's objects. `h5copy` copies reference values unchanged unless given `-f ref`, which also copies the referenced objects. Verify references after copying, especially 1.12+ `H5R_ref_t` references
- **No write-through**: A region reference does not provide write access to the referenced region — dereference to the dataset, then write normally
- **Tool support**: `h5dump -R` decodes region references; most high-level tools (xarray, pandas) ignore them
- **Stale references**: Deleting the referenced dataset leaves references silently invalid — dereferencing returns an error. Known problem in HDF5 2.x: file space may not be released when overwriting or deleting certain nested variable-length or reference types

## Common Pitfalls

**Wrong file for dereference**: `f[ref]` must use the file object of the file containing the target dataset. Using a different opened instance of the same file fails.

**Copying files with references**: h5repack rewrites legacy object and region references for the new file; `h5copy` needs `-f ref` to keep them resolvable. Verify references after either, especially 1.12+ `H5R_ref_t` references.

**Point selection via high-level API**: The `dataset.regionref[...]` syntax does not support point selections. Attempting to use a list of indices raises an error — use `h5py.h5s.select_elements()` instead.

**Reference size change**: the 1.12+ `H5R_ref_t` is a 64-byte opaque struct vs. 12 bytes for legacy region references. Code that serializes reference blobs externally will break across APIs.

## When to Use Region References

- Building an annotation or label layer over large, immutable source datasets
- Storing detected features / ROIs as part of the same HDF5 file as the source data
- Creating index datasets that map identifiers to exact data regions

## When NOT to Use

- Cross-file scenarios — use external links or VDS instead
- Simple rectangular slices where storing offset+shape as integers is equally clear and more portable
- Performance-critical hot paths — dereference adds per-reference overhead
- Files that will be repacked or reorganized frequently

## Citations and References

- [HDF Group: References in HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_r.html)
- [HDF Group: Object and Region References](https://support.hdfgroup.org/documentation/hdf5-docs/advanced_topics/Intros/References.html)
- [h5py: References Documentation](https://docs.h5py.org/en/stable/refs.html)
- [HDF Group: References API (1.12+)](https://docs.hdfgroup.org/hdf5/v1_14/group___h5_r.html)
