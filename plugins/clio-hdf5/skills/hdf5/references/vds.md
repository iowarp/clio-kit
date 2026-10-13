# HDF5 Virtual Datasets (VDS)

## Purpose

Virtual Datasets enable unified access to data distributed across multiple HDF5 files as a single logical dataset, without copying or reorganizing data. The mapping is persistent and transparent to applications.

**Related**: `references/chunking.md` for chunk configuration, `references/filters.md` for compression.

## What Problem VDS Solves

Accessing data scattered across many HDF5 files typically requires opening each file separately, tracking locations, and managing coordination logic. VDS creates a single dataset that maps to distributed sources, allowing standard HDF5 operations on the unified view.

## Primary Use Cases

**Parallel data acquisition**: Detector systems write frames to parallel processes simultaneously. VDS wraps outputs into coherent single file.

**Multi-instrument experiments**: Combine data from different sensors/instruments into single logical dataset.

**Dataset sharding**: Large datasets split across files for manageability, presented as single dataset.

## h5py API Basics

**Workflow**:
1. Create `VirtualLayout` with target shape and dtype
2. Create `VirtualSource` objects for each source dataset
3. Map source slices into layout using array indexing
4. Write with `Group.create_virtual_dataset()`

```python
import h5py
import numpy as np

# Stack four 1D datasets into 2D array
layout = h5py.VirtualLayout(shape=(4, 100), dtype='f')

for i in range(4):
    vsource = h5py.VirtualSource(f'source_{i}.h5', 'dataset', shape=(100,))
    layout[i] = vsource

# libver=('v110', 'v110') keeps the file readable by HDF5 1.10-1.14;
# on HDF5 2.x plain 'latest' selects the v200 format
with h5py.File('vds.h5', 'w', libver=('v110', 'v110')) as f:
    f.create_virtual_dataset('data', layout, fillvalue=0)
```

h5py 3.14+ can create a VDS with zero mappings from the high-level API (a dataset with virtual layout).

**VirtualSource creation**:
- From filename: `VirtualSource(filename, dataset_name, shape=shape)` - allows pre-defining before sources exist
- From dataset: `VirtualSource(dataset_object)` - extracts shape automatically

## Common Patterns

**Stacking**: Combine multiple datasets into higher-dimensional array. Stack four (100,) datasets → (4, 100) array.

**Concatenation**: Append datasets along existing axis. Concatenate three (10, 20) datasets → (30, 20) array.

**Printf-style mapping**: Map numbered datasets with single call using `%b` format specifier for block count substitution. Maps `dataset-0`, `dataset-1`, `dataset-2`... automatically.

**Unlimited dimensions**: Use `h5py.h5s.UNLIMITED` for dimensions that grow as sources resize.

```python
# Concatenation along new axis
layout = h5py.VirtualLayout(shape=(len(files),) + source_shape, dtype='f')
for i, filename in enumerate(files):
    vsource = h5py.VirtualSource(filename, 'data', shape=source_shape)
    layout[i, :, :] = vsource
```

## Requirements & Limitations

**Version**: Requires HDF5 1.10+. VDS files cannot open with older library versions. The default lower libver bound is 1.8 since HDF5 2.0, which is below what VDS needs, so set `libver` explicitly. With HDF5 2.x, `libver='latest'` selects the v200 format, and the shared-string VDS global-heap format (used only when the lower bound is 2.0) makes the file unreadable by 1.x; use `libver=('v110', 'v110')` (or a v110 upper bound) for files that older tools must read.

**Unlimited dimensions**: Must be slowest-changing dimension. If source has unlimited dimension, VDS must have same rank with same dimension unlimited.

**Source dataset limits**: Unlimited number of sources allowed, but individual source datasets have size limits.

**Read-only source issue**: If VDS file opened read/write but points to read-only sources, VDS returns fill values. Open VDS as read-only to access data from read-only sources.

**Metadata**: VDS does NOT inherit attributes from source datasets. All metadata must be explicitly attached to VDS itself.

**Tool compatibility**: Requires HDF5 1.10+ aware tools. Older tools cannot read VDS files.

## Performance Considerations

**Known issues**:
- VDS files can be unexpectedly large (master file larger than combined sources)
- Read performance can be slow with many mappings (orders of magnitude slower than direct access on older libraries)
- Inefficient "version 1/legacy" hyperslab encoding impacts performance on older libraries

**HDF5 2.0+ improvements**: VDS with many mappings is much faster (an r-tree for mapping searches is on by default; toggle it with `H5Pset_virtual_spatial_tree()` on the dataset access property list), opening is faster (delayed layout copy), and a new global-heap format stores source file and dataset names once, reducing size and close time. The shared-string format is used only when the lower libver bound is 2.0.

**When VDS works well**: Logical organization more important than performance, data already distributed, avoiding duplication critical.

**When to avoid VDS**:
- Small datasets with complex mappings (overhead dominates)
- Performance-critical applications requiring fast reads
- Backward compatibility needed (< HDF5 1.10)
- Very large individual source datasets
- Applications requiring rapid random access

**Recommendation**: Profile with representative data before committing to VDS for production use.

## Key Features

**Fill values**: Missing source files automatically display fill values instead of errors. Resilient to incomplete data.

**Gap handling**: `H5Pset_virtual_printf_gap(gap_size)` controls how many consecutive missing files tolerated before stopping. Default: 0 (stop at first gap).

**View options**:
- `H5D_VDS_FIRST_MISSING`: View includes only continuous data up to first gap
- `H5D_VDS_LAST_AVAILABLE`: View includes all available data regardless of gaps

**Same-file references**: Paths to sources in same file automatically changed to "." (current file reference).

**Transparent access**: Once created, VDS accessed like any other dataset. Applications don't need VDS-specific code.

## Best Practices

- **Use for logical unity without duplication** - Primary strength is avoiding data copies
- **Ensure consistent access modes** - All sources should have same read/write permissions as VDS file
- **Set appropriate fill values** - Choose fill values that indicate missing data clearly
- **Use printf-style mapping** for numbered file sequences - Single mapping call instead of many
- **Document mapping patterns** - VDS structure not always obvious from inspection
- **Test compatibility requirements** - Verify tools support HDF5 1.10+
- **Profile performance** - Measure with realistic access patterns before production use
- **Avoid for performance-critical paths** - Consider alternatives if speed critical
- **Keep source files accessible** - VDS requires sources available at access time
- **Use unlimited dimensions** for growing datasets - Allows VDS to grow as sources grow

## Common Patterns in Practice

**Time-series aggregation**: Files per day/hour, VDS provides continuous time axis across all files.

**Detector module assembly**: Each detector module writes to separate file, VDS assembles complete detector image.

**Multi-resolution data**: Different resolutions in separate files, VDS provides unified access to all resolutions.

**Experiment campaigns**: Multiple experimental runs in separate files, VDS combines for analysis.

## Gotchas

- **Writes go through to the sources**: VDS supports read and write; writes are forwarded to the mapped source datasets, and only mapped regions are written. Source files must be accessible and writable.
- **Source paths are persistent**: Moving source files breaks VDS unless paths updated.
- **Shape mismatches**: If a source's actual shape differs from its `VirtualSource` shape, reads give errors or fill values; check the mapping on a small example.
- **Access mode mismatch**: Opening VDS read/write with read-only sources returns fill values.
- **No lazy loading cache**: All source files opened simultaneously (no implemented caching).

## When NOT to Use VDS

- Frequent random access across many small files
- Source files change locations frequently
- Backward compatibility with HDF5 < 1.10 required
- Performance is critical constraint
- Simple concatenation better done with h5repack or custom script

## Additional Information

VDS feature introduced in HDF5 1.10. Original design document (RFC) available from HDF Group describes requirements, use cases, and implementation details. Some features from the original RFC remain unimplemented, including source dataset caching; HDF5 2.0 added a more compact (shared-string) storage encoding for mappings.

## Citations and References

- [HDF Group: Introduction to Virtual Datasets](https://support.hdfgroup.org/documentation/hdf5/latest/_v_d_s_t_n.html)
- [HDF Group: VDS Advanced Topics](https://support.hdfgroup.org/documentation/hdf5-docs/advanced_topics/intro_VDS.html)
- [HDF Group: VDS Documentation Archive](https://docs.hdfgroup.org/archive/support/HDF5/docNewFeatures/NewFeaturesVirtualDatasetDocs.html)
- [h5py: Virtual Datasets Documentation](https://docs.h5py.org/en/stable/vds.html)
- [h5py: Dataset Concatenation Example](https://github.com/h5py/h5py/blob/master/examples/dataset_concatenation.py)
- [HDF Group: Dataset Access Properties (gap handling, views)](https://docs.hdfgroup.org/hdf5/v1_14/group___d_a_p_l.html)
