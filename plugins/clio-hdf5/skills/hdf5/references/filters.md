# HDF5 Compression and Filters

## Purpose

Guidance for applying compression filters: choosing filters, configuring levels, combining filters, and understanding trade-offs.

**Related**: `references/chunking.md` for chunk layout (required for compression), `references/io.md` for write buffering.

**Key concept**: Filters operate on chunks only. Contiguous datasets cannot use filters.

## Available Filters

### GZIP (Deflate)
**Type**: Lossless | **Availability**: Depends on the build (in HDF5 2.0+ zlib support, CMake option `HDF5_ENABLE_ZLIB_SUPPORT`, defaults to OFF in self-built libraries; prebuilt packages normally enable it) | **Syntax**: `GZIP=N` (N=1-9)

Most widely supported compression; check with `h5z.filter_avail` (see below). Levels: 1 (fast, ~2-3x), 3 (balanced, ~4-5x, **recommended**), 6 (~5-6x), 9 (max ~5-7x, slow writes). Ratios are typical and data-dependent; benchmark on your own data.

```bash
h5repack -f GZIP=3 input.h5 output.h5  # Single dataset or all
```

### Shuffle Filter
**Type**: Preprocessing | **Availability**: Built into the library | **Syntax**: `SHUF`

Rearranges bytes to improve compression for numeric data. Often improves GZIP ratios substantially (data-dependent) with minimal CPU cost. **Must combine with compression filter**. Always pair with GZIP for numeric arrays.

```bash
h5repack -f SHUF -f GZIP=3 input.h5 output.h5
```

### SZIP
**Type**: Lossless | **Availability**: Optional at build time (provided by libaec) | **Syntax**: `SZIP=pixels,coding`

NASA-developed, faster than GZIP with similar ratio (~2-3x, data-dependent). Parameters: pixels_per_block (2-32, even, typically 8 or 16), coding (NN=nearest neighbor, EC=entropy coding). Optional at build time (`HDF5_ENABLE_SZIP_SUPPORT`, using libaec); check `h5z.filter_avail(h5z.FILTER_SZIP)`.

```bash
h5repack -f /dataset:SZIP=8,NN input.h5 output.h5
```

### LZF
**Type**: Lossless | **Availability**: h5py only | **Syntax**: Not in h5repack

Very fast, moderate compression (~2-3x, data-dependent). Not portable outside h5py. Use when speed paramount and h5py exclusive.

### Fletcher32
**Type**: Integrity check | **Availability**: Built into the library | **Syntax**: `FLET`

Adds 32-bit checksum per chunk to detect corruption. NOT compression. Small overhead (~5-10% performance, ~4 bytes/chunk). Use for critical data, archival, unreliable storage.

```bash
h5repack -f SHUF -f GZIP=3 -f FLET input.h5 output.h5
```

### N-Bit
**Type**: Lossless bit packing | **Availability**: Built into the library | **Syntax**: `NBIT`

Packs integer data to minimum bits (e.g., 12-bit sensor in 12 bits vs 16). Complex configuration, only for integers with known precision.

### Scale-Offset
**Type**: Lossy quantization | **Availability**: Built into the library | **Syntax**: `SOFF=factor,type`

Quantizes floats to specified precision. Parameters: scale_factor (decimal digits), scale_type (IN=integer, DS=decimal). Use only when precision loss acceptable.

```bash
h5repack -f /dataset:SOFF=2,DS input.h5 output.h5  # Keep 2 decimals
```

## Filter Combinations

Filters are applied in order and form a pipeline. **Order matters**.

### Recommended Combinations

**Standard compression (numeric data)**:
```bash
h5repack -f SHUF -f GZIP=3 input.h5 output.h5
```
- Best general-purpose option
- 4-5x compression
- Good read/write speed

**Fast compression**:
```bash
h5repack -f SHUF -f GZIP=1 input.h5 output.h5
```
- Fast writes and reads
- 2-3x compression
- Use for working datasets

**Maximum compression**:
```bash
h5repack -f SHUF -f GZIP=9 input.h5 output.h5
```
- 5-7x compression
- Slow writes, acceptable reads
- Use for archival data

**With data integrity**:
```bash
h5repack -f SHUF -f GZIP=3 -f FLET input.h5 output.h5
```
- Add Fletcher32 checksums for critical data

## Using h5repack for Filtering

### Basic Syntax

```bash
h5repack -i input.h5 -o output.h5 [filter options]
```

Or simplified:
```bash
h5repack [filter options] input.h5 output.h5
```

### Common Options

| Option | Purpose | Example |
|--------|---------|---------|
| `-f` | Apply filter | `-f /dset:GZIP=3` |
| `-f` (no path) | Apply to all datasets | `-f GZIP=3` |
| `-l` | Set chunks (see `references/chunking.md`) | `-l /dset:CHUNK=100x100` |

### Examples

**Apply compression to specific dataset**:
```bash
h5repack -f /data/raw:SHUF -f /data/raw:GZIP=3 input.h5 output.h5
```

**Apply to all datasets**:
```bash
h5repack -f SHUF -f GZIP=3 input.h5 output.h5
```

**Multiple datasets with different settings**:
```bash
h5repack \
  -f /data/raw:SHUF -f /data/raw:GZIP=1 \
  -f /data/processed:SHUF -f /data/processed:GZIP=6 \
  input.h5 output.h5
```

**Combine rechunking and compression**:
```bash
h5repack \
  -l /dataset:CHUNK=100x100 \
  -f /dataset:SHUF \
  -f /dataset:GZIP=3 \
  input.h5 output.h5
```

## Trade-offs and Recommendations

### Storage vs. Speed Table

Ratios are typical and data-dependent; benchmark on your own data.

| Priority | Filters | Ratio | Write Speed | Read Speed |
|----------|---------|-------|-------------|------------|
| Speed | GZIP=1 + SHUF | 2-3x | Fast | Fast |
| Balanced | GZIP=3 + SHUF | 4-5x | Good | Good |
| Storage | GZIP=6 + SHUF | 5-6x | Moderate | Good |
| Maximum | GZIP=9 + SHUF | 5-7x | Slow | Moderate |

## Checking Filter Availability

Use Python to check which filters are available:

```python
import h5py.h5z as h5z

filters = {
    'GZIP': h5z.filter_avail(h5z.FILTER_DEFLATE),
    'Shuffle': h5z.filter_avail(h5z.FILTER_SHUFFLE),
    'Fletcher32': h5z.filter_avail(h5z.FILTER_FLETCHER32),
    'SZIP': h5z.filter_avail(h5z.FILTER_SZIP),
    'N-bit': h5z.filter_avail(h5z.FILTER_NBIT),
    'Scale-offset': h5z.filter_avail(h5z.FILTER_SCALEOFFSET),
}

for name, available in filters.items():
    status = "Available" if available else "Not available"
    print(f"{name}: {status}")
```

Other 2.x / h5py 3.16 notes:

- Since HDF5 2.1 the library invokes a filter's `can_apply` and `set_local` callbacks for datasets with variable-length datatypes (previously skipped).
- `H5Dread_chunk()` gained an `nalloc` parameter in HDF5 2.0 (new signature, `H5Dread_chunk2`).
- Filtered-chunk sizes use 64-bit encoding when the lower library version bound is 2.0 or later.
- h5py `write_direct_chunk` now wraps `H5Dwrite_chunk` (h5py 3.14), and `Dataset.filter_ids` / `Dataset.filter_names` (h5py 3.16) report a dataset's filter pipeline.

### Plugins in 2.x

For third-party filters (Blosc, Zstd, LZF and similar):

- The default plugin path now includes `CMAKE_INSTALL_PREFIX` (`<prefix>/lib/plugin`), so `HDF5_PLUGIN_PATH` is often unnecessary.
- Libraries built with `-DHDF5_REQUIRE_SIGNED_PLUGINS=ON` (and OpenSSL) refuse unsigned plugins; sign plugins with `h5sign` and point `HDF5_PLUGIN_KEYSTORE` at the trusted-key directory.
- The library now checks that a filter's returned data size fits its returned buffer and that filtering then unfiltering restores the original size, so buggy third-party filters may now fail where they previously did not.

## Best Practices

1. **Default to GZIP=3 + Shuffle** for numeric data
2. **Test on representative data** before committing
3. **Measure before optimizing** - know your baseline
4. **Use h5repack** for existing files
5. **Document filter choices** in metadata
6. **Check portability** when sharing files
7. **Don't over-optimize** - GZIP=3 vs GZIP=9 rarely matters
8. **Combine with chunking** optimization (see `references/chunking.md`)
