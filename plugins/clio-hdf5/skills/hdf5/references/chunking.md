# HDF5 Chunking Strategies

## Purpose

Guidance for configuring HDF5 dataset chunking: choosing storage layouts, chunk sizes, chunk shapes, and chunk cache settings.

**Related**: `references/filters.md` for compression, `references/io.md` for write buffering.

## Storage Layouts

### Contiguous Storage
Data stored in single continuous block. Use when: dataset small (< 100 MB), always read entirely, fixed size, no compression needed. Convert with: `h5repack -l /dataset:CONTI input.h5 output.h5`

### Chunked Storage
Data divided into fixed-size chunks. Required for: compression, extensibility, partial access. Has metadata overhead.

## Chunk Size Guidelines

**Target: 10 KB to 1 MB per chunk**

- < 10 KB: Excessive overhead, metadata bloat
- 10-100 KB: Good for random access
- 100 KB-1 MB: Good for sequential access
- \> 1 MB: Wastes bandwidth, memory pressure

Calculate: `chunk_shape[0] × chunk_shape[1] × ... × dtype_size_bytes`

**Default to 100 KB chunks** when access pattern unknown.

Since HDF5 2.0, chunks of 4 GiB or more are allowed, but a file using them cannot be opened by older libraries.

## Chunk Shape: Align with Access Patterns

**Critical principle**: Match chunk shape to how data is accessed.

- **Row-wise** (`data[i, :]`): Row-oriented chunks `(1, 1000)` - NOT `(1000, 1)`
- **Column-wise** (`data[:, j]`): Column-oriented chunks `(1000, 1)` - NOT `(1, 1000)`
- **Block access** (`data[0:100, 0:100]`): Square chunks `(100, 100)`
- **3D slices** (along axis 0): Thin chunks `(1, 100, 100)`
- **3D volumes**: Balanced cubic `(10, 50, 50)`
- **Unknown**: Balanced dimensions proportional to dataset shape

## Rechunking with h5repack

```bash
# Basic rechunking
h5repack -l /dataset:CHUNK=100x200 input.h5 output.h5

# Multiple datasets
h5repack -l /raw:CHUNK=100x1x1 -l /grid:CHUNK=1x100x100 input.h5 output.h5

# With compression (see hdf5_filters)
h5repack -l /dataset:CHUNK=100x100 -f /dataset:SHUF -f /dataset:GZIP=3 input.h5 output.h5
```

## Chunk Cache Configuration

Configure when opening file for repeated/random access:

```python
import h5py

f = h5py.File('data.h5', 'r',
              rdcc_nbytes=10*1024**2,   # Cache size: 10 MB (default 1 MiB on HDF5 1.14, 8 MiB on 2.0+)
              rdcc_nslots=10009,        # Hash slots: prime number (default 521 on 1.14, 8191 on 2.0+)
              rdcc_w0=0.75)             # Preemption: 0.0-1.0 (default 0.75)
```

Defaults depend on the HDF5 library the h5py wheel wraps (h5py 3.16 wheels bundle HDF5 2.0; earlier wheels bundle 1.14). Print `f.id.get_access_plist().get_cache()` to see the values in effect.

**rdcc_nbytes**: Cache size. Increase for random access, repeated reads. Rule of thumb: 10-50× chunk size.

**rdcc_nslots**: Hash table slots. Use prime number > expected chunks accessed. Common: 1009, 5003, 10009.

**rdcc_w0**: Preemption policy. 0.0 = pure LRU; 1.0 = evict chunks that were fully read or written first; 0.75 is the library default and a good start. Chunks larger than the cache cannot be cached; since HDF5 2.1 those go through the data sieve buffer instead of degrading to element-at-a-time I/O, but a cache at least one chunk in size is still preferred.

## Best Practices

- Align chunk shape with access pattern
- Target 10 KB - 1 MB per chunk
- Use contiguous storage for small datasets read entirely
- Increase cache for random access (10-50× chunk size)
- Use prime numbers for rdcc_nslots
- Test with realistic access patterns
- HDF5 2.0 changed the default lower libver bound to 1.8; set `libver` explicitly (e.g. `libver='latest'`, or a v110 bound) if you need newer chunk index types
- h5py 3.12+ takes the fill time for chunked datasets from HDF5 (`ifset`) rather than forcing `alloc`, which lowers first-write cost
- Document chunk choices and rationale

## Common Pitfalls

- Misaligned chunks (row chunks for column access)
- Too-small chunks (< 10 KB): metadata bloat
- Too-large chunks (> 1 MB): wastes bandwidth
- Insufficient cache for random access
