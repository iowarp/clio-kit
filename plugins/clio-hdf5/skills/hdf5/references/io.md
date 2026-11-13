# HDF5 I/O Optimization

## Purpose

Guidance for optimizing HDF5 I/O performance through runtime strategies: write buffering, buffer size tuning, and access patterns.

**Related**: `references/chunking.md` for chunk layout and cache, `references/filters.md` for compression.

## Write Performance

### Avoid Small Writes

**Problem**: Element-by-element writes are extremely slow.
**Solution**: Batch writes into larger operations.

```python
# Bad: Many small writes
for i in range(10000):
    dset[i, :] = compute_row(i)  # 10,000 separate I/O operations!

# Good: Batch writes
batch_size = 1000
for i in range(0, 10000, batch_size):
    batch = np.array([compute_row(j) for j in range(i, i+batch_size)])
    dset[i:i+batch_size, :] = batch  # Far fewer I/O operations

# Best: Single write if possible
data = np.array([compute_row(i) for i in range(10000)])
dset[:] = data
```

### Application-Level Buffering

Buffer writes in memory, flush periodically:

```python
write_buffer = []
buffer_size = 1000

with h5py.File('data.h5', 'a') as f:
    dset = f['/dataset']
    for i in range(10000):
        write_buffer.append(compute_row(i))
        if len(write_buffer) >= buffer_size:
            start = i - len(write_buffer) + 1
            dset[start:i+1, :] = np.array(write_buffer)
            write_buffer = []
    # Flush remaining
    if write_buffer:
        dset[-len(write_buffer):, :] = np.array(write_buffer)
```

### Chunk Cache for Incremental Writes

Increase cache when making many small updates (see `references/chunking.md` for details):

```python
f = h5py.File('data.h5', 'a', rdcc_nbytes=20*1024**2)  # 20 MB cache
```

## h5repack Buffer Size

Control memory usage during h5repack:

```bash
# Default hyperslab buffer: 32 MB
h5repack -f GZIP=3 input.h5 output.h5

# Larger buffer for big datasets. The value is a whole number of MB; it
# also raises the tools' 256 MB single-allocation cap if it is larger.
export H5TOOLS_BUFSIZE=128
h5repack -f GZIP=3 large_file.h5 output.h5
```

Increase for large files (> 1 GB) when system memory allows.

## Read Performance

**Sequential access** (entire dataset): Use contiguous storage (see `references/chunking.md`), read large blocks.

**Random access** (scattered elements): Use chunked storage aligned with pattern (see `references/chunking.md`), increase chunk cache.

```python
# Random access with large cache
f = h5py.File('data.h5', 'r', rdcc_nbytes=50*1024**2, rdcc_nslots=10007)
```

**Batch reads when possible**: Use fancy indexing `dset[indices]` instead of looping (indices must be increasing; h5py 3.16 made selecting many indices much faster).

`rdcc_nslots` defaults to 8191 on HDF5 2.0+ (521 earlier): raise it to a prime well above the number of chunks kept in cache, never below the default.

## File Management

Use context managers for automatic cleanup:

```python
with h5py.File('data.h5', 'a') as f:
    dset = f['/dataset']
    dset[:] = new_data
    f.flush()  # Manual flush if needed during long operations
# File is flushed and closed on exit
```

## 2.x / h5py 3.16 Notes

- HDF5 2.1 extends the data sieve buffer to chunks that do not fit in the chunk cache, so a too-small `rdcc_nbytes` no longer degrades to element-at-a-time I/O (a cache of at least one chunk is still preferred).
- h5py 3.13 opens existing datasets faster; h5py 3.16 indexes with many indices much faster.
- HDF5 2.2.0 speeds up h5dump, h5ls, h5diff, h5repack, h5stat and h5format_convert on files where many objects are linked by multiple hard links.

## Best Practices

- **Batch writes** into large blocks, avoid element-by-element
- **Buffer incremental writes** in application code
- **Use context managers** (`with` statements) for automatic cleanup
- **Increase h5repack buffer** for large files (> 1 GB)
- **Profile before optimizing** - measure actual bottleneck
- Combine with chunking (see `references/chunking.md`) and compression (see `references/filters.md`)

## Common Patterns

**Incremental dataset building**:
```python
with h5py.File('data.h5', 'w') as f:
    dset = f.create_dataset('data', shape=(0, 1000), maxshape=(None, 1000), chunks=(100, 1000))
    batch_size, batch = 100, []
    for i in range(10000):
        batch.append(compute_row(i))
        if len(batch) >= batch_size:
            dset.resize(dset.shape[0] + batch_size, axis=0)
            dset[-batch_size:, :] = np.array(batch)
            batch = []
```

**h5repack with large buffer**:
```bash
export H5TOOLS_BUFSIZE=128   # MB
h5repack -f SHUF -f GZIP=3 large_input.h5 compressed.h5
```

## Common Issues

- **Slow writes**: Batch operations, buffer in application
- **Slow reads**: Increase chunk cache (see `references/chunking.md`), check alignment
- **Using h5repack on large files**: Increase H5TOOLS_BUFSIZE

## When NOT to Optimize

Skip optimization for: small files (< 10 MB), temporary files, one-time operations, or when I/O is not the bottleneck.
