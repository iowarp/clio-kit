# HDF5 Core VFD (Memory File Driver)

## Purpose

The Core Virtual File Driver (VFD) maintains HDF5 files entirely in RAM rather than on disk, eliminating file system I/O overhead for temporary data operations and high-performance scenarios.

**Related**: `references/chunking.md` for chunk configuration, `references/filters.md` for compression.

## What Problem Core VFD Solves

Traditional HDF5 operations involve file system I/O with associated overhead from disk latency, buffering, and system calls. Core VFD keeps the entire file in memory, providing direct RAM access with optional persistence to disk on close. This enables fast temporary file operations, testing without disk I/O, and delayed storage decisions.

## Primary Use Cases

**I/O buffer applications**: Large numbers of small, random-access I/O operations benefit from eliminating disk overhead.

**Temporary/scratch space**: Create and manipulate HDF5 files without persistent storage. Useful for intermediate computations, data transformations, or algorithm testing.

**Baseline performance testing**: Isolate computational performance from disk I/O by running tests entirely in memory.

**Delayed storage decisions**: Maintain "shadow" files in memory while determining which data to persist and in what format. Postpone layout decisions (compact vs chunked) until final dataset size is known.

**Fast prototyping and testing**: Develop and test HDF5 code without creating disk files. Useful for unit tests and development workflows.

**Network transmission**: Create HDF5 file in memory, extract as bytes, transmit over network, reconstruct on receiving end.

## h5py API Usage

### Basic In-Memory File

```python
import h5py

# Create purely in-memory file (no disk persistence)
f = h5py.File('memory.h5', 'w', driver='core', backing_store=False)
f['dataset'] = [1, 2, 3, 4, 5]
data = f['dataset'][:]
f.close()  # Data discarded, no file written
```

### With Backing Store (Persist on Close)

```python
# Create in-memory file, write to disk on close
f = h5py.File('data.h5', 'w', driver='core', backing_store=True)
f['dataset'] = [1, 2, 3, 4, 5]
f.close()  # Writes data.h5 to disk
```

### Load Existing File into Memory

```python
# Read entire file into memory for fast access
f = h5py.File('large_data.h5', 'r', driver='core')
# All subsequent reads from memory, not disk
data = f['dataset'][:]
```

### Convenient In-Memory Constructor

```python
# Recommended approach for purely in-memory files (h5py >= 3.13;
# on older h5py use driver='core', backing_store=False)
f = h5py.File.in_memory()
f['data'] = [1, 2, 3]
# Likely to cause fewer issues than BytesIO objects
```

## Key Parameters

**backing_store** (boolean, default True):
- True: Changes written to disk on `close()` or `flush()`
- False: Changes discarded when file closed, no disk I/O

**block_size / increment** (integer, default 64KB):
- Memory allocation increment in bytes
- File buffer grows by this amount when more space needed
- Larger values reduce reallocation overhead for growing files
- Example: `driver='core', backing_store=False, block_size=1024*1024` (1 MB increments)

## Write Tracking (C API)

When using Core VFD with backing store enabled, write tracking optimizes disk writes by recording only modified regions.

**H5Pset_core_write_tracking**: Enable tracking of write operations to minimize disk I/O.

**Parameters**:
- `is_enabled`: Activates write tracking
- `page_size`: Size in bytes for aggregation (should be power of two)

**Benefit**: Only modified bytes written to disk, not entire file. Significant savings for large files with small changes.

## Memory Management

**Allocation behavior**: Files grow in specified increments. A file with 24 bytes of data might allocate 1 MB if increment is 1 MB.

**Automatic trimming**: HDF5 library removes unused space before closing file. In-memory file may be much larger during operation than persisted file.

**Memory limit**: Total file size limited by available system RAM. No explicit maximum besides memory constraints.

## Best Practices

- **Use `backing_store=False` for temporary data** - Avoid unnecessary disk I/O for ephemeral files
- **Set appropriate `block_size`** - Larger increments (1-10 MB) for files expected to grow significantly
- **Prefer `File.in_memory()`** (h5py >= 3.13) for purely in-memory files - Cleaner API, fewer edge cases; on older h5py use `driver='core', backing_store=False`
- **Use for testing and development** - Fast, no disk cleanup needed, isolated from file system state
- **Profile memory usage** - Monitor RAM consumption for large in-memory files
- **Consider write tracking** - Enable for backing store with selective modifications
- **Load read-only files** into memory for repeated access - Faster than repeated disk reads

## Performance Characteristics

**Advantages**:
- No disk I/O latency for read/write operations
- Fast random access patterns
- Eliminates file system overhead
- Useful for baseline performance measurement

**Overhead**:
- Initial file load reads entire file into memory (for existing files)
- Memory allocation/reallocation as file grows
- Memory consumption can be significant for large files

## Limitations

**Memory constraints**: File size limited by available RAM. Not suitable for datasets larger than system memory.

**No persistence without backing store**: Data lost when file closed if `backing_store=False`.

**Single process**: Core VFD not designed for concurrent access by multiple processes (use appropriate parallel VFD instead).

**Entire file in memory**: Even small accesses require full file loaded. Not suitable for partial access to very large files.

## When to Use Core VFD

- Temporary files during computation pipelines
- Testing and development (especially unit tests)
- Small to medium files requiring very fast access
- Files with many small random-access operations
- Baseline performance benchmarking
- Prototyping before implementing persistent storage
- Network transmission of HDF5 data

## When NOT to Use Core VFD

- Files larger than available system memory
- Need persistent storage without explicit backing store management
- Concurrent access by multiple processes required
- Long-running applications where memory is constrained
- Large files where only small portions accessed (use disk with chunking instead)
- Production systems with limited RAM

## Common Patterns

**Testing pattern**:
```python
def test_hdf5_operations():
    with h5py.File.in_memory() as f:
        f['test'] = [1, 2, 3]
        assert f['test'][0] == 1
    # No cleanup needed, file discarded
```

**Selective persistence pattern**:
```python
# Work in memory, decide what to save
mem_file = h5py.File.in_memory()
mem_file['candidate1'] = data1
mem_file['candidate2'] = data2

# Later: save only selected datasets to disk
with h5py.File('final.h5', 'w') as disk_file:
    mem_file.copy('candidate1', disk_file)
```

**Fast repeated access**:
```python
# Load once, access many times
f = h5py.File('data.h5', 'r', driver='core')
for i in range(1000):
    result = process(f['dataset'][:])  # Fast memory access
f.close()
```

## Gotchas

- **Memory over-allocation**: Files may allocate more memory than data size due to increment setting
- **Forgotten backing_store**: Setting `backing_store=True` without realizing data will be written to disk
- **Large file OOM**: Loading large existing file into memory can exhaust RAM
- **No partial loading**: Cannot selectively load parts of file; entire file enters memory
- **Version-specific bugs**: Historical issues with h5py core driver in some versions

## Citations and References

- [HDF Group: Virtual File Layer Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/_v_f_l_t_n.html)
- [HDF Group: File Access Properties (H5Pset_fapl_core)](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html)
- [HDF Group: HDF5 File Image Operations](https://support.hdfgroup.org/documentation/hdf5/latest/_f_i_l_e_i_m_g_o_p_s.html)
- [h5py: File Objects (core driver documentation)](https://docs.h5py.org/en/stable/high/file.html)
- [GitHub: Literate HDF5 - Core VFD Examples](https://github.com/gheber/literate-hdf5/blob/master/core-vfd.org)
- [PyTables: In-Memory HDF5 Files](https://www.pytables.org/cookbook/inmemory_hdf5_files.html)
- [HDF Group: Registered Virtual File Drivers](https://support.hdfgroup.org/documentation/hdf5-docs/registered_virtual_file_drivers_vfds.html)
