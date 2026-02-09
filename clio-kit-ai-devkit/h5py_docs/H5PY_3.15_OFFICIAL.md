# h5py 3.15.1 Official Features

**Source:** Context7 documentation (/h5py/h5py)
**Date:** October 19, 2025
**Version:** 3.15.1 (Latest stable)

## New Features in 3.15.x

### 1. iter_chunks() - Efficient Chunk Iteration

**Method:** `Dataset.iter_chunks()`

```python
# Iterate over dataset chunks efficiently
for chunk_slice in dset.iter_chunks():
    arr = dset[chunk_slice]  # Get numpy array for chunk
    process(arr)
```

**Benefits:**
- Optimized iteration over chunked datasets
- No manual chunk calculation needed
- Follows HDF5's internal chunking

### 2. nbytes - Accurate Byte Size

**Attribute:** `Dataset.nbytes`

```python
# Get actual bytes required to load dataset into RAM
actual_bytes = dset.nbytes  # Accounts for compression, sparse storage
```

**Difference from size:**
- `dset.size` - Number of elements
- `dset.nbytes` - Actual bytes in RAM (more accurate)
- Does NOT include array overhead

### 3. Per-Dataset Chunk Cache Configuration

```python
# Configure chunk cache when creating dataset
dset = group.create_dataset(
    "data",
    shape=(10000, 10000),
    chunks=(100, 100),
    rdcc_nbytes=10*1024*1024,  # 10MB cache
    rdcc_w0=0.75,               # Write policy (0-1)
    rdcc_nslots=1009            # Hash table slots (prime number)
)

# Or when opening existing dataset
dset = group.require_dataset(
    "data",
    shape=(10000, 10000),
    dtype='f',
    rdcc_nbytes=5*1024*1024
)
```

**Parameters:**
- `rdcc_nbytes` - Cache size in bytes
- `rdcc_w0` - Write preference (0=LRU, 1=always cache writes)
- `rdcc_nslots` - Hash table size (use prime numbers)

### 4. astype() - Zero-Copy Type Conversion

```python
# Read with type conversion (no intermediate copy)
with dset.astype('float32'):
    data = dset[:]  # Read as float32 directly
```

**Benefits:**
- Single copy in memory (vs read then convert = 2 copies)
- Faster exports
- Lower memory usage

### 5. chunk_iter (Low-level API)

```python
# Low-level chunk iteration for advanced use
import h5py.h5d as h5d

def chunk_callback(chunk_info):
    print(f"Chunk at offset {chunk_info.offset}")
    print(f"Size: {chunk_info.size} bytes")
    return 0  # Continue iteration

dataset_id.chunk_iter(chunk_callback)
```

**Use case:** Performance profiling, chunk-level operations

## Core Dataset API

### Attributes

```python
dset.shape          # (100, 50) - dimensions
dset.dtype          # float64 - data type
dset.size           # 5000 - total elements
dset.nbytes         # 40000 - bytes in RAM (NEW in 3.x)
dset.ndim           # 2 - number of dimensions
dset.chunks         # (10, 50) - chunk shape or None
dset.compression    # 'gzip' or None
dset.compression_opts  # 9 (gzip level)
dset.fletcher32     # True/False - checksum enabled
dset.shuffle        # True/False - shuffle filter
dset.fillvalue      # Default value for uninitialized
```

### Methods

```python
# Reading
data = dset[:]              # Read all
data = dset[0:10, 0:5]      # Slice
data = dset[0, :]           # Single row

# Iteration (NEW)
for chunk in dset.iter_chunks():
    process(dset[chunk])

# Type conversion (NEW)
with dset.astype('float32'):
    data = dset[:]

# Attributes
attrs = dset.attrs          # Attribute manager
unit = dset.attrs['unit']   # Read attribute
```

## Compression Filters

### GZIP (Most common)

```python
dset = f.create_dataset(
    "compressed",
    data=np.random.rand(1000, 1000),
    compression="gzip",
    compression_opts=9  # 0-9, default 4
)
```

### Shuffle + GZIP (Best for scientific data)

```python
dset = f.create_dataset(
    "optimized",
    data=np.random.rand(1000, 1000),
    compression="gzip",
    compression_opts=6,
    shuffle=True  # Rearrange bytes for better compression
)
```

### Fletcher32 Checksum

```python
dset = f.create_dataset(
    "verified",
    data=np.random.rand(1000, 1000),
    fletcher32=True  # Detect corruption
)
```

## Performance Best Practices

1. **Always use chunking for large datasets**
2. **Enable shuffle filter before compression**
3. **Use iter_chunks() instead of manual iteration**
4. **Use nbytes for accurate memory estimation**
5. **Tune chunk cache for access patterns**
6. **Use astype() for type conversions during read**

---

**Version:** 3.15.1
**Python Support:** >= 3.8
**Latest Release:** October 16, 2025
