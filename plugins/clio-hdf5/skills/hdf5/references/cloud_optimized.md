# Cloud-Optimized HDF5

## Purpose

Cloud-optimized HDF5 is a pattern for writing regular HDF5 files that perform efficiently when accessed from cloud object storage (S3 and S3-compatible stores via the ROS3 VFD; other object stores via HSDS or an fsspec-style layer) via HTTP byte-range requests.

**Related**: `references/chunking.md` for chunk size configuration, `references/filters.md` for compression strategies.

## What Cloud-Optimized HDF5 Is

**It's a pattern, not a format**: Cloud-optimized HDF5 files are regular `.h5` files written with specific strategies that enable efficient remote access. No special format or conversion required.

**Recent advances** (HDF5 2.0 and 2.2): ROS3 was rewritten on aws-c-s3 (`s3://` URIs, automatic credential and region discovery, parallel range requests, retries), and 2.2 added an I/O block cache. Properly configured HDF5 files achieve performance on par with cloud-native formats like Zarr without reformatting data or generating metadata sidecar files (Kerchunk, DMR++).

**Core challenge**: Traditional HDF5 was designed for local filesystem I/O with low-latency random access. Cloud object storage has high latency (10-100ms per request) but good bandwidth. Optimization focuses on minimizing request count rather than maximizing transfer speed.

## Key Concepts

**Byte-Range Requests**: Cloud object storage doesn't support POSIX file operations. Instead, clients make HTTP GET requests with byte-range headers to fetch specific portions of files. Each request has latency overhead.

**The Metadata Problem**: Default HDF5 scatters metadata in small blocks (often 4KB) throughout files. On cloud storage, this requires dozens or hundreds of separate HTTP requests just to locate data before accessing it.

**Latency vs Bandwidth Trade-off**:
- Local disk: Low latency (~0.1ms), moderate bandwidth
- Cloud storage: High latency (10-100ms), high bandwidth
- Strategy: Reduce request count (minimize latency hits), accept larger transfers

## Core Optimization Patterns

### 1. Consolidated Metadata (Paged Aggregation)

**Use paged aggregation file space management strategy** to consolidate file-level metadata into page-aligned metadata pages, so it can be fetched in few requests.

```python
import h5py

# Create file with paged aggregation
f = h5py.File('cloud_optimized.h5', 'w',
              fs_strategy='page',
              fs_page_size=1024*1024)  # 1 MB pages (NOT default 4KB!)

# Create datasets as normal
dset = f.create_dataset('data', (10000, 10000), chunks=(100, 10000), dtype='f4')
f.close()
```

**Why this matters**: Metadata consolidated in initial pages = few requests to find data = much faster access.

**Page size**: Default 4KB defeats the purpose. Use MB-scale pages (1-10 MB) to truly consolidate metadata.

### 2. Larger Chunk Sizes (10-100 MB)

**Target 10-100 MB chunks** for cloud storage, much larger than typical local disk chunks (1-10 MB).

```python
import h5py
import numpy as np

# Cloud-optimized chunking: larger chunks
with h5py.File('cloud_data.h5', 'w') as f:
    # Example: 1000 x 10000 float32 = 40 MB per chunk
    dset = f.create_dataset(
        'measurements',
        shape=(100000, 10000),
        chunks=(1000, 10000),  # 40 MB chunks
        dtype='f4',
        compression='gzip',
        compression_opts=3
    )
```

**Sizing guidelines**:
- Minimum: 1 MB (sub-MB chunks cause severe performance degradation)
- Target: 10-100 MB depending on dataset size and access patterns
- AWS recommendation: 8-16 MB for optimal byte-range requests
- Pangeo recommendation: ~100 MB for high-performance scientific workloads

**Trade-offs**:
- Larger chunks = fewer requests but more bandwidth per access
- Must balance chunk size with typical read sizes
- Consider memory constraints on client side
- Chunks larger than `rdcc_nbytes` are not cached: set `rdcc_nbytes` to at least one chunk (ideally several) when reading 10-100 MB chunks (default 1 MiB on HDF5 1.14, 8 MiB on 2.0+). HDF5 2.1+ mitigates the uncached case with the data sieve buffer

### 3. Align Chunk Shape with Access Patterns

Design chunk dimensions to match how data will be read:

```python
# Time-series data: read all variables for time ranges
# Chunk along time dimension
time_series = f.create_dataset(
    'timeseries',
    shape=(100000, 50),  # 100k timesteps, 50 variables
    chunks=(10000, 50),  # Read 10k timesteps at once
    dtype='f4'
)

# Spatial data: read spatial slices at specific times
# Chunk along spatial dimensions
spatial = f.create_dataset(
    'spatial',
    shape=(1000, 1000, 1000),  # time, lat, lon
    chunks=(1, 1000, 1000),    # Single time slice
    dtype='f4'
)
```

**Wrong chunk shape** = every read touches many chunks = wasted bandwidth and excessive requests.

## h5py API for Cloud-Optimized Files

### Basic Pattern

```python
import h5py

# Step 1: Create file with paged aggregation
f = h5py.File('output.h5', 'w',
              fs_strategy='page',
              fs_page_size=4*1024*1024)  # 4 MB pages

# Step 2: Create datasets with appropriate chunking
f.create_dataset(
    'data',
    shape=(50000, 5000),
    chunks=(1000, 5000),  # ~20 MB chunks for float32
    dtype='f4',
    compression='gzip',
    compression_opts=3
)

f.close()
```

### Reading from Cloud Storage

```python
# Using ros3 VFD (read-only S3 access)
# `s3://` URLs are native from HDF5 2.0; on 1.14 use https:// URLs.
# Check that this h5py build has the driver: 'ros3' in h5py.registered_drivers()
import h5py

# Access public S3 bucket
f = h5py.File(
    's3://bucket-name/path/to/file.h5',
    'r',
    driver='ros3',
    aws_region=b'us-west-2'
)

# Access with credentials (on HDF5 2.0+ you can omit them to use the AWS
# credential chain: env vars, ~/.aws files, STS, EC2 metadata)
f = h5py.File(
    's3://bucket-name/path/to/file.h5',
    'r',
    driver='ros3',
    aws_region=b'us-west-2',
    secret_id=b'ACCESS_KEY_ID',
    secret_key=b'SECRET_ACCESS_KEY'
)

# Read data (each chunk access = one HTTP request)
data = f['dataset'][1000:2000, :]
f.close()
```

## Performance Characteristics

**Well-Optimized Files**:
- Few metadata requests (1-10) due to consolidated metadata
- Predictable data access: one HTTP request per chunk touched
- Performance comparable to Zarr and other cloud-native formats

**Poorly-Optimized Files**:
- Hundreds of small metadata requests before data access begins
- Many requests for small chunks (<1 MB)
- Can be 10-100x slower than optimized files

**Comparison to Local Storage**:
- Local: Random access cheap, many small reads acceptable
- Cloud: Each access costs latency penalty, must batch into larger operations

## Best Practices

- **Always use paged aggregation** for cloud-destined files with MB-scale page size
- **Target 10-100 MB chunks** depending on dataset size and typical access patterns
- **Align chunk shape** with expected read patterns (row-major vs column-major, time slices vs spatial slices)
- **Use moderate compression** (gzip level 3-5) to balance transfer size with decompression overhead
- **Test read patterns** from cloud storage before finalizing chunk strategy
- **Document intended access patterns** so users can optimize their reads
- **Avoid variable-length datatypes** (vlen strings, ragged arrays) - they don't work efficiently with byte-range requests
- **Consider total file size**: Very small files (<100 MB) may not benefit from cloud optimization
- **Profile request count**: Monitor actual HTTP requests during reads to validate optimization

## Major Pitfalls

**Small Chunks (<1 MB)**: Creates excessive HTTP requests. Each request has 10-100ms latency. Chunks <1 MB can reduce throughput to unusable levels for parallel processing frameworks.

**Forgetting Paged Aggregation**: Files without paged aggregation require dozens to hundreds of metadata requests before data access begins. Can add seconds of latency.

**Wrong Page Size**: Using default 4KB page size defeats paged aggregation's purpose. Must explicitly set MB-scale pages.

**Misaligned Chunk Shape**: If chunks don't align with access patterns, every read touches many chunks. Reading one row when chunks are column-oriented wastes 10-100x bandwidth.

**Many Small Reads**: Application code that reads element-by-element or in small slices creates request storms. Batch reads into chunk-aligned operations.

**Caching is limited**: ROS3 caches the first 16 MiB of the file. With paged aggregation on HDF5 2.0+ it also sets a 64 MiB page buffer, and on 2.2+ files without paged allocation get an LRU block cache (16 MiB blocks, up to 128 MiB). Older versions issue roughly one HTTP request per library read outside the initial cache, so keep reads chunk-aligned.

**Special characters in keys**: before HDF5 2.2.0, object keys containing `=`, `+` or spaces (e.g. Hive-style `key=value` partition paths) failed with HTTP 403; upgrade to 2.2.0 and pass keys unencoded. For non-AWS S3-compatible stores, use `H5Pset_fapl_ros3_endpoint()` or the `AWS_ENDPOINT_URL_S3` environment variable (HDF5 2.0+).

**Variable-Length Datatypes**: Vlen strings and datatypes don't work efficiently with HTTP range requests. Use fixed-length alternatives.

**Over-Compression**: Heavy compression (gzip level 9) increases decompression time without much size reduction. Level 3-5 usually optimal.

**Ignoring Access Patterns**: Writing chunks for convenience rather than expected read patterns. Always design chunking for how data will be consumed.

## Citations and References

- [HDF5 in the Cloud - The HDF Group](https://www.hdfgroup.org/solutions/hdf5-in-the-cloud/)
- [Cloud-Optimized HDF/NetCDF – Cloud-Optimized Geospatial Formats Guide](https://guide.cloudnativegeo.org/cloud-optimized-netcdf4-hdf5/)
- [Cloud Storage Options for HDF5 - The HDF Group](https://www.hdfgroup.org/2022/08/08/cloud-storage-options-for-hdf5/)
- [Evaluating Cloud-Optimized HDF5 for NASA's ICESat-2 Mission](https://nsidc.github.io/cloud-optimized-icesat2/)
- [HDF in the Cloud - Matthew Rocklin](https://matthewrocklin.com/blog/work/2018/02/06/hdf-in-the-cloud)
- [Optimization Practices - Chunking (ESIP)](https://esipfed.github.io/cloud-computing-cluster/optimization-practices.html)
