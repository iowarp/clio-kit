# HSDS and h5pyd

## Purpose

HSDS (Highly Scalable Data Service) is a cloud-native REST-based service for reading and writing HDF5 data. h5pyd is a Python client library that provides an h5py-compatible interface for accessing HSDS, allowing existing h5py code to work with cloud-hosted HDF5 data with minimal changes.

**Related**: `references/cloud_optimized.md` for file-level cloud optimization, `references/ros3_vfd.md` for direct S3 access without a service layer.

## What Problem HSDS Solves

Traditional HDF5 files on cloud object storage (S3, Azure Blob) require either downloading the entire file or using byte-range requests with limited efficiency. HSDS solves this by:

- Breaking HDF5 data into individual objects on cloud storage for granular access
- Providing a REST API that supports partial reads/writes without downloading whole files
- Enabling concurrent access from thousands of clients (multiple writers, multiple readers)
- Scaling horizontally by adding more service nodes
- Caching frequently accessed data for faster repeated queries

## Architecture Overview

HSDS runs as a set of cooperating service nodes, typically deployed as Docker containers:

**Service Node (SN)**: Handles client HTTP requests, authentication, and authorization. Routes data requests to the appropriate Data Node.

**Data Node (DN)**: Manages actual data storage and retrieval from the backend (S3, Azure, POSIX). Performs chunk-level I/O operations.

**Head Node**: (In multi-node deployments) Coordinates SN and DN nodes, manages cluster health.

```
Clients (h5pyd, REST API, curl)
        |
   [Service Nodes]  -- handle auth, routing, caching
        |
   [Data Nodes]     -- read/write objects on storage backend
        |
   [Object Storage]  -- S3, Azure Blob, or POSIX filesystem
```

In single-node deployments, SN and DN functionality runs in a single process.

## How HDF5 Data Maps to Cloud Objects

HSDS does **not** store data as monolithic `.h5` files. Instead, it decomposes HDF5 structures into individual objects:

- **Groups** are stored as separate JSON objects containing links and attributes
- **Datasets** are split into metadata objects (shape, dtype, chunk layout) and **chunk objects** (the actual data)
- **Committed datatypes** are stored as individual objects
- **Attributes** are stored inline with their parent group or dataset object

Each chunk of a dataset becomes its own object in the storage backend. This means:
- Reading a single chunk only fetches that one object (no need to read the whole dataset)
- Writing to one chunk doesn't affect other chunks
- Multiple clients can read/write different chunks concurrently
- Only modified chunks are updated on write

**Object key format** (simplified): Objects are organized under a "domain" path (analogous to a file path) with UUIDs identifying individual HDF5 objects and chunk indices.

## h5pyd: The Python Client

h5pyd mirrors the h5py API so that existing code requires minimal changes to work with HSDS.

### Installation

```bash
pip install h5pyd
```

### Initial Configuration

Run `hsconfigure` to store connection settings:

```bash
$ hsconfigure
Enter endpoint (e.g., http://localhost:5101 or https://hsdshdflab.hdfgroup.org):
Enter username:
Enter password:
Enter API key (or press enter for none):
```

This saves credentials to `~/.hscfg`. Alternatively, use environment variables.

### Basic Usage

```python
import h5pyd

# Open a domain (analogous to opening a file)
f = h5pyd.File("/home/user/mydata.h5", "r")

# Access datasets and groups exactly like h5py
dset = f["temperature"]
data = dset[0:100, 0:100]  # Partial read - only fetches needed chunks

# List contents
for name in f:
    print(name)

f.close()
```

### Creating and Writing Data

```python
import h5pyd
import numpy as np

f = h5pyd.File("/home/user/newfile.h5", "w")

# Create datasets
dset = f.create_dataset("sensor_data", shape=(10000, 256), dtype="f4",
                        chunks=(1000, 256))
dset[0:1000, :] = np.random.random((1000, 256))

# Create groups and attributes
grp = f.create_group("metadata")
grp.attrs["description"] = "Sensor readings from experiment 42"

f.close()
```

### Using Folders (Domains)

HSDS organizes data into "domains" (analogous to file paths) and "folders" (analogous to directories):

```python
# Create a folder
h5pyd.Folder("/home/user/project/", mode="w")

# List folder contents
folder = h5pyd.Folder("/home/user/project/")
for name in folder:
    print(name)
```

## Key API Differences: h5py vs h5pyd

While h5pyd aims for h5py compatibility, there are important differences:

| Feature | h5py | h5pyd |
|---------|------|-------|
| File paths | Local filesystem paths | Domain paths (e.g., `/home/user/data.h5`) |
| File constructor | `h5py.File("data.h5", "r")` | `h5pyd.File("/home/user/data.h5", "r")` |
| `driver` parameter | `core`, `mpio`, etc. | Not used (REST transport) |
| Folder listing | `os.listdir()` | `h5pyd.Folder()` |
| Concurrent writers | Requires SWMR or parallel HDF5 | Natively supported (MWMR) |
| Low-level API | Full `h5py.h5x` low-level access | Not available (no direct HDF5 C library calls) |
| VFDs | Supported | Not applicable |
| Compression | Applied client-side | Applied server-side; standard filters supported |
| File-level flush | `f.flush()` writes to disk | `f.flush()` ensures server persistence |
| Authentication | None (local files) | Username/password or API key |
| `Folder` class | Not available | `h5pyd.Folder` for domain management |

**Code migration tip**: In many cases, changing `import h5py` to `import h5pyd as h5py` and adjusting file paths to domain paths is sufficient for basic read/write workflows.

## Environment Variables

### HSDS Connection

| Variable | Purpose | Example |
|----------|---------|---------|
| `HS_ENDPOINT` | HSDS server URL, or `"local"` for direct mode | `http://localhost:5101` |
| `HS_USERNAME` | Authentication username | `test_user1` |
| `HS_PASSWORD` | Authentication password | `test` |
| `HS_API_KEY` | API key (alternative to user/pass) | `<key>` |
| `BUCKET_NAME` | Default storage bucket/container | `my-hdf5-bucket` |

### Cloud Backend Credentials

- **AWS S3**: Set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, and optionally `AWS_S3_GATEWAY` (for S3-compatible stores like MinIO)
- **Azure Blob**: Set `AZURE_CONNECTION_STRING`
- **POSIX/local**: Set `ROOT_DIR` to the parent directory for bucket folders

## Direct Mode (No Server)

h5pyd can access cloud storage directly without a running HSDS server:

```bash
export HS_ENDPOINT=local       # Single-process direct access
export HS_ENDPOINT=local[4]    # Multi-process (4 workers) for parallelism
```

Direct mode is useful for:
- Simple scripts that don't need multi-client access
- Batch processing jobs
- Environments where running a service is impractical

It still requires the appropriate cloud credentials (AWS, Azure) to be configured.

## Command-Line Tools

h5pyd includes utilities for managing HSDS domains:

| Tool | Purpose | Example |
|------|---------|---------|
| `hsconfigure` | Save connection credentials to `~/.hscfg` | `hsconfigure` |
| `hsinfo` | Show server status and domain info | `hsinfo /home/user/data.h5` |
| `hsls` | List domain/folder contents | `hsls -v /home/user/project/` |
| `hstouch` | Create new domains or folders | `hstouch /home/user/newfile.h5` |
| `hsload` | Import local HDF5 file to HSDS | `hsload data.h5 /home/user/data.h5` |
| `hsget` | Export HSDS domain to local HDF5 file | `hsget /home/user/data.h5 local.h5` |
| `hsrm` | Delete a domain or folder | `hsrm /home/user/old.h5` |
| `hscopy` | Copy a domain | `hscopy /home/user/src.h5 /home/user/dst.h5` |
| `hsmv` | Rename/move a domain | `hsmv /home/user/old.h5 /home/user/new.h5` |
| `hsdiff` | Compare local HDF5 file with HSDS domain | `hsdiff local.h5 /home/user/data.h5` |
| `hsacl` | Manage access control lists | `hsacl /home/user/data.h5` |

All tools accept `--help` for detailed usage. Use `--endpoint`, `--username`, `--password` flags to override `~/.hscfg` settings.

## Deployment

- **Single-node Docker**: `docker run -d -p 5101:5101 -e ROOT_DIR=/data -v /local/path:/data hdfgroup/hsds`
- **Multi-node Kubernetes**: HSDS provides Helm charts for production deployments with multiple SN/DN nodes. Scale by increasing replica counts.
- **Cloud-hosted**: The HDF Group operates a public instance at `https://hsdshdflab.hdfgroup.org` for testing.

## Multiple Writers / Multiple Readers (MWMR)

Unlike traditional HDF5 (which requires SWMR for concurrent access), HSDS natively supports multiple writers and multiple readers on the same domain simultaneously. The HSDS service coordinates access so data remains consistent. This is a major advantage for collaborative and real-time data workflows.

## Best Practices

- **Use chunked datasets** - HSDS stores each chunk as a separate object; chunking is essential for efficient partial reads
- **Choose chunk sizes carefully** - Too small creates excessive objects and overhead; too large wastes bandwidth on partial reads. 1-10 MiB per chunk is a good range.
- **Use `hsload` for bulk imports** - More efficient than writing data programmatically for initial data loading
- **Leverage direct mode for batch jobs** - Avoids the overhead of running a server for simple one-off scripts
- **Set `BUCKET_NAME`** - Avoid specifying the bucket on every call
- **Use folders to organize domains** - Treat HSDS domains like a filesystem hierarchy
- **Cache credentials with `hsconfigure`** - Avoid embedding credentials in scripts

## Common Pitfalls

- **Domain paths vs file paths**: HSDS uses `/home/user/...` domain paths, not local filesystem paths. Forgetting this causes confusing "not found" errors.
- **Missing `hstouch` for folders**: Folders must be explicitly created before creating domains inside them.
- **Chunk size mismatch on import**: `hsload` reuses each chunked dataset's chunk shape, and requests a single dataset-sized chunk for contiguous datasets. Re-chunk beforehand (e.g. with `h5repack -l`) if the source layout doesn't suit cloud access patterns.
- **Direct mode credential issues**: Direct mode still needs cloud credentials (AWS/Azure) even though no HSDS server is involved.
- **Low-level h5py API calls**: Code using `h5py.h5f`, `h5py.h5d`, or other low-level APIs will not work with h5pyd. Only the high-level API (`File`, `Dataset`, `Group`) is supported.

## Citations and References

- [HDF Group: HSDS Overview](https://www.hdfgroup.org/solutions/highly-scalable-data-service-hsds/)
- [GitHub: HDFGroup/hsds](https://github.com/HDFGroup/hsds)
- [GitHub: HDFGroup/h5pyd](https://github.com/HDFGroup/h5pyd)
- [NASA Earthdata: HSDS for Earth System Science](https://www.earthdata.nasa.gov/esds/competitive-programs/access/hsds)
- [HDF Group: Cloud Storage Options for HDF5](https://www.hdfgroup.org/2022/08/08/cloud-storage-options-for-hdf5/)
- [HDF in the Cloud (Matthew Rocklin)](https://matthewrocklin.com/blog/work/2018/02/06/hdf-in-the-cloud)
