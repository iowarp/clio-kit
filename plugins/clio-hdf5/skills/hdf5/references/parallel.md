# Parallel HDF5

## Purpose

Parallel HDF5 enables multiple MPI processes to share and concurrently access HDF5 files, providing high-performance parallel I/O for large-scale scientific computing applications.

**Related**: `references/chunking.md` for optimal chunk layouts in parallel contexts, `references/filters.md` for compression with parallel I/O.

## How Parallel HDF5 Works

Parallel HDF5 is built on **MPI (Message Passing Interface)**, not threads. It allows processes within an MPI communicator to share open files and coordinate access. Under the hood, it leverages **MPI-IO**, which has been extensively optimized for various parallel filesystems and access patterns.

**Key concept**: All processes in the communicator can access all parts of the file simultaneously. Different processes typically read/write different portions of datasets using **hyperslabs** (rectangular slices of multi-dimensional arrays).

## Collective vs Independent Operations

HDF5 operations in parallel contexts fall into two fundamental classes:

**Collective Operations** (all processes must participate):
- Creating/deleting groups, datasets, attributes
- Opening/closing files
- Modifying any file structure or metadata
- **All processes must call with identical parameters**

**Independent Operations** (subsets of processes can participate):
- Data transfers (reads/writes) to existing datasets
- Individual processes can work on different regions
- Processes without data can pass NONE selection

**Critical rule**: Anything modifying file structure or metadata **must** be done collectively by all processes.

## Setup Functions

### File Access Configuration

```c
H5Pset_fapl_mpio(hid_t fapl_id, MPI_Comm comm, MPI_Info info)
```

Configures file access property list for parallel access. Pass MPI communicator and optional hints to the MPI-IO layer.

```python
# h5py example
from mpi4py import MPI
import h5py

comm = MPI.COMM_WORLD
f = h5py.File('parallel.h5', 'w', driver='mpio', comm=comm)
```

### Data Transfer Mode

```c
H5Pset_dxpl_mpio(hid_t dxpl_id, H5FD_mpio_xfer_t xfer_mode)
```

Controls whether data I/O is collective or independent:
- `H5FD_MPIO_COLLECTIVE`: Coordinated, potentially optimized by MPI-IO
- `H5FD_MPIO_INDEPENDENT`: Each process operates separately

```python
# h5py with collective I/O
dset = f.create_dataset('data', (1000, 1000))
with dset.collective:
    dset[rank*100:(rank+1)*100, :] = local_data  # Collective write
```

### Low-Level Optimization

```c
H5Pset_dxpl_mpio_collective_opt(...)
```

Controls whether collective operations at the HDF5 API level use collective or independent I/O at the MPI-IO layer. Can improve performance depending on access patterns.

## Typical Usage Pattern

```python
from mpi4py import MPI
import h5py
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 1. Open file with parallel driver (collective)
f = h5py.File('parallel_data.h5', 'w', driver='mpio', comm=comm)

# 2. Create dataset collectively (all processes)
total_size = size * 100
dset = f.create_dataset('shared_data', (total_size, 1000), dtype='f')

# 3. Each process writes its portion
local_data = np.random.random((100, 1000))
start = rank * 100
end = (rank + 1) * 100

# 4. Write with collective I/O
with dset.collective:
    dset[start:end, :] = local_data

# 5. Close (collective operation)
f.close()
```

## Key Constraints

**Type Conversion Breaks Collective I/O**: When data type conversion is required during transfer, HDF5 may fall back to independent I/O even when collective mode is requested, because constructing MPI-derived datatypes for conversions is not implemented.

**Identical Metadata Required**: When creating datasets collectively, all processes must provide identical parameters (shape, dtype, chunks, compression, etc.).

**Processes Without Data**: If some processes don't have data for a particular operation, they can still participate in collective mode by passing a NONE selection (empty hyperslab).

**Filtered Dataset Writes**: Writes to compressed/filtered datasets **must** use collective I/O (compression algorithms need coordinated access).

**Filtered Dataset Reads**: Reads from compressed/filtered datasets **can** use independent I/O and often perform better that way.

## Hyperslabs for Domain Decomposition

Hyperslabs are the primary mechanism for dividing work among processes:

```python
# 1D domain decomposition
local_size = total_size // size
start = rank * local_size
dset[start:start+local_size] = local_data

# 2D row decomposition
rows_per_process = total_rows // size
start_row = rank * rows_per_process
dset[start_row:start_row+rows_per_process, :] = local_data

# 2D block decomposition with arbitrary selection
dset = f['data']
dset[start_x:start_x+count_x, start_y:start_y+count_y] = local_block
```

## Filesystem Tuning

Parallel filesystem parameters significantly impact performance:

**Stripe Size**: Controls the size of contiguous data written to each storage target. Should align with typical I/O operation sizes.

**Stripe Count**: Number of storage targets (OSTs on Lustre, etc.) to stripe across. Balance between parallelism and metadata overhead.

**MPI Hints**: Pass filesystem-specific hints via `MPI_Info`:

```python
# C API pattern
MPI_Info_create(&info);
MPI_Info_set(info, "romio_cb_write", "enable");
MPI_Info_set(info, "striping_factor", "4");
```

Check your system's documentation for recommended settings. Common parameters include collective buffering, striping configuration, and alignment settings.

## Best Practices

- **Enable collective metadata operations** - Significant performance and scalability improvements in most cases
- **Use collective I/O for writes**, especially with compression or filters
- **Use independent I/O for reads** from filtered datasets - Often better performance than collective reads
- **Avoid type conversion** during parallel I/O to maintain collective mode
- **Align chunks with filesystem blocks** - Match chunk sizes to stripe sizes when possible
- **Design for contiguous access** per process - Each process reading/writing contiguous regions is faster
- **Pass MPI hints** appropriate for your filesystem via `MPI_Info` parameter
- **Profile and tune** stripe size/count for your workload and filesystem
- **Batch metadata operations** - Create all datasets collectively at initialization, not incrementally
- **Test at scale** - Parallel I/O performance can differ significantly between small and large process counts

## Performance Characteristics

**Collective I/O Advantages**:
- MPI-IO layer can optimize access patterns
- Two-phase I/O: processes redistribute data for optimal filesystem access
- Better performance for large transfers with good access patterns

**Independent I/O Advantages**:
- Lower coordination overhead
- Better for irregular access patterns
- Simpler for processes with vastly different data sizes
- Faster for filtered dataset reads

**Scalability Considerations**:
- Metadata operations scale poorly - minimize dataset creation overhead
- Collective operations have coordination cost
- Filesystem contention increases with process count
- Chunk cache becomes more important at scale

## Common Issues

**Deadlocks**: Mixing collective and independent operations incorrectly. Ensure all processes agree on operation mode.

**Poor Performance**: Often due to:
- Unaligned chunk sizes with filesystem blocks
- Type conversion forcing independent I/O
- Inadequate stripe count or stripe size
- Small I/O operations (batch into larger transfers)
- Collective reads of filtered data (use independent reads)

**Metadata Bottlenecks**: Creating many objects collectively scales poorly. Create file structure once, then perform data I/O.

**Compilation Issues**: Parallel HDF5 must be built with CMake (`-DHDF5_ENABLE_PARALLEL=ON`; Autotools was removed in 2.0) and linked against MPI; downstream CMake projects can check `HDF5_PROVIDES_PARALLEL`. h5py must be built against parallel-enabled HDF5; as of h5py 3.16 MPI builds require `mpi4py >= 4.0.0` on Python <= 3.12.

**Known problems (HDF5 2.x)**:
- Metadata cache images cannot be generated by parallel applications, and reading a file that has one is collective, so it can deadlock if a process does not participate.
- The `ph5diff` subsetting option fails; use serial `h5diff` for subsets.
- Before 2.2.0, parallel chunked datasets with fixed dimensions and no filters got the wrong ("none") chunk index type.
- `HDF5_ENABLE_CONCURRENCY` (CMake) enables multithreaded operation, is mutually exclusive with the threadsafe option, and is not an MPI feature.

## Citations and References

- [HDF5: A Brief Introduction to Parallel HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_intro_par_h_d_f5.html)
- [Parallel HDF5 — h5py documentation](https://docs.h5py.org/en/stable/mpi.html)
- [Introduction to Parallel HDF5 - HDF Group](https://confluence.hdfgroup.org/display/HDF5/Introduction+to+Parallel+HDF5)
- [Parallel I/O with HDF5 - The HDF Group](https://www.hdfgroup.org/2015/08/11/parallel-io-with-hdf5/)
- [HDF5: HDF5 Parallel Compression](https://support.hdfgroup.org/documentation/hdf5/latest/_par_compr.html)
