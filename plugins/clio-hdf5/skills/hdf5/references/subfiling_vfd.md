# HDF5 Subfiling VFD

## Purpose

The Subfiling Virtual File Driver (VFD) distributes a single logical HDF5 file across multiple physical "subfiles" on disk, using designated I/O concentrator processes to manage striped data access. This reduces file system contention and improves parallel I/O scalability on systems like Lustre, GPFS, and other parallel file systems.

**Related**: `references/parallel.md` for general parallel HDF5 patterns, `references/chunking.md` for chunk layout optimization.

## What Problem Subfiling Solves

In large-scale parallel HDF5 applications, all MPI processes writing to a single shared file creates contention at the file system level. The subfiling VFD addresses this by:

- Splitting the logical file into multiple subfiles, each managed by a dedicated I/O concentrator process
- Striping data across subfiles in fixed-size segments to distribute load
- Reducing the number of processes that directly perform file I/O (concentrators act as intermediaries)
- Enabling better utilization of parallel file system bandwidth

## When to Use Subfiling

- Large-scale MPI applications (hundreds to thousands of processes) on parallel file systems
- Workloads where single-file parallel I/O is a bottleneck
- Systems with Lustre, GPFS, or similar parallel storage where file-level contention limits throughput
- Situations where the standard MPIO driver doesn't scale well enough

## When NOT to Use Subfiling

- Serial or small-scale parallel applications (overhead outweighs benefit)
- Applications that don't use MPI
- When file portability is critical (subfiles must be recombined before sharing)
- Systems without a parallel file system

## Requirements

**MPI Thread Support**: The subfiling VFD requires `MPI_THREAD_MULTIPLE` support. Your application must initialize MPI with:

```c
int provided;
MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
if (provided < MPI_THREAD_MULTIPLE) {
    /* Subfiling will not work - handle error */
}
```

**HDF5 Version**: Available since HDF5 1.14.0.

**Build Requirements**: HDF5 must be built with parallel support using CMake (`-DHDF5_ENABLE_PARALLEL=ON`; Autotools was removed in 2.0). Check the build's `libhdf5.settings` for subfiling support.

## C API Usage

### Basic Setup

```c
#include "hdf5.h"

hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);

/* Use subfiling with default configuration:
   - One I/O concentrator per node
   - 32 MiB stripe size
   - IOC VFD as the underlying driver */
H5Pset_fapl_subfiling(fapl_id, NULL);

/* Set MPI communicator (required for parallel access) */
H5Pset_mpi_params(fapl_id, MPI_COMM_WORLD, MPI_INFO_NULL);

hid_t file_id = H5Fcreate("output.h5", H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);

/* ... perform I/O ... */

H5Fclose(file_id);
H5Pclose(fapl_id);
```

### Custom Configuration

```c
H5FD_subfiling_config_t subf_config;

/* Initialize with defaults first (or call H5Pget_fapl_subfiling on a
   default-configured FAPL to start from the library defaults) */
subf_config.magic = H5FD_SUBFILING_FAPL_MAGIC;
subf_config.version = H5FD_SUBFILING_CURR_FAPL_VERSION;

/* Configure subfiling parameters */
subf_config.shared_cfg.stripe_size = 64 * 1024 * 1024;  /* 64 MiB stripes */
subf_config.shared_cfg.stripe_count = 0; /* 0 = auto (one per IOC) */
subf_config.shared_cfg.ioc_selection = SELECT_IOC_ONE_PER_NODE;

/* Configure the underlying IOC VFD */
subf_config.ioc_fapl_id = H5I_INVALID_HID;  /* Use default IOC config */
subf_config.require_ioc = TRUE;

hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
H5Pset_fapl_subfiling(fapl_id, &subf_config);
```

### Querying Configuration

```c
H5FD_subfiling_config_t config_out;
H5Pget_fapl_subfiling(fapl_id, &config_out);

/* Note: ioc_fapl_id in the returned config is a copy
   and must be closed with H5Pclose() when done */
H5Pclose(config_out.ioc_fapl_id);
```

## h5py Usage

Subfiling is C/Fortran API only: h5py has no `subfiling` driver (none of the h5py 3.12-3.16 release notes mention one). Use `H5Pset_fapl_subfiling` from C or Fortran, or a wrapper around it; h5py's `driver='mpio'` is plain parallel HDF5 without subfiling.

## I/O Concentrator Selection Policies

The I/O concentrator (IOC) selection policy determines which MPI ranks perform actual file I/O. Available policies:

| Policy | Description |
|--------|-------------|
| `SELECT_IOC_ONE_PER_NODE` | One concentrator per compute node (default). Balances load across nodes. |
| `SELECT_IOC_EVERY_NTH_RANK` | Every Nth MPI rank is a concentrator. Set `H5FD_SUBFILING_IOC_SELECTION_CRITERIA=1:N`. |
| `SELECT_IOC_TOTAL` | Specify exact total number of concentrators. |
| `SELECT_IOC_WITH_CONFIG` | Provide a configuration file listing specific MPI ranks to use. |

Non-IOC processes send their I/O requests to an assigned IOC, which performs the actual file system operations on their behalf.

## Stripe Size

The stripe size controls how data is divided across subfiles. Each stripe-sized block of the logical file is assigned to the next subfile in round-robin order.

- **Default**: 32 MiB
- **Tuning**: Align stripe size with the parallel file system's stripe size for best performance
- **Larger stripes**: Reduce metadata overhead, better for large sequential I/O
- **Smaller stripes**: Better load balancing for random or irregular access patterns

## Subfile Organization on Disk

When you create a file named `output.h5` with N subfiles, the driver produces a stub file plus subfiles, illustrated below. The real subfile names also embed the file's inode and a `.config` file is written, so do not parse or construct names; list the physical subfiles with `H5FDsubfiling_get_file_mapping()` (HDF5 2.0+; Fortran `h5fdsubfiling_get_file_mapping_f()`).

```
output.h5                    # Stub file (small, contains metadata for reassembly)
output.h5.subfile_0_of_N     # Subfile managed by IOC rank 0
output.h5.subfile_1_of_N     # Subfile managed by IOC rank 1
...
output.h5.subfile_(N-1)_of_N # Subfile managed by IOC rank N-1
```

The stub file is required to open the logical file. All subfiles must be present and accessible. The `h5fuse` script (installed in HDF5's `bin` directory when the subfiling VFD is enabled; named `h5fuse.sh` in older releases) can merge subfiles back into a single standard HDF5 file for portability.

## Environment Variables

These override programmatic configuration at runtime:

| Variable | Purpose | Example |
|----------|---------|---------|
| `H5FD_SUBFILING_STRIPE_SIZE` | Override stripe size in bytes | `67108864` (64 MiB) |
| `H5FD_SUBFILING_IOC_PER_NODE` | Number of IOCs per node | `2` |
| `H5FD_SUBFILING_IOC_SELECTION_CRITERIA` | IOC selection as `method:criteria`, where `method` is the selection enum's integer value (`1` = every Nth rank, `3` = N concentrators in total) | `1:64` |

**Note**: `H5Pget_fapl_subfiling` returns the programmatic configuration, not the runtime overrides from environment variables.

## Best Practices

- **Start with defaults** - One IOC per node and 32 MiB stripes are good starting points for most workloads
- **Align stripe size with file system** - Match the subfiling stripe size to Lustre stripe size or GPFS block size
- **Use `h5fuse` for portability** - Merge subfiles before sharing or archiving data
- **Keep subfiles together** - All subfiles and the stub file must remain in the same directory
- **Test at target scale** - Subfiling benefits increase with process count; small-scale tests may not show gains
- **Monitor IOC distribution** - Ensure IOCs are evenly distributed across nodes for balanced I/O
- **Use collective I/O** for writes to filtered/compressed datasets, same as standard parallel HDF5
- **Initialize MPI correctly** - Always use `MPI_Init_thread` with `MPI_THREAD_MULTIPLE`

## Common Pitfalls

- **Missing `MPI_THREAD_MULTIPLE`**: Using `MPI_Init` instead of `MPI_Init_thread` with `MPI_THREAD_MULTIPLE` will cause failures. The subfiling VFD uses internal threads for I/O concentrator communication.
- **Moving subfiles independently**: Subfiles must stay together with their stub file. Moving or renaming individual subfiles breaks the logical file.
- **Forgetting to fuse before sharing**: Recipients without subfiling support cannot read the distributed subfiles. Use `h5fuse` first.
- **Mismatched environment variables**: Environment variable overrides apply at runtime and can silently change behavior. Be deliberate about which configuration method you use.
- **Too many IOCs**: More IOCs than file system targets can cause contention rather than alleviating it. Match IOC count to available storage targets.
- **Too few IOCs**: Under-utilizing available parallelism. One per node is usually a good balance.
- **Known test failures**: `MPI_TEST-t_subfiling_vfd` and `ph5_subfiling` fail with cray-mpich on theta and with XL compilers on ppc64le (listed as known problems in the 2.0 and 2.2.0 release notes).

## Performance Considerations

**When subfiling helps most**:
- High process counts (100+) where single-file contention is the bottleneck
- Large aggregate I/O volumes distributed across many processes
- Parallel file systems with many storage targets (OSTs/NSDs)

**When subfiling adds overhead**:
- Small process counts where contention isn't an issue
- Very small I/O operations (IOC communication overhead dominates)
- Applications already well-optimized with MPI-IO hints

## Merging Subfiles

```bash
# Merge subfiles into a single standard HDF5 file. Pass the subfiling
# configuration file written next to the subfiles, or run in that directory
# and h5fuse finds the *.config file itself.
h5fuse -f <subfiling .config file>

# This produces a standard HDF5 file readable by any HDF5 application
# The original subfiles can be removed after verification
```

## Citations and References

- [HDF Group: Subfiling VFD User Guide Announcement](https://www.hdfgroup.org/2023/02/01/subfiling-virtual-file-driver-user-guide-is-now-available/)
- [HDF Group: Subfiling VFD Presentation (2022)](https://www.hdfgroup.org/wp-content/uploads/2022/09/HDF5-Subfiling-VFD.pdf)
- [HDF5: Virtual File Driver Documentation](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_f_d__u_g.html)
- [HDF5: File Access Properties (H5Pset_fapl_subfiling)](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html)
- [HDF5: Virtual File Driver Features](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_v_f_d.html)
