# HDF5 SWMR (Single-Writer/Multiple-Reader)

## Purpose

Enable concurrent access to HDF5 files where one writer process modifies a file while multiple reader processes read simultaneously, without locks or inter-process communication. Critical for real-time data acquisition, streaming sensors, and live monitoring applications.

**Related**: `references/io.md` for I/O optimization, `references/chunking.md` for chunk configuration, `references/parallel.md` for MPI-based concurrent access.

## What Problem SWMR Solves

Traditional HDF5 requires exclusive access for writing - readers must wait until writer closes the file. SWMR breaks this constraint, allowing readers to observe data as it's being written. Key benefit: **files remain valid and non-corrupt even if writer crashes mid-operation**.

## How SWMR Works Under the Hood

**The Challenge**: Writers maintain metadata in memory cache plus disk storage, but readers only see disk. This creates a visibility gap where file pointers might reference unflushed addresses.

**The Solution**: SWMR implements **flush dependencies** - parent-child relationships ensuring referenced metadata flushes before the metadata referencing it. Combined with **metadata checksums**, this guarantees readers never encounter invalid pointers.

**Locking Behavior**: Normal writes hold exclusive locks. SWMR removes the lock when `H5Fstart_swmr_write()` completes, allowing multiple simultaneous readers. Locks exist only for safe setup, not SWMR operation.

## Requirements

**HDF5 Version**: 1.10+ required. SWMR needs the v110 (or newer) file format (NOT compatible with HDF5 < 1.10). In HDF5 2.x `libver='latest'` means the v200 format, which 1.x libraries may be unable to read even though SWMR only needs v110: pass `libver=('v110', 'v110')` (a v110 upper bound) when 1.10-1.14 readers must open the file, and plain `'latest'` only when every reader is HDF5 2.x.

**Filesystem**: Must be POSIX-compliant with correct `write()` semantics. **DOES NOT WORK** on NFS, SMB, Windows network shares, or any network filesystem. Use local disk only.

**File Structure**: All groups, datasets, and attributes must exist before enabling SWMR. No structural changes allowed after SWMR activation.

## Writer Workflow

**Creating New SWMR File**:

```python
import h5py
import numpy as np

# 1. Create file with a SWMR-capable format
#    ('latest' = v200 on HDF5 2.x; use ('v110', 'v110') for 1.x readers)
f = h5py.File('data.h5', 'w', libver='latest')

# 2. Create ALL structure upfront (groups, datasets, attributes)
dset = f.create_dataset('sensor_data',
                         shape=(0,),
                         maxshape=(None,),
                         chunks=(1000,),
                         dtype='f')
dset.attrs['units'] = 'celsius'

# 3. Close any attributes and named datatypes (important!)
# Already done above

# 4. Enable SWMR mode
f.swmr_mode = True

# 5. Write and flush regularly
for i in range(1000):
    new_data = np.random.randn(100)
    dset.resize((dset.shape[0] + 100,))
    dset[-100:] = new_data
    dset.flush()  # CRITICAL - makes data visible to readers
    time.sleep(0.1)

f.close()
```

**Opening Existing File for SWMR Writing**:

```python
f = h5py.File('data.h5', 'r+', libver='latest', swmr=True)
dset = f['sensor_data']
# Write and flush regularly...
```

**Critical**: Call `dset.flush()` after each write to make changes visible to readers.

## Reader Workflow

```python
import h5py
import time

# 1. Open with SWMR flag
f = h5py.File('data.h5', 'r', libver='latest', swmr=True)
dset = f['sensor_data']

# 2. Polling loop
last_size = 0
while True:
    # Refresh to see writer's changes
    dset.refresh()

    # Check for new data
    current_size = dset.shape[0]
    if current_size > last_size:
        # Read new data only
        new_data = dset[last_size:current_size]
        print(f"Read {len(new_data)} new points")
        last_size = current_size

    time.sleep(0.5)  # Polling interval

f.close()
```

**Critical**: Call `dset.refresh()` before reading to synchronize with writer's changes.

## C API Equivalents

```c
// Writer: H5Fstart_swmr_write(file) then H5Dflush(dset) after writes
// Reader: H5Fopen with H5F_ACC_SWMR_READ, then H5Drefresh(dset) before reads
```

Fortran (HDF5 2.1+): `h5fstart_swmr_write_f`, `h5dflush_f`, `h5pset_append_flush_f` and `h5pget_append_flush_f` wrap the corresponding C calls, and the `H5F_ACC_SWMR_READ_F` / `H5F_ACC_SWMR_WRITE_F` access flags are available.

h5py 3.16: `Group.get` accepts `elink_mode`, `elink_swmr` and `elink_locking`, useful when a SWMR file is reached through an external link.

## Allowed vs Prohibited Operations

**Writer CAN**:
- ✅ Append data along unlimited dimensions
- ✅ Modify existing data in place
- ✅ Flush specific datasets (`H5Dflush()`)
- ✅ Control metadata flushing (`H5Odisable_mdc_flushes()`, `H5Oenable_mdc_flushes()`)

**Writer CANNOT**:
- ❌ Create new groups or datasets
- ❌ Delete objects
- ❌ Modify variable-length datatypes
- ❌ Modify string datatypes
- ❌ Modify region reference datatypes
- ❌ Reclaim freed file space (files grow larger than normal)

## Coordination and Synchronization

**Critical**: SWMR provides **no automatic notification** between processes. You must implement coordination:

**Options**: Polling (simple `sleep()` loop), inotify (Linux file watch), or custom IPC (shared memory, pipes, sockets).

## Best Practices

- **Structure upfront**: Create all groups/datasets before enabling SWMR
- **Flush regularly**: Writer must call `flush()` - readers can't see unflushed data
- **Refresh before reading**: Readers must call `refresh()` to see changes
- **Use chunked datasets**: Required for extending data (set `maxshape=(None,)`)
- **Set appropriate poll intervals**: Balance latency vs overhead (0.1-1.0 seconds typical)
- **Use local filesystems only**: Network filesystems violate POSIX requirements
- **Handle crashes gracefully**: File remains valid through last flush
- **Test ordering**: Verify reader opens after writer enables SWMR
- **Document flush frequency**: Critical for readers to understand data freshness

## Common Pitfalls and Deadlock Scenarios

**Writer forgets to flush**:
- Symptom: Readers see no new data despite writer running
- Solution: Always call `dset.flush()` after writes

**Reader forgets to refresh**:
- Symptom: Readers see stale data forever
- Solution: Always call `dset.refresh()` before reads

**Non-SWMR reader opens first**:
- Symptom: Writer fails to enable SWMR mode (error: cannot start SWMR write)
- Solution: Ensure only SWMR-aware readers access file, or start writer first

**Creating objects after SWMR enabled**:
- Symptom: Operations fail with "cannot create object" error
- Solution: Create all structure before calling `f.swmr_mode = True`

**Network filesystem usage**:
- Symptom: Silent data corruption, readers see garbage
- Solution: Use local POSIX-compliant filesystem only

**Mixed SWMR and non-SWMR readers**:
- Symptom: Standard tools (h5dump, HDFView) prevent SWMR activation
- Solution: Close all non-SWMR file handles before enabling SWMR

**File format incompatibility**:
- Symptom: Older tools/libraries cannot open SWMR files
- Solution: Ensure all tools support HDF5 1.10+, or convert with `h5format_convert`

## When to Use SWMR

**Ideal Use Cases**:
- Real-time sensor data acquisition (temperature, pressure, detectors)
- Live monitoring dashboards
- Streaming data from instruments
- Long-running simulations with live visualization
- Data logging with concurrent analysis

**Key Characteristics**:
- Fixed file structure known upfront
- Primarily append-only operations
- Multiple consumers need live data
- Data validity critical (crash tolerance)

## When NOT to Use SWMR

**Avoid if**:
- File structure changes dynamically
- Need to create/delete objects during writing
- Network filesystem required (use `references/parallel.md` with MPI instead)
- Backward compatibility with HDF5 < 1.10 needed
- Frequent modifications to variable-length/string data

**Alternative Approach**: For simpler cases, use normal write mode with readers that disable locking and catch/handle exceptions. More flexible but less safe.

## Supporting Tools

**h5watch**: Monitor datasets in real-time
```bash
h5watch --dim --label data.h5/sensor_data
```

**h5clear**: Reset superblock flags after crash
```bash
h5clear -s data.h5  # Clear status flags
```

**h5format_convert**: Convert file formats
```bash
h5format_convert -d data.h5  # Downgrade for compatibility
```

## Performance Considerations

- **File size overhead**: SWMR disables space reclamation - deleted data not reclaimed
- **Flush overhead**: Frequent flushes reduce write throughput - balance freshness vs performance
- **Metadata cache pressure**: Many readers increase system memory usage
- **Lock contention**: POSIX file locks during SWMR transition (brief)

The write-throughput cost depends mostly on how often the writer flushes; measure it for your workload.

## Troubleshooting

**Error: "unable to start SWMR write"**
- Non-SWMR reader has file open
- File not on POSIX filesystem
- File format older than v110 (missing a `libver` setting such as `'latest'` or `('v110', 'v110')`)

**Readers see no data**
- Writer not calling `flush()`
- Readers not calling `refresh()`
- Synchronization timing issue

**File corruption on network drive**
- Network filesystem lacks POSIX write semantics
- Move to local disk

## Citations and References

- [HDF Group: SWMR Introduction](https://support.hdfgroup.org/documentation/hdf5/latest/_s_w_m_r_t_n.html)
- [HDF Group: SWMR Tutorial](https://support.hdfgroup.org/archive/support/HDF5/Tutor/swmr.html)
- [h5py: SWMR Documentation](https://docs.h5py.org/en/stable/swmr.html)
- [HDF Group: File Locking in HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_file_lock.html)
- [HDF Group: SWMR Design Document](https://docs.hdfgroup.org/archive/support/HDF5/docNewFeatures/SWMR/Design-HDF5-SWMR-20130629.v5.2.pdf)
