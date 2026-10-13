# HDF5 File Space Management

## Purpose

Control how HDF5 allocates, tracks, and reuses free space. By default, deleted objects leave unrecoverable dead space — files do not shrink. Understanding file space strategies prevents bloat in mutable, long-lived files.

**Related**: `references/io.md` for write performance, `references/chunking.md` for allocation timing, `references/filters.md` for h5repack usage.

## Why Files Don't Shrink

HDF5 is append-oriented by default. When you delete a dataset or overwrite data:
- The file boundary never shrinks automatically
- Freed space is tracked in the **free space manager (FSM)** but may not be reused
- Physical size only decreases after running `h5repack` (offline defragmentation)

This is by design — partial writes that fail mid-operation leave prior data intact.

## File Space Strategies

Set via `H5Pset_file_space_strategy()` at **file creation only** — cannot change afterward.

| Strategy | Constant | Behavior |
|---|---|---|
| **FSM Aggregator** | `H5F_FSPACE_STRATEGY_FSM_AGGR` | Default. Tracks free space in B-tree; reuses freed blocks for new allocations. |
| **Page** | `H5F_FSPACE_STRATEGY_PAGE` | File divided into fixed pages; enables **page buffering**. Best for metadata-heavy workloads. |
| **Aggregator only** | `H5F_FSPACE_STRATEGY_AGGR` | Simple aggregator, no free-space tracking. Freed space is permanently lost. Minimal overhead. |
| **None** | `H5F_FSPACE_STRATEGY_NONE` | No management. All freed space is permanently lost. Use only for write-once files. |

Default (`FSM_AGGR`) is correct for most workloads.

In h5py, pass the settings to the high-level constructor: `h5py.File('f.h5', 'w', fs_strategy='fsm', fs_persist=True, fs_threshold=1)` (also `fs_page_size`). Use h5repack to change the strategy of an existing file.

## Page Buffering (Page Strategy Only)

With `H5F_FSPACE_STRATEGY_PAGE`, the library buffers entire pages in memory before flushing:
- Reduces small metadata I/O by batching page-aligned reads/writes
- Reduces per-request cost on cloud/network storage
- Set page size via `H5Pset_file_space_page_size()` (default: 4096 bytes)
- Set page buffer size via `H5Pset_page_buffer_size()` (default: 0, i.e., disabled)

**Trade-off**: Partially-filled pages waste space. Requires tuning page size to match access patterns. For local POSIX storage with infrequent metadata operations, FSM_AGGR is usually better.

## FSM Persistence and Threshold

`H5Pset_file_space_strategy(fcpl, strategy, persist, threshold)`:

- `persist=true` — FSM state written to file, so later sessions can reuse space freed earlier. Recommended for files opened and closed repeatedly with many deletions.
- `persist=false` (default) — free-space tracking lasts only for the current open; space freed in an earlier session is not reused and stays dead until `h5repack`.
- `threshold` — minimum free-space block size tracked (bytes). Micro-fragments below threshold are abandoned. Set large enough to avoid tracking fragments that cost more to manage than they save.

## Reclaiming Space: h5repack

The only way to physically shrink a file. Rewrites all objects into a new, compact file, removing dead space, deleted objects, and stale metadata.

```bash
h5repack input.h5 output.h5                           # compact only
h5repack -f SHUF -f GZIP=3 bloated.h5 compact.h5     # compact + recompress
h5repack -S PAGE -G 65536 old.h5 new.h5               # change strategy and page size
```

- Original file is not modified — output is a new file
- Requires roughly equal temporary disk space to input file size
- Can simultaneously change chunking, compression, and file space strategy
- Do not run on files that other processes have open

**Automate**: After bulk deletes or before archiving/distributing, always repack.

## Inspecting File Space

```bash
h5stat -S file.h5          # Summary: raw data, metadata, unaccounted space
h5stat -F file.h5          # Free space section details
h5dump -B file.h5            # Superblock, including file space strategy settings
```

Key metric: `Unaccounted space: N bytes` in `h5stat -S` output — this is the dead/fragmented space shrinkable by h5repack.

## HDF5 2.x notes

- `h5repack` now takes its default low and high library version bounds from the library (default low bound is 1.8), so repacked files can change format level; pass `--low=0` to restore the previous behavior.
- `H5Pset_fapl_ros3()` sets a 64 MiB page buffer if none is set, but this only has an effect when the file uses paged allocation.
- The VDS global-heap and filtered-chunk-size format changes apply only when the lower library version bound is 2.0 or later.

## Dataset Allocation Time

Controls when disk space is physically reserved for a dataset:

| Mode | Space Reserved | Default For | Use Case |
|---|---|---|---|
| `H5D_ALLOC_TIME_EARLY` | At creation | — | Parallel HDF5, predictable layout |
| `H5D_ALLOC_TIME_INCR` | As chunks written | Chunked | Normal chunked datasets |
| `H5D_ALLOC_TIME_LATE` | At first write | Contiguous | Space-efficient sparse data |

Set via `H5Pset_alloc_time()` on the dataset creation property list. Parallel HDF5 **requires** `EARLY` to avoid serialization during collective writes.

## Do / Do Not

**Do**:
- Run `h5repack` before archiving or distributing files that have had objects deleted
- Use `h5stat -S` to diagnose unexpectedly large files before investigating code
- Use `H5F_FSPACE_STRATEGY_NONE` for write-once archival files (zero tracking overhead)
- Set `persist=true` for files that cycle through many open/close/delete operations
- Set `threshold` large enough to avoid tracking micro-fragments

**Do not**:
- Expect files to shrink after deleting datasets — they will not without h5repack
- Change file space strategy after file creation — it is a creation-time property
- Run h5repack while other processes have the file open
- Use paged allocation without profiling — page size mismatch can hurt more than it helps
- Assume SWMR mode reclaims space — SWMR explicitly disables space reclamation

## Common Pitfalls

**Unbounded growth through edit cycles**: Each write-modify-close cycle appends new metadata even when overwriting. Run `h5repack` periodically for mutable files.

**h5repack doubles disk requirement**: Requires a full copy. Verify available space before repacking large files.

**Lost free space across sessions**: With `persist=false` (default), free space is tracked only while the file is open, so space freed in one session is not reused by the next. For mutable long-lived files, set `persist=true` or repack periodically.

**Allocation time mismatch in parallel**: Without `H5D_ALLOC_TIME_EARLY`, parallel writes may trigger unexpected collective metadata operations and serialize.

## When to Care

- Files with frequent deletions or overwrites (simulation checkpoints, rolling logs)
- Long-lived files edited over many sessions
- Files to be distributed or archived (minimize size)
- Cloud storage where per-byte cost matters
- Metadata-heavy files with many small objects (consider page strategy)

## Citations and References

- [HDF Group: File Space Management](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html)
- [HDF Group: File Creation Property Lists (file space)](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_c_p_l.html)
- [HDF Group: h5repack Reference](https://support.hdfgroup.org/documentation/hdf5/latest/view/reference/tools_h5repack.html)
- [HDF Group: h5stat Reference](https://support.hdfgroup.org/documentation/hdf5/latest/view/reference/tools_h5stat.html)
