---
name: clio-fs
description: 'Use when mounting, configuring or debugging the CLIO Core POSIX file layer: the filesystem ChiMod (560.0), the stream ChiMod (565.0), the clio_cte_fuse mount on Linux/Windows/macOS, the checkpoint fault handler (566.0), and the LD_PRELOAD/MPI-IO/HDF5/ADIOS2 adapters. Triggers on "clio_cte_fuse", "mount CLIO", "metadata_log_path", "CLIO_FUSE_", "fsync EIO", "case_insensitive", "stream pool". Not for chain design or blob APIs; use storage-compose or clio-cte-core.'
metadata:
  bundle: clio-core
  servers: none
  provenance: designed
  eval-status: scenarios-recorded
---

# clio-fs: files over the Context Transfer Engine

Context FS turns CTE tags and blobs into an ordinary directory tree. It has
two parts:

- **`clio_cte_filesystem`** (pool 560.0), a ChiMod inside the Context Runtime.
  It owns the namespace: directories, inode records, links, xattrs and open
  handles. A file is a CTE tag named by its path. Its bytes are 1 MiB page
  blobs named `"0"`, `"1"`, ... Its attributes are the blob `~i`.
- **`clio_cte_fuse`**, a client process that mounts the filesystem pool
  through the OS FUSE driver (libfuse3, WinFsp, macFUSE).

Every file's logical size and its `O_APPEND` merges belong to a third pool,
**`clio_cte_stream`** (565.0). Pages travel down the CTE chain you composed,
so tiers, caching, replication and search apply to files unchanged. The
cost: every syscall is a FUSE kernel round trip plus one or more runtime
tasks. Measured: `stat` about 31 us, `readdir` of 10 entries about 255 us,
`rename` about 570 us; a Linux kernel `git clone` took 171 s (ext4: 39 s).

Facts were checked against clio-core `dev` (bd1dc3183, 2026-11-21).

## Files in this skill

| file | read when |
|---|---|
| **MOUNT.md** | starting, tuning or unmounting `clio_cte_fuse`; every flag and `CLIO_FUSE_*` variable; Linux, Windows, macOS, containers, multi-node; the other adapters (POSIX, MPI-IO, HDF5, ADIOS2). |
| **GUARANTEES.md** | before saying what `write`, `close` or `fsync` guarantee, what each failure loses, or how nodes see each other's changes. |
| **STREAM_CHECKPOINT.md** | configuring or calling `clio_cte_stream` directly, or `clio_cte_checkpoint` (lazy copies behind `Vector::Copy`). |

## Availability

| piece | `mod_name` / binary | default pool id | pip wheel | build option otherwise |
|---|---|---|---|---|
| filesystem | `clio_cte_filesystem` | 560.0 | yes (all platforms) | `CLIO_CTE_ENABLE_FILESYSTEM`; forced on by any of the FUSE, POSIX, STDIO, MPI-IO or VFD adapter options |
| stream | `clio_cte_stream` | 565.0 | yes | `CLIO_CTE_ENABLE_STREAM` (default ON) |
| checkpoint | `clio_cte_checkpoint` | 566.0 | **no** (built by default, shipped in deb/rpm, not in the wheel) | `CLIO_CTE_ENABLE_CHECKPOINT` (default ON) |
| mount daemon | `clio_cte_fuse` | n/a | Linux and Windows wheels; **not macOS** | `-DCLIO_CTE_ENABLE_FUSE_ADAPTER=ON` (presets `release-fuse`, `release-mac-fuse`) |
| POSIX/STDIO/MPI-IO interceptors, HDF5 VOL/VFD, ADIOS2 engine | see MOUNT.md | n/a | **no** (wheel builds with ELF, MPI, HDF5 off) | source, Spack or conda |

Linux needs the system `fuse3`/`libfuse3-3` packages (the wheel does not
bundle `libfuse3.so.3`). Windows needs WinFsp. The macOS wheel installs a
`clio_cte_fuse` console script that prints "binary not found".

## When to use it

- Use it when an unmodified program must read and write files (shell tools,
  git, Python `open`, compilers, ParaView) and you want CTE tiering,
  replication or keyword search under them.
- Do not use it for bulk data your own code writes: the CTE client
  (`PutBlob`/`GetBlob`) or a gpu_vector skips the FUSE kernel transit.
- Do not expect POSIX locks: `flock`/`fcntl` locks, `mknod`, FIFOs,
  hole punching, `RENAME_EXCHANGE` and `copy_file_range` are not provided.

## Compose entry

The filesystem pool must exist before a mount: `clio_cte_fuse` binds it by
name and **creates no pool** (it fails with "no filesystem pool named
'clio_cte_filesystem'" if your compose lacks it). The shipped default
config composes both entries below. Copy-paste version for the default chain
(indexer 564.0 at the top, replication 561.0 above the persistent core):

```yaml
compose:
  # ... bdev, cores 512.0/540.0, replication 561.0, write-back 567.0,
  #     shm cache 568.0, indexer 564.0 above this point ...
  - mod_name: clio_cte_stream          # BEFORE the filesystem
    pool_name: clio_cte_stream
    pool_query: local
    pool_id: "565.0"
    next_pool_id: "564.0"              # chain top
    staging_pool_id: "561.0"           # staged appends skip the caches
    log_path: "${CLIO_STORAGE_ROOT}/cte_stream_log"   # keep exact sizes
  - mod_name: clio_cte_filesystem
    pool_name: clio_cte_filesystem     # clients find the pool by this name
    pool_query: local
    pool_id: "560.0"
    next_pool_id: "564.0"              # required
    stream_pool_id: "565.0"
    metadata_log_path: "${CLIO_STORAGE_ROOT}/cfs_namespace_log"
    # dir_split_entries: 1024
    # dir_log_compact_bytes: 256KB
    # case_insensitive: false
    # data_score: -1.0
    # data_score_head_pages: 1
```

`${CLIO_STORAGE_ROOT}` is `clio-run start --disk <dir>`, default `~/.clio`.

### `clio_cte_filesystem` keys

| key | type | default | meaning and constraints |
|---|---|---|---|
| `next_pool_id` | `"major.minor"` | none; **required** | Pool every page, directory block and inode record goes through. Point at the chain top. Missing: pool creation fails (rc 1). `"512.0"` bypasses every cache and the indexer. |
| `metadata_log_path` | path | unset | Node-local journal of the directory blocks this node is home for, written to `<path>.dirs.<container_id>`; `~` and `$VARS` expand. One small record per change; fsynced by `fsync`/`fsyncdir` and every 1000 ms. Each change is also mirrored to a CTE blob that a stand-in reads while this node is down. Unset: directory blocks live only as CTE blobs (one blob write per change; slower creates). |
| `stream_pool_id` | `"major.minor"` | unset | Stream pool for sizes and appends. Unset: the filesystem creates its own pool named `<pool_name>_stream` at `<major>.<minor+1>` (560.1 for 560.0), logging to `<metadata_log_path>.stream`. If set to a composed pool, that pool's own `log_path` is what counts. |
| `dir_split_entries` | unsigned int | 1024 | Rows a directory block holds before it splits in two. Values below 16 are raised to 16. Blocks never merge. |
| `dir_log_compact_bytes` | size (`"256KB"`) | 256 KiB | Delta bytes a block's CTE blob may hold past its last full image before the next commit rewrites it whole. 0 = rewrite on every commit. |
| `case_insensitive` | bool | false | Compare names case-insensitively (Unicode simple fold), keeping their spelling, as NTFS does. Fixed when the volume is created: a runtime configured the other way refuses the volume (pool create fails with EINVAL). |
| `data_score` | float | -1.0 (off) | Placement score for file data pages. A value in (0, 1] steers bulk data to that tier and leaves the top tier to directory blocks, xattrs and file heads. Any other value is ignored with a warning. |
| `data_score_head_pages` | unsigned int | 1 | Leading 1 MiB pages per file that keep the default score when `data_score` is set. 0 = every page uses `data_score`. |

Not keys: page size (fixed 1 MiB), `block_size`, and `fsync_mode`. **`fsync_mode`
is a CTE core key** (`performance.fsync_mode` on the persistent core entry,
`"durable"` default or `"deferred"`); see GUARANTEES.md. A value that fails to
parse (for example `case_insensitive: maybe`) is silently ignored and all
keys after it keep their defaults: check the runtime log after editing.

### Environment variables (runtime side and library clients)

| variable | read by | default | effect |
|---|---|---|---|
| `CLIO_CFS_POOL` | every filesystem client | `clio_cte_filesystem` | Pool name, or `"major.minor"`, of the filesystem to bind. |
| `CLIO_CFS_SHM_READ` | clients | on | `0` disables the zero-IPC shared-memory read path. |
| `CLIO_CFS_SHM_FILE_CAPACITY` | runtime | 262144 files | Size of the shared-memory attribute mirror that answers `stat` in about 10 us. |
| `CLIO_CFS_SHM_MIRROR_MULTINODE` | runtime | off | The mirror is off on multi-node runtimes; `1` forces it on (only safe if files are never shared across nodes). |
| `CLIO_CFS_PREALLOC` | runtime | 0 | Flat per-page preallocation in bytes; 0 = doubling growth capped at 64 KiB. |
| `CLIO_NET_DEAD_FAIL_FAST` | runtime | off | `1`: operations on names a declared-dead node owns return EIO at once instead of stalling. |

Mount-daemon variables (`CLIO_FUSE_*`) are in MOUNT.md.

## Where it goes in the chain

```
clio_cte_fuse / interceptors
   -> clio_cte_filesystem 560.0  (namespace)  -- sizes/appends --> clio_cte_stream 565.0
   -> chain top (indexer 564.0 -> shm cache 568.0 -> write-back 567.0
        -> volatile core 540.0 | replication 561.0 -> persistent core 512.0)
```

- Ordering rule: a pool must be composed after every pool it names. Compose
  the chain, then the stream, then the filesystem.
- The FUSE daemon writes pages **directly** to the filesystem's
  `next_pool_id` through the CTE client (not through the ChiMod), so that
  pool must be one a client can address: the chain top.
- `staging_pool_id` should point below any cache (561.0 in the default):
  staged appends are written once and read at most once.
- With the indexer above, every byte written is tokenized into an in-memory
  term index. For bulk-data volumes point `next_pool_id` (both entries) at
  568.0 to skip it; see Pitfalls.

## Guarantees (summary; details in GUARANTEES.md)

- `write()` returns once the page is in the chain: by default in the
  write-back cache's RAM copy on the page's owner node. Not yet power-safe.
- `close()` publishes the size (close-to-open): the next `open` on any node
  sees the bytes and size.
- `fsync()` (default `fsync_mode: durable`) returns once the file's pages are
  on a persistent device, the size log and the parent directory are synced.
  It returns **EIO** if a node that may have held the file's unsynced writes
  died since the last fsync.
- Default `replication_factor: 1`: a node's data is unavailable while it is
  down and lost if it never returns, including the directory blocks and inode
  records it held.
- `clio-run start --fresh` discards this node's journal, stream log and CTE
  state.

## Client API

There is no Python binding for the filesystem or stream pools, and the wheel
ships no C++ headers. **The mount is the API.** To read mounted files from
Python, use the CTE client on the chain top: a file `/notes/a.txt` is the
tag `/notes/a.txt`, pages are blobs `"0"`, `"1"`, ...

```python
import os
os.environ["CLIO_WITH_RUNTIME"] = "0"     # attach to the running runtime
os.environ["CLIO_CTE_POOL"] = "564.0"     # chain top (search lives here)
import clio_cte_core_ext as cte
cte.clio_init(cte.RuntimeMode.kClient, False)
cte.initialize_cte("", cte.PoolQuery.Dynamic())
client = cte.get_cte_client()
print(client.TagQuery(".*notes/.*", 0))   # mounted paths under /notes
```

Write through the mount, not by `PutBlob` into a file's tag: a direct put
does not move the file's size, which the stream pool owns. Source builds can
use the C++ `clio::cte::filesystem::Client` (`CLIO_CFS_CLIENT`, e.g.
`AsyncOpen`, `OpenFd`/`PwriteFd`/`SyncFd`/`CloseFd`); all methods return 0
or an errno-style code.

## Pitfalls and failure signatures

| symptom | cause | fix |
|---|---|---|
| Files written through the mount are invisible to Python or to a second mount | daemon started without `CLIO_WITH_RUNTIME=0` and no runtime was up, so it became a private runtime | `clio-run start` first; mount with `CLIO_WITH_RUNTIME=0` |
| Mount exits: "no filesystem pool named 'clio_cte_filesystem'" | compose lacks the filesystem entry, or a different `pool_name` | compose it (above) or set `CLIO_CFS_POOL` |
| Every write blocks about 120 s, no error; `ls` hangs | the runtime died (often OOM-killed) under the mount; the daemon does not turn this into EIO | `fusermount3 -uz <mnt>` (macOS `umount -f`, Windows `Stop-Process`), kill `clio_cte_fuse`, `clio-run start` (no `--fresh`), remount |
| Runtime RSS grows by GBs during large copies, then it is killed | indexer (564.0) keeps an in-memory term index of every byte; measured OOM at about 10 GB written on a 4 GB host | for bulk data, point stream and filesystem `next_pool_id` at 568.0, or narrow the indexer's `tag_re`; cap container memory in benchmarks |
| `Transport endpoint is not connected` | daemon exited, stale mount left | `fusermount3 -u <mnt>`, remount |
| Pool create fails EINVAL after an edit | `case_insensitive` differs from the volume's | restore the value, or `clio-run start --fresh` (destroys data) |
| `fsync` returns EIO after a peer reboot | that peer may have held unsynced writes (default single copy) | rewrite the data since the last fsync; use `replication_factor` > 1 to avoid it |
| File reads stale size on another node | `CLIO_FUSE_ATTR_CACHE_S` > 0 on a multi-node mount, or reader opened before the writer closed | keep the multi-node default 0; open after the writer's `close` |
| Directory listings or tree lost after a node was decommissioned | its directory journal is node-local (#1263); with one copy the CTE mirror died too | `replication_factor` >= 2 with `failover_to_successor: true` before relying on node removal |
| Windows: `open(path, "x")` on an existing file succeeded (older builds) | WinFsp skips the lookup before an exclusive create | fixed: `O_EXCL` is decided by the ChiMod; upgrade |

## Verify it works

```bash
clio-run start &                          # wait for the "bound on" log line
clio-run status                           # RUNNING
mkdir -p ~/clio-mnt
CLIO_WITH_RUNTIME=0 clio_cte_fuse ~/clio-mnt -f &
echo hello > ~/clio-mnt/hello.txt && cat ~/clio-mnt/hello.txt
dd if=/dev/urandom of=~/clio-mnt/big bs=1M count=64 conv=fsync
fusermount3 -u ~/clio-mnt && clio-run stop
```

The dashboard (`http://127.0.0.1:8080`) Pools page lists
`clio_cte_filesystem` (560.0) and `clio_cte_stream` (565.0); the core's tier
roster shows the bytes. The crash-and-restart check is in GUARANTEES.md.
