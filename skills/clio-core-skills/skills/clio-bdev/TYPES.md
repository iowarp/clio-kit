# Bdev types in detail

Every type shares one allocator and one task interface. The allocator is a
bump pointer over the capacity plus per-worker free lists in size classes from
512 B to 1 MB, with stealing between workers. A large request can come back
as several extents, so callers must handle block lists. Free space is never
compacted, so long churn fragments it into blocks of 1 MB or less.

## ram

- Host memory, one sparse shared-memory segment per pool so client processes
  on the node can read blob payloads directly.
- `pool_name`: any unique string, conventionally `ram::<name>`.
- `capacity: 0` = 80% of system DRAM. Pages are committed `populate_unit`
  (default 64 MB) at a time as allocations advance, so committed memory tracks
  the high-water mark, not the declared capacity.
- `alloc: eager` pre-faults everything at creation (only with a non-zero
  capacity, and only up to half the process memory budget).
- Contents are lost on any restart. Never restores persisted perf figures.
- Use it for the hot tier and for the required default bdev at `301.0`.

## file

- `pool_name` is the backing file path. Parent directories must exist and be
  writable by the runtime user.
- Capacity: `capacity` if set, else the existing file's size, else 1 GiB. The
  file starts at one `growth_unit` (default 1 GiB) and grows on demand; where
  the filesystem supports it each step reserves real blocks, so a full disk
  fails at allocation time, not mid-write. A failed growth is remembered for
  2 s so requests past it fail fast.
- Remaining capacity is the allocator's free space capped by the host
  filesystem's free space, sampled at most once per second.
- I/O backend chosen at build time: NIXL, io_uring, libaio, then POSIX AIO
  (Windows: I/O completion ports). If io_uring cannot open the file at run
  time (container seccomp) it falls back to POSIX AIO. Network filesystems
  (Lustre, NFS, dfuse) get POSIX AIO automatically.
- Aligned writes of at least 64 KiB go through direct I/O.
- If the path later names a different file (deleted and re-created), reopen
  fails instead of silently using an empty file.
- Allocator WAL: see SKILL.md "Durability of the allocator".
- Disk-class bdevs save measured performance figures and reload them at the
  next start; memory types never do.
- Works on Linux, macOS and Windows (NTFS growth reserves clusters eagerly,
  which is why `growth_unit` exists).

## hbm

- One device buffer covering the whole capacity, allocated at creation.
- In a CUDA or ROCm build a failed allocation is **fatal**; it never falls
  back to host memory. In a CPU build (the plain pip wheel) and in SYCL builds
  it is host memory and a warning says so.
- Volatile. Type default bandwidth 1,500,000 MB/s, so the placement engine
  ranks it above ram.
- In a CTE core use it as an ordinary score-1.0 volatile tier. Do not build a
  manual loader around it or move durable data into it as a special role.

## pinned

- Page-locked host memory from the GPU API, so device copies can run
  asynchronously. Not shared-memory backed.
- If page-locking fails, it logs an error and falls back to pageable memory.
- `CLIO_BDEV_NO_PIN=1` turns off pinning of ram-tier pages (a ram-tier
  option, not a pinned one).
- Volatile.

## noop

- No storage. Allocate, free, write and read succeed without moving bytes.
  Use it only to measure runtime overhead. Reads return garbage.

## s3

- Needs a source build with `-DCLIO_ENABLE_AMAZON_DRIVE=ON` and Poco (the
  option is turned off automatically when Poco is missing). No AWS SDK.
- `pool_name`: `bucket`, `bucket/prefix`, or `s3://bucket/prefix`. In a CTE
  tier always give a prefix: the CTE appends `_node<N>`, which must land on
  the prefix, not the bucket name.
- Each block is one object `[prefix/]block_<offset>`. A block never written
  reads as zeros (404). One keep-alive connection per worker.
- Capacity 0 = 1 TiB of address space.
- Allocator state is memory only: after a restart the objects remain but the
  allocator starts empty.

Environment (of the process that starts the runtime, set before `clio-run start`):

| Variable | Default | Meaning |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | none; required | pool creation fails without it |
| `AWS_SECRET_ACCESS_KEY` | none; required | |
| `AWS_SESSION_TOKEN` | unset | for temporary credentials |
| `AWS_DEFAULT_REGION` | `us-east-1` | |
| `S3_ENDPOINT` | real AWS (virtual-hosted bucket) | override, switches to path-style addressing (MinIO, test stubs) |
| `S3_ALLOW_BUCKET_CREATE` | unset | `1` creates a missing bucket; otherwise a missing bucket fails creation |
| `https_proxy` / `HTTPS_PROXY`, `http_proxy` / `HTTP_PROXY` | unset | `http://` proxies only |

## gcs

- Needs `-DCLIO_ENABLE_GOOGLE_CLOUD=ON` and Poco.
- `pool_name`: `bucket`, `bucket/prefix`, or `gcs://bucket/prefix`. Same
  object layout, zero-fill and 1 TiB default as s3. The bucket is created if
  missing. A new HTTPS session per request.
- Allocator state is memory only.

| Variable | Default | Meaning |
|---|---|---|
| `GCS_ACCESS_TOKEN` | none; required | OAuth2 bearer token, e.g. `$(gcloud auth print-access-token)`. Tokens expire (about 1 h); service-account JWT flow is not implemented |
| `GCS_ENDPOINT` | `https://storage.googleapis.com` | |
| `GCS_PROJECT_ID` | `clio-prototype` | used only when creating the bucket |

## daos

- Needs `-DCLIO_ENABLE_DAOS=ON` (Spack: `iowarp +daos`) and the DAOS client
  libraries (libdaos, libdfs, libgurt). The device is one file of a DAOS POSIX
  container, driven through libdfs; no dfuse mount in the data path.
- `pool_name`: `daos://<pool>/<container>/<file path>[?key=value&...]`.
  The container must exist and be POSIX (`daos cont create --type POSIX`).
  Missing directories are created. A CTE tier on node 3 uses
  `<file>_node3`.
- `capacity` is required (or an existing non-empty file); it is a cap,
  nothing is preallocated. `growth_unit` is ignored.
- The process needs the normal DAOS client environment (agent socket, for
  example `DAOS_AGENT_DRPC_DIR`).

URI options:

| Option | Default | Meaning |
|---|---|---|
| `chunk_size` | `1m` | DFS chunk size; applies when the file is created |
| `oclass` | container default | DAOS object class (`SX`, `EC_16P2GX`, ...); at creation |
| `alloc_log` | none | **local** path of the allocator WAL. The `alloc_log` key wins if both are set |
| `prezero` | `1` | zero the destination before each read so holes read as zeros; `0` only on a DAOS version known to zero holes |
| `chunk_align` | `1` | blocks of at least `chunk_size` start on a chunk boundary (unaligned 1 MiB writes measured 3.4x slower) |
| `io_threads` | `CLIO_DAOS_IO_THREADS`, else 4 | DAOS I/O threads, 1-64, shared by every DAOS device of the process; the first device opened decides |

| Variable | Default | Meaning |
|---|---|---|
| `CLIO_DAOS_IO_THREADS` | 4 | thread count when `io_threads` is not given |
| `CLIO_DAOS_POLL_US` | 200 | event-queue poll wait in us while I/O is in flight; 0 spins |

Without a local `alloc_log`, a restart that keeps CTE metadata can hand out
blocks that are still live. Always set one for a persistent tier.

## Other environment variables (all types; read when the pool is created)

| Variable | Effect |
|---|---|
| `CLIO_BDEV_AIO=posix\|default` | force POSIX AIO, or the build default, for file bdevs |
| `CLIO_BDEV_STRIPE_FILES=<N>` | stripe each file bdev over N files `<path>.s0` .. `.s<N-1>` (1-64, default 1). Keep it the same across restarts |
| `CLIO_BDEV_STRIPE_UNIT=<size>` | stripe unit, default 1 MiB; must be a multiple of the direct-I/O alignment |
| `CLIO_PREFAULT=0\|<size>` | ram: pre-fault the whole mapping (`0`) or a prefix at creation |
| `CLIO_NUMA_INTERLEAVE=0` | ram: do not interleave pages across NUMA nodes |
| `CLIO_BDEV_FORCE_SYNC=1` | hbm/pinned: synchronous GPU copies (debugging) |
| `CLIO_BDEV_THROTTLE_MBPS=<n>` / `CLIO_BDEV_THROTTLE_MATCH=<substr>` | synthetic bandwidth cap on bdevs whose pool name contains the substring (experiments only) |
| `CLIO_BDEV_IO_TRACE=<period>` | log every Nth bdev operation |
