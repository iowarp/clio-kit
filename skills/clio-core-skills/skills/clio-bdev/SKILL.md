---
name: clio-bdev
description: 'Use when configuring CLIO Core block devices: a clio_bdev pool (ram, file, hbm, pinned, noop, s3, gcs, daos), a CTE storage tier entry, capacity, persistence_level, alloc_log and --fresh, or a clio_safe_bdev erasure-coded array (members, parity, max_failures, rebuild). Triggers on "bdev_type", "capacity_limit", "alloc_log", "persistence_level", "safe_bdev", "parity disk", "S3 tier". Not for designing the whole chain (use storage-compose) or CTE core keys (use clio-cte-core).'
metadata:
  bundle: clio-core
  servers: none
  provenance: designed
  eval-status: scenarios-recorded
---

# clio-bdev: block devices and the safe_bdev array

Facts here were checked against clio-core `dev` (bd1dc3183) on 2026-11-21.

## What it is

A **bdev** pool (`clio_bdev`) is a flat byte space with a block allocator in
front of it. It hands out extents, moves bytes in and out of them, frees them,
and reports how much room is left. It knows nothing about blobs or files.
Every CTE storage tier is one bdev pool. It is a leaf: it calls nothing below it.

A **safe_bdev** pool (`clio_safe_bdev`) looks like a bdev to everything above
it. It spreads bytes over several member bdevs and keeps Cauchy Reed-Solomon
parity, so it survives up to `max_failures` dead members. It costs one parity
write per data write and dedicated parity disks. See [SAFE_BDEV.md](SAFE_BDEV.md).

Most users never call either directly. A CTE core creates one bdev per
`storage:` entry, and clients use the CTE API.

## Files in this skill

| File | Contents |
|---|---|
| `SKILL.md` | availability, types, every key (standalone and tier entry), guarantees, pitfalls, verification |
| [TYPES.md](TYPES.md) | per-type detail: ram, file, hbm, pinned, noop, s3, gcs, daos; their env vars and build options |
| [SAFE_BDEV.md](SAFE_BDEV.md) | the erasure-coded array: keys, layout, failure injection, rebuild, guarantees |
| [API.md](API.md) | C++ client API for both modules, the Python helpers, return codes |
| [evals.md](evals.md) | test scenarios |

## Availability

| Module | `mod_name` | Default / conventional pool id | In the pip wheel | Build option |
|---|---|---|---|---|
| bdev | `clio_bdev` | `301.0` (the required default RAM bdev `ram::chi_default_bdev`). CTE tier bdevs get `<core major + device index>.<1 + node>`, e.g. `512.1` | yes | always built |
| safe_bdev | `clio_safe_bdev` | none; `350.0` in the reference config | **no** | always built in a source / conda / Spack build; the `clio_safe_bdev_recover` tool also ships only there |

Bdev types inside the wheel:

| Type | CPU wheel `iowarp-core` | GPU wheels (`iowarp-core-cuda12`, `-rocm6`, `-sycl`) | Needs |
|---|---|---|---|
| `ram`, `file`, `noop` | yes | yes | nothing |
| `hbm`, `pinned` | accepted, but **host memory** (a warning is logged for `hbm`) | real device / page-locked memory on CUDA and ROCm; `hbm` under SYCL is host memory | a GPU build |
| `s3` | no | no | source build with `-DCLIO_ENABLE_AMAZON_DRIVE=ON` and Poco |
| `gcs` | no | no | source build with `-DCLIO_ENABLE_GOOGLE_CLOUD=ON` and Poco |
| `daos` | no | no | `-DCLIO_ENABLE_DAOS=ON` (Spack: `iowarp +daos`) and the DAOS client libraries |

A type that is not compiled in fails pool creation (return code 1, log
"Failed to create bdev transport", or for `daos` a message naming the option).

## When to use which

| Need | Use |
|---|---|
| Fast tier that may be lost on restart | `ram` (or `hbm` for GPU-resident data) |
| Tier that survives a process restart | `file` with `persistence_level: temporary` or `long_term` |
| Tier that survives a dead disk on one node | `safe_bdev` over several `file` members |
| Capacity beyond local disks | `s3`, `gcs`, `daos` |
| Staging buffer for GPU DMA | `pinned` |
| Measuring runtime overhead with no I/O | `noop` |

Do not compose a bdev by hand just to add a CTE tier; add a `storage:` entry.
Compose one by hand when it is a safe_bdev member, when you need a key the
tier entry does not pass (`alloc_log`, `io_depth`, `alignment`, `perf_metrics`,
`alloc`, `populate_unit`), or for the required default at `301.0`.

## Two ways to get a bdev

### 1. Standalone compose entry (parsed by bdev `CreateParams::LoadConfig`)

```yaml
compose:
  - mod_name: clio_bdev                  # required default, keep it
    pool_name: "ram::chi_default_bdev"   # memory types: any unique name
    pool_query: local
    pool_id: "301.0"
    bdev_type: ram
    capacity: "0g"                       # 0 on ram = 80% of system DRAM
  - mod_name: clio_bdev
    pool_name: "/mnt/nvme/disk0.bin"     # file type: pool_name IS the path
    pool_query: local
    pool_id: "302.0"
    bdev_type: file
    capacity: "100GB"
    persistence_level: long_term
    alloc_log: "/var/lib/clio/disk0.alloc_log"   # optional, see Durability
    growth_unit: "1GB"
    io_depth: 32
    alignment: 4096
```

| Key | Type | Default | Meaning and constraints |
|---|---|---|---|
| `pool_name` | string | required | `file`: backing file path. `ram`/`hbm`/`pinned`/`noop`: any unique name (convention `ram::<name>`). `s3`/`gcs`: `bucket/prefix`. `daos`: `daos://pool/cont/file[?opts]` |
| `pool_id` | `"major.minor"` | required | must be unique; compose it before every pool that names it |
| `pool_query` | string | required | use `local` |
| `bdev_type` | `file` `ram` `hbm` `pinned` `noop` `s3` `gcs` `daos` | `file` | an **unknown string is silently ignored** and the pool becomes `file` |
| `capacity` | size string | `0` | 0 means: ram/hbm/pinned 80% of system DRAM; file the existing file's size, else 1 GiB; s3/gcs 1 TiB; daos the existing file's size, else refused |
| `persistence_level` | `volatile` `temporary` `long_term` | unset (treated as volatile, but the allocator log is kept) | only an explicit `volatile` turns the file allocator log off; other values are labels. Unknown values are ignored |
| `alloc_log` | path | unset | allocator write-ahead log. Unset on `file`: `<pool_name>.alloc_log`. Unset on `daos`: no log. Ignored by ram/hbm/pinned/s3/gcs |
| `growth_unit` | size string | `1GB` | file only: how far the backing file grows (and reserves disk) at a time. `0` grows exactly as needed |
| `io_depth` | integer | `32` | async I/O queue depth per worker (file) |
| `alignment` | integer bytes | `4096` | allocation granularity. Memory types turn the default 4096 (or 0) into 512 |
| `alloc` | `auto` `eager` `lazy` | `auto` | ram/pinned page commit. `auto` preallocates only when `capacity` is non-zero; eager on a capacity over half the memory budget degrades to lazy with a warning |
| `populate_unit` | size string | `64MB` | ram only: bytes pre-faulted each time allocation crosses the populated mark; `0` = pure lazy faulting |
| `perf_metrics` | map | set by type | `read_bandwidth_mbps`, `write_bandwidth_mbps`, `read_latency_us`, `write_latency_us`, `iops`. Placement ranks tiers by these. Type defaults: hbm 1,500,000 MB/s; ram 20,000 MB/s; pinned 12,000 MB/s; everything else 100 MB/s read / 80 MB/s write, 1000 us / 1200 us, 1000 IOPS |
| `restart` | bool | false | compose-level: record the pool in the pool log so `clio-run start` recreates it (pools in the server config are recreated anyway) |

Size strings (`capacity`, `growth_unit`, `populate_unit`): a number with an
optional suffix whose first letter is k, m, g, t or p, case-insensitive,
binary (1 KB = 1024 B). Decimals work (`1.5g`). An unparseable suffix in a
standalone entry **exits the runtime**.

### 2. A CTE `storage:` entry (parsed by CTE `Config::ParseStorageConfig`)

```yaml
  - mod_name: clio_cte_core
    pool_name: cte_main
    pool_query: local
    pool_id: "512.0"
    storage:
      - path: "ram::cte_ram_tier1"
        bdev_type: ram
        capacity_limit: "0g"
        score: 1.0
      - path: "${CLIO_STORAGE_ROOT}/cte_disk_tier.dat"
        bdev_type: file
        capacity_limit: "10GB"
        growth_unit: "64MB"
        score: 0.2
        persistence_level: temporary
      - path: "safe_array"                 # attach to a pool composed earlier
        existing_pool_id: "350.0"
        existing_pool_module: clio_safe_bdev
        score: 0.4
        persistence_level: long_term
```

| Key | Type | Default | Meaning and constraints |
|---|---|---|---|
| `path` | string | required unless attaching | becomes the bdev `pool_name` with `_node<N>` appended (`cte_disk_tier.dat_node0`). `~` and `${VAR}` are expanded. For s3/gcs use `bucket/prefix`, never a bare bucket |
| `bdev_type` | same 8 values | required unless attaching | an unknown value is a config error (the core is not created) |
| `capacity_limit` | size string | required unless attaching | becomes the bdev `capacity`. `0`/`0g` is allowed **only for ram** (80% of DRAM); 0 on any other type is a config error. Suffixes: b, k/kb, m/mb, g/gb, t/tb, p/pb (`16GiB` is rejected) |
| `score` | float 0.0-1.0 | -1 (automatic) | read by the CTE placement engine, not the bdev. Out of range is a config error |
| `persistence_level` | `volatile` `temporary` `long_term` | `volatile` | passed to the bdev and used by the CTE: a persistent core refuses a volatile tier (rc 16), and a core with no `durability` key is persistent only if every tier is temporary/long_term |
| `growth_unit` | size string | `1GB` | file tiers only; passed through |
| `existing_pool_id` | `"major.minor"`, major != 0 | unset | attach to a pool composed earlier (a safe_bdev, or a hand-made bdev) instead of creating one. `path`, `bdev_type`, `capacity_limit` become optional |
| `existing_pool_module` | string | unset | label for logs only (`clio_safe_bdev`, `clio_bdev`); routing is by pool id |

A tier entry has **no** `alloc_log`, `io_depth`, `alignment`, `perf_metrics`,
`alloc` or `populate_unit`. The CTE creates every tier bdev with io_depth 32,
alignment 4096, type-default perf figures and the default allocator log
(`<path>_node<N>.alloc_log` for file tiers). To set those keys, compose the
bdev yourself and attach it with `existing_pool_id`.

Memory tiers (ram, hbm, pinned) register on their own node only. File and
object-store tiers span the core's `neighborhood`.

## Where it goes in the chain

- A bdev names no pool. It must appear in `compose:` **before** anything that
  names it: a safe_bdev that lists it as a member, or a CTE tier that attaches
  to it with `existing_pool_id`.
- A safe_bdev comes after all its members and before the CTE core that
  attaches to it.
- Keep `num_replicas: 0` in any replication / CTE setting above a safe_bdev
  tier. The array already survives dead disks; a second copy on the same array
  protects nothing and doubles the bytes written.

## Durability of the allocator

The allocator decides which bytes are live. If it forgets, it hands live
blocks out twice and corrupts data. Each type keeps it differently:

| Type | Allocator state across a restart |
|---|---|
| `file` | WAL at `<pool_name>.alloc_log` (or `alloc_log`). Every alloc/free is appended before blocks are handed out, flushed to the kernel on every call, fsynced every 50 ms. Replayed on every recovering `clio-run start` |
| `file` with explicit `persistence_level: volatile` and no `alloc_log` | no WAL; an old default log is deleted |
| `daos` | no WAL unless `alloc_log` (or `?alloc_log=` in the URI) names a **local** path. Without one, a restart that keeps CTE metadata can hand out live blocks again |
| `s3`, `gcs` | memory only; objects persist but the allocator restarts empty |
| `ram`, `hbm`, `pinned`, `noop` | nothing persists, contents included |

`clio-run start --fresh` deletes a file bdev's **default** log (the allocator
starts empty) but not the backing file. A log set with `alloc_log` is
replayed **even after `--fresh`**, so its blocks stay allocated while the CTE
metadata that referenced them is gone: capacity leaks until you delete that
log by hand. The same holds for a safe_bdev's `alloc_log`.

## Guarantees (plain bdev)

An acknowledged `Write` means the I/O completed into the device. For `file`
that is the kernel page cache (aligned writes of 64 KiB or more go direct),
**not** stable storage. Data is fdatasynced only by `Sync`, which the CTE's
fsync path calls.

| Event | ram / hbm / pinned | file (temporary / long_term) | s3 / gcs | daos |
|---|---|---|---|---|
| client crash | nothing lost | nothing lost | nothing lost | nothing lost |
| runtime crash (kill -9) | all contents | nothing acked is lost; allocator recovered | objects kept; allocator lost | data kept; allocator lost unless `alloc_log` |
| power loss | all | writes not yet `Sync`ed; allocator records of the last 50 ms | objects kept (completed PUTs); allocator lost | as kill -9, plus unsynced local `alloc_log` |
| disk loss | n/a | everything on that disk; use safe_bdev or CTE replication | n/a | depends on the DAOS object class |
| node lost temporarily | lost | unavailable until it returns | reachable from no node (bdev is node-local) | same |
| node lost permanently | lost | lost | objects remain in the bucket but nothing maps them | same |
| `clio-run start --fresh` | lost | default log deleted; bytes orphaned | allocator empty | allocator empty |

Whether the bytes are **findable** after a restart is the CTE's job: it needs
`metadata_log_path` on a persistent core. A bdev alone keeps no map from blob
to block.

## Pitfalls and failure signatures

| Symptom | Cause | Fix |
|---|---|---|
| Data gone after restart although it was on a file tier | tier had no `persistence_level`, so the core was inferred volatile and starts empty | set `persistence_level: temporary` or `long_term` on every tier of the persistent core, plus `metadata_log_path` |
| CTE log "refusing volatile bdev ... this core is persistent", rc 16 | `durability: persistent` core with an unlabeled or `volatile` tier (an attached safe_bdev too) | label the tier |
| "Config error: Storage device capacity_limit must be greater than 0" | `0`/`0g` on a non-ram tier | give a size |
| RAM tier much larger than expected, or OOM | `capacity: 0g` = 80% of DRAM, per tier | give an explicit size; several `0g` tiers each claim 80% |
| "Failed to open bdev file: /tmp/... (Permission denied)", then "Failed to create bdev container" and puts refused | backing file or `.alloc_log` left by another user (often root) | `chown` the files or choose a new path |
| A `bdev_type: rma` pool appears as a file named `ram::x` | standalone `bdev_type` typo falls back to `file` | spell the type; check the log line naming the transport |
| Runtime exits at start: "Could not parse the size" | bad suffix in a standalone size string | use k/m/g/t/p |
| s3 pool fails: "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are not set" | credentials not in the **runtime's** environment | export them before `clio-run start`; see TYPES.md |
| s3 tier writes to bucket `mybucket_node0` | `path` was a bare bucket | use `mybucket/prefix` |
| "HBM bdev requested in a build with no GPU support" | `hbm` in a CPU wheel | install a CUDA/ROCm wheel or use `ram` |
| hbm: "cudaMalloc of N bytes failed" (fatal) | capacity larger than free device memory | lower `capacity_limit`; hbm never falls back to host |
| Capacity leaks after `--fresh` | explicit `alloc_log` replayed | delete the log file before `--fresh` |
| Each file tier eats 1 GiB of RAM on tmpfs | first `growth_unit` is reserved at creation | `growth_unit: "64MB"` |
| DAOS warning "recovering start without an allocator log" | no local `alloc_log` | add one |
| Several file tiers on one disk all report the disk's free space | remaining space is an upper bound per device | budget `capacity_limit` yourself |

Test-only fault injection: while a file named `<backing file>.fail` exists,
every read, write and sync of that file bdev fails with a device fault; with
`<backing file>.nospace` every write fails with ENOSPC. Checked every 100 ms.
Delete the marker and the device works again. Off unless the file exists.

## Verify it works

```bash
clio-run start                 # in another terminal
clio-run compose list          # the bdev pool ids should be listed
# Dashboard (served by `clio-run start` on 127.0.0.1:8080 by default):
curl -s http://127.0.0.1:8080/api/mod/clio_bdev/pools
curl -s "http://127.0.0.1:8080/api/pools/301.0/monitor?query=stats&routing=local"
```

The `stats` reply holds `pool_name`, `bdev_type` (0 file, 1 ram, 2 hbm,
3 pinned, 4 noop, 5 s3, 6 gcs, 7 daos), `total_capacity`,
`remaining_capacity`, bandwidth, latency and I/O counters. The Pools page of
the dashboard shows the same.

For CTE tiers, from Python:

```python
import os
os.environ["CLIO_WITH_RUNTIME"] = "0"     # attach to the running runtime
import clio_cte_core_ext as cte
assert cte.clio_init(cte.RuntimeMode.kClient, False)
assert cte.initialize_cte("", cte.PoolQuery.Dynamic())
c = cte.get_cte_client()
print(c.AsyncListTargets(cte.PoolQuery.Local()).result())   # e.g. [..._node0]
cap = c.AsyncGetCapacity(cte.PoolQuery.Local()).result()
print(cap.total_capacity, cap.remaining_capacity)
```

Durability check: write data, `kill -9` the runtime, run `clio-run start`
(never `--fresh`), read it back. The log should say
"bdev <path>: recovered N live blocks from <path>.alloc_log".
