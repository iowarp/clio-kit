---
name: clio-cte-core
description: 'Use when configuring, calling or debugging the Context Transfer Engine core (clio_cte_core): the persistent core at 512.0, the volatile core at 540.0, tiers, scores, the metadata WAL and the C++/Python blob API (Tag, PutBlob, GetBlob, TagQuery, SyncTag) every chain speaks. Triggers on "clio_cte_core", "512.0", "540.0", "PutBlob rc=11", "metadata_log_path", "clio_cte_core_ext", "persistence_level". Not for choosing or ordering the chain; use storage-compose. Not for bdev keys; use clio-bdev.'
metadata:
  bundle: clio-core
  servers: none
  provenance: designed
  eval-status: scenarios-recorded
---

# CTE core (`clio_cte_core`)

The core is the storage engine at the bottom of every Context Transfer Engine
(CTE) chain. It stores **blobs** (named byte ranges) inside **tags** (named
groups, like a bucket or a file) on **targets** (block-device pools, one per
`storage:` entry). It places each blob by its **score** (0.0 slow to 1.0
fast), keeps a metadata write-ahead log (WAL) so a restart can find the bytes
again, and executes every put, get, delete, query and reorganize. Caches,
replication, the indexer, the compressor and the filesystem all speak the
core's own task interface and end at a core. It costs one runtime container
per node, the metadata tables in memory, and one bdev pool per tier.

Facts here were checked against clio-core `dev` at bd1dc3183 (2026-11-21).

## Files in this skill

| file | read when |
|---|---|
| **CONFIG.md** | writing or reviewing a core entry. Every key `LoadConfig` reads, types, defaults, constraints, environment variables. |
| **API.md** | calling the core. C++ `Client` and `Tag` signatures, pool queries, return codes, the Python module `clio_cte_core_ext`, targeting another pool id. |

## Availability

| item | value |
|---|---|
| `mod_name` | `clio_cte_core` |
| default pool ids | `512.0` persistent core (`kCtePoolId`, pool name `cte_main`); `540.0` volatile core (pool name `cte_volatile`) |
| pip wheel | yes: the runtime library, and the Python client `clio_cte_core_ext` |
| other installs | conda, Spack, deb and Docker images ship it too. It is always built when `CLIO_CORE_ENABLE_CTE=ON` (the default) |
| C++ headers | not in the wheel. C++ clients need a source, conda or Spack install |

## Concepts

- **Tag.** A named container. Its id is `(major, minor)`. A tag name hashes to
  the container that holds its name binding. A tag's size is split across the
  containers holding its blobs.
- **Blob.** Addressed by `(tag id, blob name)`. It has a list of blocks, each
  an extent on one target. It can span several targets.
- **Owner.** Each blob has exactly one owner container, chosen by
  `hash(tag, blob)` over the containers. Every operation on that blob goes to
  the owner, which serializes its writes. There is no metadata server.
- **Target (tier).** A bdev pool that the core created from a `storage:`
  entry, or an existing pool it attached (`existing_pool_id`). Each target has
  a score, a capacity and a `persistence_level` (`volatile`, `temporary`,
  `long_term`).
- **Score and placement.** A new blob without a score gets 1.0. A put with
  score -1 keeps the blob's current score. The `max_bw` placement engine
  prefers the fastest target whose score is not above the blob's score, and
  spills to others when full. `ReorganizeBlob` sets a new score and moves the
  blob. The optional organizer rescans and rescores periodically.
- **Durability class.** Each core is `volatile` or `persistent` (key
  `durability`). A persistent core refuses volatile devices (rc 16). A
  volatile core accepts persistent devices with a warning, writes no WAL and
  is empty after every start. Moving data between the two is the write-back
  cache's job (`clio-cte-caches`).
- **Metadata WAL and snapshot.** Persistent core only, and only when
  `performance.metadata_log_path` is set. The WAL is sharded per worker
  (`<path>.blob.<N>`, `<path>.tag.<N>`), with one sequence number across the
  shards; replay merges them in sequence order. The snapshot is the file
  `<path>` itself, rebuilt when the WAL exceeds `transaction_log_capacity`
  (32 MiB) or at most once a minute when score changes are pending.
- **Droppable blobs.** A put with flag `kCtePutDroppable` (0x2) marks the blob
  expendable. Only a droppable put may evict, and only droppable blobs are
  evicted to make room. The flag is fixed at creation: a later put without it
  is refused with rc 14.
- **Replica slots.** A blob can hold extra block lists besides its primary.
  The core only stores them; modules above decide policy.
  `REPLICA_PERSISTENT` (0x2) slots never land on volatile tiers;
  `REPLICA_CACHE` (0x4) is the slot the cache module (563.0) fills with a
  node-local, untransformed copy, dropped under pressure and at restart. The
  replication module's `num_replicas` (extra same-node copies) and
  `remote_copies` use these slots; see `clio-replication`.
- **`failover_to_successor`.** While a blob's owner node is dead, route its
  operations to the next live container, which serves the replication
  module's remote copy. Off by default: operations on a dead owner's blobs
  fail until it returns.
- **`read_cache`.** Not a core key on dev. The core's parser ignores it
  silently. Use the write-back cache's `fill` policy or the cache module for
  read copies.

## When to compose it

- Always, if you use CTE at all. The default config has two: a persistent core
  (512.0, file tier, WAL) and a volatile core (540.0, DRAM), joined by the
  write-back cache.
- Use **one persistent core alone** (no caches) when every ack should already
  be on a persistent device and you do not need a DRAM tier.
- Use **one volatile core alone** for scratch data that may vanish at restart.
- **Do not** put a DRAM or hbm tier and a durable tier in the same core.
  Compose two cores and a policy cache instead (pitfalls below).

## Compose entries (default config)

Both entries come straight from the shipped `clio_default.yaml` (comments
trimmed). `${CLIO_STORAGE_ROOT}` defaults to `~/.clio`.

```yaml
compose:
  # Persistent core: the canonical CTE entry point (kCtePoolId).
  - mod_name: clio_cte_core
    pool_name: cte_main
    pool_query: local
    pool_id: "512.0"
    durability: persistent
    storage:
      - path: "${CLIO_STORAGE_ROOT}/cte_disk_tier.dat"
        bdev_type: "file"
        capacity_limit: "10GB"
        score: 0.2
        persistence_level: "temporary"
    performance:
      metadata_log_path: "${CLIO_STORAGE_ROOT}/cte_metadata_log"
      transaction_log_capacity: "32MB"
    dpe:
      dpe_type: "max_bw"
    targets:
      neighborhood: 1
      default_target_timeout_ms: 30000
      poll_period_ms: 5000
      # failover_to_successor: true    # with replication remote_copies >= 1

  # Volatile core: DRAM, no WAL, empty after every start.
  - mod_name: clio_cte_core
    pool_name: cte_volatile
    pool_query: local
    pool_id: "540.0"
    durability: volatile
    storage:
      - path: "ram::cte_ram_tier1"
        bdev_type: "ram"
        capacity_limit: "0g"            # 0g = 80% of system DRAM
        score: 1.0
    dpe:
      dpe_type: "max_bw"
    targets:
      neighborhood: 1
```

Key summary (full table with types and constraints in **CONFIG.md**):

| key | default | meaning |
|---|---|---|
| `durability` | inferred | `volatile` or `persistent`. Inferred: persistent only if there are devices and all are `temporary`/`long_term` |
| `storage[]` | none | tiers: `path`, `bdev_type`, `capacity_limit` (required), `score`, `persistence_level`, `growth_unit`, `existing_pool_id`, `existing_pool_module` |
| `performance.metadata_log_path` | empty (no WAL) | WAL and snapshot path prefix. **Required for a persistent core to survive restart** |
| `performance.transaction_log_capacity` | `32MB` | WAL size that triggers a snapshot |
| `performance.flush_metadata_period_ms` | 5000 | WAL fsync period (ms) |
| `performance.fsync_mode` | `durable` | `deferred` makes `SyncTag` return at once |
| `performance.stat_targets_period_ms` | 1000 | target free-space refresh (ms), 10-60000 |
| `dpe.dpe_type` | `max_bw` | `max_bw`, `round_robin`, `random` |
| `targets.neighborhood` | 4 (code); shipped config sets 1 | nodes whose persistent tiers each container registers |
| `targets.failover_to_successor` | false | serve a dead owner's blobs from its successor |
| `organizer`, `organizer_tasks`, `organizer_period_ms` | `none`, 1, 5000 | periodic rescoring. Top level of the entry, not under `performance` |

The core creates one bdev pool per `storage[]` entry with pool id
`<core major + device index>.<1 + node>`. So two cores on one runtime never
share a device, but each device needs its own `path`.

## Where it goes in the chain

- A core names no other pool, so it can be composed first (after the
  `clio_bdev` at 301.0). Every module that names it with `next_pool_id`,
  `next_volatile` or `next_persistent` must come **after** it.
- Default chain: indexer 564 -> shm cache 568 -> write-back 567 -> volatile
  core 540, and write-back -> replication 561 -> persistent core 512.
- Clients normally address the **top** of the chain (564.0), not 512.0. A
  client that writes 512.0 directly while a write-back cache is composed
  bypasses that cache: the cache can then hold a stale or dirty copy of the
  same blob. Address a core directly only when nothing sits above it, or for
  read-only inspection.
- Never compose CAE (`clio_cae_core`) at 512.0: its method ids collide with
  the core's.

## Guarantees of a core used directly (nothing above it)

| event | persistent core (512, file tier, WAL on) | volatile core (540) |
|---|---|---|
| **ack of PutBlob** | bytes written to the owner's device (in the OS page cache, not fdatasynced); WAL record written to the OS after the data | bytes in DRAM on the owner |
| client process crash | nothing acked is lost | nothing acked is lost |
| runtime crash (`kill -9`) | nothing acked is lost: data and WAL records are in the kernel. Up to ~1 min of score changes is lost | everything |
| power loss | device writes not yet synced (kernel writeback; CLIO sets no bound) and WAL records not yet fsynced (up to `flush_metadata_period_ms`, 5 s) | everything |
| disk loss | everything on that device | n/a |
| node down, temporarily | blobs it owns are unavailable until it returns (unless replication + `failover_to_successor`) | same, then empty after restart |
| node lost permanently | every blob it owned | every blob it owned |
| `clio-run start --fresh` | everything: the WAL shards and snapshot are deleted, bdev alloc logs too (unless `alloc_log` is set) | (always empty) |

- **Power-loss point:** `SyncTag(tag)` syncs every non-volatile device holding
  the tag's blocks, then fsyncs the WAL. It fails with rc 5 (EIO) or 28
  (ENOSPC) rather than returning early. With `fsync_mode: deferred` it returns
  at once with `deferred_ = 1`. On a volatile core it syncs nothing useful.
  `SyncTag` is C++ only; the Python module has no binding for it.
- **Graceful `clio-run stop`** closes (and syncs) the WAL.
- **Without `metadata_log_path`** the persistent core logs nothing: the bytes
  survive on disk but nothing records which blob they belong to.
- **After restart**, blocks that were on volatile targets are dropped and
  marked lost. Reading into a lost range returns rc 62 (I/O error), never
  zeros. Cache copies (`REPLICA_CACHE`) are dropped.
- **Consistency:** single owner per blob, so a reader on any node sees the
  latest acked write to that blob. A put is all-or-nothing for growth; a
  failed overwrite inside the old size can leave partial bytes.

## Client API (summary)

Full signatures, every return code and the complete Python reference are in
**API.md**.

- C++: `clio::cte::core::Client` (one per pool id) and the `Tag` wrapper,
  which always uses the process-wide client `CLIO_CTE_CLIENT`.
- Python: `import iowarp_core` then `import clio_cte_core_ext as cte`.
  `cte.Tag`, `cte.Client`, `cte.PoolQuery`, `cte.PoolId`.
- Blob verbs (put, get, delete, size, score, reorganize) route to the owner
  whatever pool query you pass. Queries (`TagQuery`, `BlobQuery`) default to
  **Broadcast**. `GetContainedBlobs`, `GetTagSize` and `Evict` answer for one
  container unless you pass Broadcast.
- **Which pool a client talks to:** `initialize_cte()` / `CLIO_CTE_CLIENT_INIT()`
  binds the process client to the shm cache's entry pool (564.0 in the
  default config) when the client is attached by shared memory, **otherwise
  to 512.0 directly**. Set `CLIO_CTE_POOL=564.0` before init to pin it. For
  any other pool, construct `Client(PoolId(major, minor))`.

```python
import os
os.environ.setdefault("CLIO_WITH_RUNTIME", "0")  # attach to a running clio-run
os.environ.setdefault("CLIO_CTE_POOL", "564.0")  # always the chain top
import iowarp_core                               # sets library paths
import clio_cte_core_ext as cte

assert cte.clio_init(cte.RuntimeMode.kClient, False)
assert cte.initialize_cte("", cte.PoolQuery.Dynamic())
tag = cte.Tag("demo/run1")                       # GetOrCreateTag
tag.PutBlob("step0", b"hello world", 0)          # raises on rc != 0
print(tag.GetBlob("step0", tag.GetBlobSize("step0"), 0))
```

## Pitfalls and failure signatures

| symptom | cause | fix |
|---|---|---|
| every `PutBlob` returns rc 11; `GetCapacity` is 0 | no target registered on the owner: a `storage[]` entry failed (bad path, YAML error, refused volatile device) | read the runtime log for `Config error`, `refusing volatile bdev`, `Failed to open bdev file`; fix the entry; check `AsyncListTargets` |
| rc 16 at registration, then rc 11 | a volatile device (no `persistence_level`, or `ram`/`hbm`/`pinned`) in a `durability: persistent` core | label disk tiers `temporary` or `long_term`; move memory tiers to the volatile core |
| data gone after restart, no error | a `file` tier without `persistence_level` and no `durability` key: the core inferred **volatile**, wrote no WAL, started empty | set `persistence_level: temporary` and `durability: persistent` |
| bytes on disk after restart but blobs missing | persistent core without `performance.metadata_log_path` | set it; the WAL then records every layout |
| primaries silently lost at restart or power loss | DRAM/hbm tier at score 1.0 next to a `long_term` disk in one core: new blobs score 1.0 and land on memory | two cores (volatile + persistent) and the write-back cache |
| hbm tier corrupts data or is refused | hbm used anywhere but as an ordinary score-1.0 tier of the volatile core (for example as a manual loader, or in a persistent core) | keep hbm a normal volatile-core tier; see `clio-bdev` |
| `MultiPutBlob` writes land on the wrong node or fail | `AsyncMultiPutVectored` routes by the **first** record's owner and assumes all records share it | group records per owner, one batch each |
| `GetContainedBlobs` / `GetTagSize` / `Evict` / `BlobQuery` with Local miss data | these answer for one container | pass `PoolQuery.Broadcast()` |
| Python client writes bypass the cache in front of the core | TCP/IPC client fell back from the shm cache to 512.0 | `CLIO_CTE_POOL=564.0`, or `Client(PoolId(564, 0))` |
| rc 14 on a put | the blob was created droppable; this put is not | use another name, or keep the flag |
| reads return rc 62 after restart | the range lived on a volatile tier and is gone | expected; write durable data to a persistent core |
| `organizer` setting has no effect | key placed under `performance:` | move it to the top level of the entry |
| WAL keeps growing, `metadata WAL is N bytes` error | snapshots are not finishing | check disk space at `metadata_log_path`; restart replays all of it |

## Verify it works

1. `clio-run status` shows the runtime up. The dashboard Pools page lists
   `cte_main` at 512.0 and `cte_volatile` at 540.0.
2. The runtime log shows `cte_core Create (cte_main): persistent core
   (stated), WAL on` and `WAL: N blob and N tag transaction logs (open)`.
3. Run this against the core itself (pool 512.0) on an idle test system:

```python
import os
os.environ.setdefault("CLIO_WITH_RUNTIME", "0")
import iowarp_core
import clio_cte_core_ext as cte

assert cte.clio_init(cte.RuntimeMode.kClient, False)
core = cte.Client(cte.PoolId(512, 0))
print("targets:", core.AsyncListTargets().result())
cap = core.AsyncGetCapacity(cte.PoolQuery.Broadcast()).result()
print("capacity bytes:", cap.total_capacity, "free:", cap.remaining_capacity)
tid = core.AsyncGetOrCreateTag("verify/core").result()
assert core.PutBlob(tid, "b0", b"x" * 4096) == 0, "put failed (rc 11 = no target)"
assert core.GetBlob(tid, "b0", 4096) == b"x" * 4096
print("blobs:", core.AsyncGetContainedBlobs(tid).result())
assert core.DelBlob(tid, "b0")
```

4. Durability check: put a blob, `kill -9` the runtime, `clio-run start`
   (without `--fresh`), and read it back.
