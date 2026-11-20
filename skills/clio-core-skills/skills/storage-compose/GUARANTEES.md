# GUARANTEES.md: consistency, durability, fault tolerance

State a deployment's guarantees in these terms and no looser ones. Each row
assumes the dev default chain unless noted. "Owner" is the container that
`hash(tag, blob)` selects. There is exactly one owner per blob, and it serializes
that blob's writes.

## What an acknowledged put means

| top of chain | ack means |
|---|---|
| persistent core directly (512) | bytes written to the owner's tier device (not fdatasynced); WAL record flushed to the OS |
| write-back cache (default) | bytes in the owner's **volatile** core. Not yet on persistent storage |
| write-thru cache | bytes written through replication to the persistent core |
| replication, `replication_factor = N > 1` | primary plus N-1 remote copies written (remote copies are synchronous) |
| safe_bdev tier | parity current for every touched stripe |
| shm cache (non-`md_only`) | see the shm cache row under consistency |
| stream append | staged on the writer's node. Merge into the stream happens within ~1 ms; `Flush` waits for it |
| filesystem write | in the client's write-behind staging (read-your-writes from the same mount) |

## When data becomes durable

A byte is durable when three things hold:
1. It is on a tier with `persistence_level` `temporary` or `long_term`.
2. That device has been synced.
3. The metadata WAL that names its blocks has been fsynced.

Under write-back, that happens on any of these:
- **`SyncTag`, or `fsync` on clio-fs** (with `fsync_mode: durable`, the
  default). The call flushes the tag's dirty copies through the persistent
  chain, syncs the devices, then syncs the WAL and the namespace log. It fails
  with ENOSPC or EIO rather than returning early. With `fsync_mode: deferred`,
  fsync returns at once and durability waits for the periodic flush.
- **A periodic flush.** This runs every `flush_period_ms` (10 s default), at
  `flush_fill_pct` (80%), or every `flush_dirty_mb` (256 MiB). The flush
  writes the data, but power-loss safety also needs the WAL fsync, which
  happens every `flush_metadata_period_ms` (5 s).
- **A graceful `clio-run stop`**, bounded by `--grace-period`. A kill or a
  crash skips it.

With no persistent core at all, `SyncTag` means what fsync means on tmpfs:
nothing.

## Loss per failure

| failure | lost (dev default: write-back, replication_factor 1) | how to shrink it |
|---|---|---|
| client process crash | the client's unsent writes. With clio-fs, its write-behind staging | fsync before relying on a write |
| runtime crash (`kill -9`) | dirty volatile copies not yet flushed: up to ~`flush_period_ms` of writes, plus anything not fsynced | write-thru, a lower `flush_period_ms`, or fsync |
| power loss on a node | the above, plus device writes not yet fdatasynced and WAL records not yet fsynced (≤ `flush_metadata_period_ms`) | fsync; `safe_bdev` with `intent_sync: true` |
| one disk of a node | everything on that device | `safe_bdev` (`max_failures` ≥ 1), or `num_replicas` ≥ 1 |
| a whole node, temporarily | **availability** of every blob it owns, until it returns. Data comes back on restart | `replication_factor` ≥ 2 plus `failover_to_successor: true` |
| a whole node, permanently | every blob it owned, and every clio-fs directory it homed (the namespace journal is node-local, #1263) | `replication_factor` ≥ 2. Directories are not covered |
| `clio-run start --fresh` | the WAL, the snapshot, and the bdev alloc logs (unless `alloc_log` is set explicitly). **Everything** | do not use `--fresh` on data you need |

Data that is dropped **by design** at restart or under pressure:
- the volatile core, and anything still dirty in it
- `read_cache` copies and `REPLICA_CACHE` copies
- blobs written as droppable, when space is needed
- clean shm-cache entries
- the replication hand-back record, if `handoff_log_path` is unset and the
  successor restarts

## Consistency

- **Same blob, any nodes.** Writes and reads go to the blob's owner. The
  write-back cache keeps the one dirty copy on the owner, so a reader on any
  node sees the latest acked write. The exception is a reader served from a
  stale cache, below.
- **Caches above the owner:**
  - **shm cache.** The home invalidates the nodes caching an entry before it
    applies a change. With `invalidate_on_flush: true` (the default chain), the
    invalidation happens when the write-back flush lands. Until then, *another*
    node's cached metadata (for example, a size) can be stale. Same-node
    readers are coherent.
  - **cache (563), if composed.** Write-through. The owner invalidates the other
    copies.
  - **FUSE kernel caches.** The attribute TTL is 0 on multi-node mounts.
    `CLIO_FUSE_DIR_ATTR_CACHE_S` and `CLIO_FUSE_READAHEAD_PAGES` trade bounded
    staleness for speed.
- **Search** is read-your-writes: a search first drains the pending index keys.
- **Streams and files.**
  - A stream's size is authoritative at its home.
  - Appends from many nodes are ordered at the home, so the order is
    deterministic but not wall-clock.
  - Files get close-to-open semantics on size.
  - A namespace change is acked only after the home has pushed it to the
    nodes that cache the block.
- **Batches.** `MultiPutBlob` routes by its first put. A batch spanning owners
  is wrong unless the caller groups it per owner.
- **Cluster-wide queries.** `TagQuery`, `BlobQuery`, `GetContainedBlobs`,
  `GetTagSize` and `Evict` answer for one container unless you broadcast them.

## Failure detection and availability

- **A silent peer.** After `CLIO_PROBE_SILENCE_S` (30 s), its tasks fail.
- **An unreachable peer.** Sends are retried for `CLIO_NET_RETRY_TIMEOUT_S`.
  With `CLIO_NET_DEAD_FAIL_FAST=1`, a task addressed to a peer already declared
  dead fails at once with EIO.
- **SWIM is off by default.** When enabled, its timeouts default to about an
  hour before containers are redistributed. Wide collectives starve probe
  replies, so short timeouts falsely kill healthy nodes. This was measured at
  256 nodes.
- **Availability during a node outage** needs both of these. With only
  replication, the copies exist, but operations on the dead owner's blobs still
  fail.
  - replication with `remote_copies` ≥ 1
  - the core's `failover_to_successor: true`
- **clio-fs `fsync`** returns EIO if a peer died during the unsynced window.
  Applications must treat that as lost writes.

## Traps that have caused real data loss

- Mixing volatile and persistent devices in one core: an hbm tier at score
  1.0 next to a `long_term` disk takes primaries that are never made durable. Use two cores and a policy cache.
- A `file` tier with no `persistence_level`. It is volatile, so its blocks are
  dropped at replay.
- A persistent core without `metadata_log_path`. The bytes survive but cannot
  be found.
- A DAOS bdev without a local `alloc_log`, restarted with metadata kept. Live
  blocks can be handed out again.
- Moving an hbm tier into a role other than an ordinary score-1.0 volatile tier
  of the volatile core, for example a manual loader. Durable-to-hbm moves have
  corrupted the store.
