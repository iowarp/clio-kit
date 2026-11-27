# Write-back cache (`clio_cte_core_persist_write_back`, 567.0)

## What it is

The policy cache in the default chain. It sits over two disconnected CTE cores:
a volatile core (DRAM, no WAL, empty after every start) and a persistent chain
(replication over the disk core). A put lands in the volatile core and is acked
at once. Dirty byte ranges are copied to the persistent chain later. It owns
every durability decision in the deployment. The cost is a window of acked but
not yet persistent data.

## Availability

- `mod_name: clio_cte_core_persist_write_back`, default `pool_id` 567.0.
- In the pip wheel. Source builds: `CLIO_CTE_ENABLE_POLICY_CACHES=ON` (default).
- Shares its code and keys with write-thru (`cache_base`).

## When to compose it

- Compose it when you want DRAM-speed writes and fsync-level durability. This
  is the right choice for nearly every deployment with a disk tier.
- Use write-thru instead only when every ack must already be persistent.
- Omit `next_persistent` for a scratch-only (volatile-only) system.

## Compose entry

```yaml
compose:
  # ... core 512.0 (persistent), core 540.0 (volatile), replication 561.0 first
  - mod_name: clio_cte_core_persist_write_back
    pool_name: clio_cte_core_persist_write_back
    pool_query: local
    pool_id: "567.0"
    next_volatile: "540.0"        # volatile core (required)
    next_persistent: "561.0"      # persistent chain; omit for volatile-only
    fill: on_hit_count:2          # on_hit_count:N | on_read_miss | none
    evict: lru                    # lru | lfu
    flush_period_ms: 10000        # periodic flush; 0 = off
    flush_fill_pct: 80            # flush when volatile is this full; 0 = off
    flush_dirty_mb: 256           # flush after this much new dirty data; 0 = off
    flush_pool: "568.0"           # only with the shm cache above (see below)
    fallback_to_persistent: true  # write around a full volatile core
```

### Keys (`PolicyCacheParams::LoadConfig`)

Malformed values are logged (`policy cache: ...`) and the default stays.

| key | type | default | meaning and constraints |
|---|---|---|---|
| `next_volatile` | pool id string `"major.minor"` | none; **required** | the volatile CTE core. Missing: Create fails (rc 1, log `next_volatile is required`). |
| `next_persistent` | pool id string | none | the persistent chain (replication or the persistent core). Omitted: every verb goes to the volatile core, nothing is flushed, `SyncTag` makes nothing durable. |
| `fill` | string | `on_hit_count:2` | when a read miss copies the whole blob into volatile. `on_read_miss`: every miss. `on_hit_count:N`: the Nth miss of a blob (N=0 is treated as 1). `none`: volatile holds only written data. |
| `evict` | string | `lru` | which copy is dropped first when volatile is full: `lru` or `lfu`. |
| `flush_period_ms` | u32, ms | 10000 | period of the background flush of all dirty copies. 0 = no periodic flush. |
| `flush_fill_pct` | u32, % (0-100) | 80 | start a flush when volatile is at least this full and holds dirty data. Checked after writes, at most once per 10 ms. Values > 100 become 100. 0 = off. |
| `flush_dirty_mb` | u32, MiB | 256 | start a flush when this much new dirty data arrived since the last triggered flush. 0 = off. |
| `fallback_to_persistent` | bool | true | when volatile cannot hold a put even after evicting clean copies: true writes it through to persistent; false fails the put with rc 13 (`kPutNoSpaceRc`, ENOSPC). |
| `flush_pool` | pool id string | this cache's own pool | where a flushed range of a blob owned by another node is landed on that owner. Set it to the shm cache (`"568.0"`) when the shm cache above uses `invalidate_on_flush: true`. |

Related CTE **core** keys (set on the persistent core 512.0, under
`performance:`), which decide what fsync and a flush buy:

| core key | default | effect here |
|---|---|---|
| `flush_metadata_period_ms` | 5000 ms | the WAL is fsynced each period. A flushed byte survives power loss only after this (or `SyncTag`). |
| `fsync_mode` | `durable` | `durable`: `SyncTag`/fsync flushes the tag's dirty copies, syncs the devices, then the WAL. `deferred`: the core syncs nothing and replies `deferred_=1`; clio-fs then skips later fsyncs, and durability waits for the periodic flush plus WAL period. |
| `flush_data_period_ms`, `flush_data_min_persistence` | n/a | removed from the core; ignored with a warning. Use `flush_period_ms` here. |

There is no `read_cache` key on dev. Read caching in a two-core layout is this
module's `fill` key.

## Where it goes in the chain

- Names `next_volatile` and `next_persistent` (and `flush_pool`). Compose it
  **after** both cores and the replication pool. If `flush_pool` names the shm
  cache, that pool is created later in the file. This works because
  `flush_pool` is used only at flush time, but the shm cache must exist before
  the first flush.
- Above it: the shm cache (568.0) and/or the compressor (562.0), then the
  indexer (564.0). Below it: nothing else of the cache family.
- The volatile and persistent cores must be **separate** pools. Each blob
  hashes to the same owner container in both.

## How it works

- **Writes.** A plain put is absorbed in volatile **on the node that made it**
  and its byte range is logged dirty. A partial write to a blob this node owns
  first fills the rest of the blob from persistent. A write to a blob another
  node owns is buffered as the written ranges only.
- **Applied at the owner instead:** appends, replaces and conditional puts
  (`kPutIfAbsent`, `kPutIfVersion`), and blobs holding a tag's own metadata
  (clio-fs directories and inodes). The owner first lands any bytes of that
  blob buffered here.
- **Reads go to the blob's owner.** The owner serves its volatile copy, else
  persistent, then fills per `fill`. A write buffered on **another** node is not
  visible until it is flushed. On one node this never matters.
- **Flush triggers:** the periodic timer, `flush_fill_pct`, `flush_dirty_mb`,
  `SyncTag` (that tag), `FlushData`, before a dirty copy is evicted, before any
  verb that must see persistent state (tag size, contained blobs, queries,
  GPU Pod puts), and a graceful stop. Periodic, fill and stop flushes share one
  round. A flush writes only dirty ranges, up to 64 ranges and 256 MiB in flight.
  Ranges owned elsewhere land on the owner in batches of up to 8 MiB.
- **Failed flush.** A blob whose ranges fail stays dirty and is retried by the
  next flush.
- **Eviction.** With `fallback_to_persistent: true`, a full volatile core drops
  clean copies only. If every copy is dirty, new puts go straight to persistent
  (no inline flush of other blobs). Eviction is a DelBlob on the volatile core.
- **`SyncTag` (fsync).** Every container flushes its dirty copies of the tag,
  then the persistent chain syncs its devices and WAL. Concurrent syncs of one
  tag share a round. Failure: rc 28 (ENOSPC) when the store had no room, else
  rc 5 (EIO). The call never returns early on error.
- **Graceful stop.** `clio-run stop` runs a flush bounded by `--grace-period`
  (default 5000 ms). Past that the runtime logs `dirty copies not flushed within
  the N s grace period; unflushed data is lost` and stops.
- **DelTag.** Persistent first, then every container discards its copies of
  the tag (dirty bytes of a deleted tag are dropped, not flushed).

## Guarantees

- **Ack:** bytes are in volatile DRAM (writer's node, or owner for the
  owner-applied puts above). Not on disk.
- **Durable:** after `SyncTag`/fsync returns 0 (with `fsync_mode: durable`).
  Without fsync: after a flush **and** the next WAL fsync (≤ 5000 ms later).
- **Consistency:** one node: read-your-writes, always. Several nodes: a reader
  sees another node's write after that write is flushed (≤ `flush_period_ms`,
  sooner with the other triggers). After the writer's `SyncTag` returns, every
  node sees it.

| failure | lost or unavailable | shrink it with |
|---|---|---|
| client process crash | nothing acked. Unacked in-flight puts may or may not land. | n/a |
| runtime crash (`kill -9`) on a node | every dirty range buffered on that node: writes since the last flush (≤ 10 s by default, or ≤ 256 MiB, whichever came first), including writes to blobs other nodes own | fsync; lower `flush_period_ms`/`flush_dirty_mb`; write-thru |
| power loss | the above, plus flushed bytes whose device writes and WAL records were not yet fsynced (≤ `flush_metadata_period_ms`) | fsync |
| disk loss | everything on that device (the cache has no copy of clean data) | `replication_factor` ≥ 2, safe_bdev, `num_replicas` |
| temporary node loss | blobs it owns are unavailable until it returns; other nodes' flushes to it fail and stay dirty for retry. Its own dirty buffer is lost if the runtime died. | `replication_factor` ≥ 2 + core `failover_to_successor: true` |
| permanent node loss | its dirty buffer and every blob it owned | `replication_factor` ≥ 2 |
| `clio-run start --fresh` | everything (WAL, snapshot, alloc logs). The volatile core is empty after any start anyway. | do not use `--fresh` on data you need |

## Pitfalls and failure signatures

| symptom | cause | fix |
|---|---|---|
| Pool create fails, log `next_volatile is required` | key missing or malformed (`"540"` without `.0`) | give `"major.minor"` |
| Data gone after restart although the app "saved" it | process killed, no fsync; or volatile-only (no `next_persistent`) | fsync / `SyncTag`; graceful `clio-run stop`; set `next_persistent` |
| `dirty copies not flushed within the N s grace period` at stop | large dirty set, slow disk | `clio-run stop --grace-period 60000`; fsync first; lower `flush_dirty_mb` |
| Writes suddenly run at disk speed | volatile full of dirty copies, writes going around | bigger volatile core, lower `flush_fill_pct`; or `fallback_to_persistent: false` to get ENOSPC instead |
| Puts fail with rc 13 | `fallback_to_persistent: false` and volatile full | as above, or set it true |
| fsync returns ENOSPC (rc 28) | persistent tier full | add capacity |
| Node B does not see node A's write for seconds | write buffered on A until a flush | `SyncTag` on A; lower `flush_period_ms` |
| Writes bypass the cache entirely | client bound to 512.0 (no shm cache composed, no `CLIO_CTE_POOL`) | `CLIO_CTE_POOL=567.0` or compose the shm cache |
| fsync slow with many ranks | expected: one shared flush round per tag | fsync once per file, not per rank |
| Stale sizes in other nodes' shm caches | shm cache has `invalidate_on_flush: true` but `flush_pool` unset | set `flush_pool: "568.0"` |

## Verify it works

1. Runtime log at start: `policy cache 567.0: write-back over volatile 540.0
   (persistent chain)`. `no persistent chain` means `next_persistent` is unset.
2. Speed and volatility: write a blob with the Python snippet in SKILL.md, then
   `kill -9` the runtime within 1 s, `clio-run start`, read it back. It is
   usually gone: it was only in DRAM, and no flush trigger had fired.
3. Durability: write, `clio-run stop` (graceful), `clio-run start`, read back.
   It must be there. On clio-fs, `write` + `fsync` + `kill -9` must also keep it.
