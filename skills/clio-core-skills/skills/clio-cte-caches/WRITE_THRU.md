# Write-thru cache (`clio_cte_core_persist_write_thru`, default 566.0)

## What it is

The strongly consistent policy cache. Like write-back, it sits over a volatile
core and a persistent chain. A put is written to the persistent chain first and
acked only when that write succeeds. The volatile copy is then refreshed so
later reads come from DRAM. The volatile copy is never dirty, so nothing needs
flushing. The cost: every write pays the persistent chain's latency. It was
measured slower than write-back on every workload of the Ares tier suite.

## Availability

- `mod_name: clio_cte_core_persist_write_thru`. Built-in default `pool_id`
  566.0, which **collides with the checkpoint module (566.0)**. Always set an
  explicit free id such as `"569.0"` when checkpoint might also be composed.
- In the pip wheel. Source builds: `CLIO_CTE_ENABLE_POLICY_CACHES=ON` (default).
- Not in the default compose. It replaces the write-back entry; never compose both.

## When to compose it

- Every acknowledged write must already be on the persistent device (the
  application never calls fsync, or a crash must lose nothing acked).
- Reads are hot and writes are rare, so the DRAM copy pays and the write cost
  does not.
- Not for write-heavy or small-write workloads: use write-back and fsync.

## Compose entry

Replace the write-back entry of the default compose:

```yaml
  - mod_name: clio_cte_core_persist_write_thru
    pool_name: clio_cte_core_persist_write_thru
    pool_query: local
    pool_id: "569.0"              # explicit: 566.0 belongs to checkpoint
    next_volatile: "540.0"        # volatile core (required)
    next_persistent: "561.0"      # replication -> persistent core 512.0
    fill: on_hit_count:2
    evict: lru
```

If the shm cache sits above it, point the shm cache's `next_pool_id` at
`"569.0"` and set its `invalidate_on_flush: false` (write-thru never lands
flushes through the shm cache, so `true` would leave other nodes' metadata
stale).

### Keys

Same parser as write-back (`PolicyCacheParams::LoadConfig`).

| key | type | default | used by write-thru? |
|---|---|---|---|
| `next_volatile` | pool id `"major.minor"` | none; **required** (Create fails with rc 1) | yes |
| `next_persistent` | pool id | none | yes. Omitted: every verb goes to the volatile core, so nothing is persistent. |
| `fill` | `on_hit_count:N` \| `on_read_miss` \| `none` | `on_hit_count:2` | yes: when a read miss copies the blob into volatile |
| `evict` | `lru` \| `lfu` | `lru` | yes |
| `flush_period_ms` | u32 ms | 10000 | parsed, ignored (no dirty data) |
| `flush_fill_pct` | u32 % | 80 | parsed, ignored |
| `flush_dirty_mb` | u32 MiB | 256 | parsed, ignored |
| `fallback_to_persistent` | bool | true | parsed, ignored: a put whose volatile refresh fails still succeeds and the volatile copy is dropped |
| `flush_pool` | pool id | own pool | parsed, ignored |

Persistent-core keys that still matter: `flush_metadata_period_ms` (5000 ms,
WAL fsync period) and `fsync_mode`.

## Where it goes in the chain

- Names `next_volatile` and `next_persistent`; compose it after both cores and
  replication. Above it: optional compressor, shm cache, indexer.
- Blob verbs route to the blob's **owner** container (hash of tag and blob).
  A put from another node is forwarded there.

## How it works

- **Put:** take the blob's lock on the owner, write the persistent chain, and if
  that succeeds update the volatile copy (a partial write to an uncached blob
  fills it first). A failed volatile update drops the copy; the put still
  returns the persistent result.
- **Special puts** (append, replace, conditional) drop the volatile copy and go
  to persistent only.
- **Batches** (`MultiPutBlob`) go to persistent; the records' volatile copies
  are dropped and refill on the next read.
- **Get:** on the owner, volatile copy if present, else persistent, then fill
  per `fill`.
- **`SyncTag`:** nothing to flush; forwarded so the persistent chain syncs its
  devices and WAL (rc 28 ENOSPC, rc 5 EIO).
- **Writes behind the cache** (GPU `PodPutBlob`/`PodMultiPutBlob` sent through
  this pool) drop the tag's volatile copies on every node before and after.

## Guarantees

- **Ack:** persistent chain wrote the bytes (and replication's remote copies if
  `replication_factor` > 1). Device not yet fdatasynced; WAL record written to
  the OS.
- **Consistency:** one owner serializes each blob; every reader on every node
  sees the latest acked write.

| failure | lost or unavailable | shrink it with |
|---|---|---|
| client process crash | nothing acked | n/a |
| runtime crash (`kill -9`) | nothing acked (device writes and WAL records are in the OS) | n/a |
| power loss | acked writes whose device data or WAL record was not yet fsynced (≤ `flush_metadata_period_ms`, 5000 ms) | `SyncTag`/fsync |
| disk loss | everything on that device | `replication_factor` ≥ 2, safe_bdev |
| temporary node loss | blobs it owns are unavailable; puts from other nodes to them retry for up to ~10 s, then fail | replication + `failover_to_successor` |
| permanent node loss | every blob it owned | `replication_factor` ≥ 2 |
| `clio-run start --fresh` | everything | do not use `--fresh` on data you need |

## Pitfalls and failure signatures

| symptom | cause | fix |
|---|---|---|
| Pool create fails or the wrong module answers at 566.0 | checkpoint also uses 566.0 | explicit `pool_id: "569.0"` and update every `next_pool_id` that pointed at it |
| Write throughput at disk speed | expected: ack waits for persistent | write-back + fsync |
| `flush_period_ms` change has no effect | ignored by write-thru | n/a |
| Data lost after a crash although write-thru is composed | client bound to the volatile core or another pool beside the chain | bind to the chain top (`CLIO_CTE_POOL`, or the shm cache's `entry_pool_id`) |
| Other nodes see stale sizes through the shm cache | shm cache `invalidate_on_flush: true` over write-thru | set it `false` |

## Verify it works

1. Runtime log: `policy cache 569.0: write-through over volatile 540.0
   (persistent chain)`.
2. Write a blob, `kill -9` the runtime immediately, `clio-run start`, read it
   back. It must be there (unlike write-back).
