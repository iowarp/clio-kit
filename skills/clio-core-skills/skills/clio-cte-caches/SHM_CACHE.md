# Shared-memory cache (`clio_cte_core_shm_cache_writethru`, 568.0)

## What it is

A per-node front end for the CTE chain. Each node's container owns one
shared-memory segment that every local client process maps. The segment caches
tag and blob **metadata** (tag name to id, blob sizes, tag sizes and times) and,
unless `md_only`, blob **bytes**. A hit completes inside the client with no
task. In data mode a write is copied into the segment, its exact byte range is
appended to a log, and a background flusher drains the log down the chain as
`MultiPutBlob` batches. It adds no durability of its own (no WAL). It costs one
segment of RAM per node (`capacity`, 256 MiB) and cross-node invalidation
messages.

## Availability

- `mod_name: clio_cte_core_shm_cache_writethru`, default `pool_id` 568.0.
- In the pip wheel. Source builds: `CLIO_CTE_ENABLE_SHM_WTC=ON` (default).
- In the default compose, in **`md_only: true`** mode over the write-back cache.

## When to compose it

- Many processes per node issuing small CTE operations (HDF5 VOL, Python
  loops, metadata-heavy tools). It removes a task round trip per hit.
- Keep `md_only: true` when a write-back cache sits below: the write-back cache
  already holds the bytes in DRAM, so data mode would buffer them twice.
- Use data mode (`md_only: false`) over write-thru, or when write latency from
  local processes matters more than the extra copy.
- Not useful for clients that cannot map the segment (other nodes, clients on
  a transport without shared memory). They use the task path through it.
- Note: clio-fs (FUSE) and the HDF5 VFD do not use this cache's segment; the
  HDF5 VOL and the CTE Python/C++ clients do.

## Compose entry (default compose)

```yaml
  - mod_name: clio_cte_core_shm_cache_writethru
    pool_name: clio_cte_core_shm_cache_writethru
    pool_query: local
    pool_id: "568.0"
    next_pool_id: "567.0"        # write-back (562.0 with the compressor)
    entry_pool_id: "564.0"       # clients bind to the indexer above
    md_only: true                # bytes are cached by the write-back cache
    invalidate_on_flush: true    # requires write-back flush_pool: "568.0"
    capacity: "256M"
    log_depth: 0
    flush_batch_bytes: "16M"
    flush_period_us: 50
    flush_retries: 3
    flush_shards: 8
    flush_depth: 4
    max_tags: 4096
    max_blobs: 65536
    max_fill: "16M"
    evict_sample: 16
    half_life_ms: 10000
    max_sharer_entries: 65536
    nospace_wait_ms: 250
```

### Keys (`WtcConfig::LoadConfig`)

Parsing is best-effort: a malformed value is silently ignored and the default
stays. Sizes accept suffixes (`256M`, `1G`).

| key | type | default | meaning and constraints |
|---|---|---|---|
| `next_pool_id` | pool id `"major.minor"` | none | pool below (policy cache or compressor). Unset: the core pool. |
| `entry_pool_id` | pool id | this pool | pool clients bind to so their tasks pass every module above this cache (the indexer, 564.0). |
| `md_only` | bool | **false** (default compose sets true) | cache metadata only; puts and gets of bytes go down the chain as tasks. |
| `invalidate_on_flush` | bool | false | a write sends no invalidation of its own; the write-back cache's flush lands each range through this pool on the blob's home, which invalidates then. **Only** with a write-back cache below whose `flush_pool` is this pool. Deletes still invalidate at once. |
| `capacity` | size | 256 MiB | segment bytes: buffers, tables and log. |
| `log_depth` | u32 entries | 0 = auto | dirty-range log entries (24 B each). Auto: `max(65536, capacity / 1 KiB)`, capped at 2^28; 65536 in `md_only`. |
| `flush_batch_bytes` | size | 16 MiB | max payload of one `MultiPutBlob` batch. |
| `flush_period_us` | double, **µs** | 50 | flusher cadence (it drains fully each run). |
| `flush_retries` | u32 | 3 | retries per record before its error sticks to the tag. |
| `flush_shards` | u32 | 8 | parallel `MultiPutBlob` tasks for this node's own blobs. |
| `flush_depth` | u32 | 4 | flush batches in flight. |
| `max_tags` | u32 | 4096 | tags tracked. |
| `max_blobs` | u32 | 65536 | blob entries (metadata + buffers). |
| `max_fill` | size | 16 MiB | larger blobs are read but not cached. |
| `evict_sample` | u32 | 16 | clean entries sampled per eviction. |
| `half_life_ms` | u64 ms | 10000 | frecency decay half-life (eviction ranks by hits × 2^(-age/half_life)). |
| `max_sharer_entries` | u32 | 65536 | blobs homed on this node that other nodes cache. |
| `nospace_wait_ms` | u32 ms | 250 | a put refused for room waits up to this long for earlier ranges to flush (they become clean and evictable), then writes around the cache. |
| `data_pool` | pool id | unset | extent mode (only with `md_only: false` and `data_capacity` > 0): keep bytes in a node-local RAM bdev. Minor 0 means this node's own device of that core (`"540.0"` = first device of core 540 here). |
| `data_capacity` | size | 0 | bytes leased from `data_pool`; blobs stored as up to 8 runs of 4 KiB granules. |

Fixed limits (not configurable): blob names longer than 200 bytes and tag
names longer than 255 bytes take the task path.

## Where it goes in the chain

```
indexer 564.0 --> shm cache 568.0 --> [compressor 562.0 -->] write-back 567.0 --> cores
```

- Below the indexer, above data transformations and the policy cache. Compose
  it after the pool in `next_pool_id`; the indexer (which names 568.0) after it.
- The CTE core's Create returns this cache's segment keys for the core at the
  bottom of its chain. `CLIO_CTE_CLIENT_INIT` then binds the client to
  `entry_pool_id` and maps the segment. One shm cache per chain.

## What it absorbs

- **Client side (segment, no task):** synchronous `PutBlob`, `GetBlob`,
  `GetBlobSize`, `GetOrCreateTag` hits. `Async*` calls are always tasks.
- **Absorbable put:** auto score, a plain context (no codec, replica or
  persistence settings), flags at most `droppable`, name ≤ 200 bytes. Anything
  else (append, replace, explicit score) goes down as a task, ordered after the
  blob's logged ranges, and drops the local copy.
- **Reads** hit only when the entry holds the whole blob (`complete`) and the
  range is inside its size. A miss reads the chain and fills (≤ `max_fill`).

## Coherence and staleness

- Each blob and tag has a **home** container: `hash % containers` of this pool.
  The home tracks which nodes cache an entry (sharers) and invalidates them
  when it changes. A node whose cached copy is invalidated re-registers on its
  next miss.
- **Data mode:** the flusher sends ranges grouped per home. The home writes
  them down the chain, invalidates every other sharer, and only then acks.
  So after a writer's `SyncTag`, no other node holds a stale copy.
- **`md_only` + `invalidate_on_flush: false`:** each write invalidates other
  nodes' metadata as it lands.
- **`md_only` + `invalidate_on_flush: true` (default):** other nodes' cached
  metadata (a blob size, a tag size) can be stale until the write-back cache
  flushes that write (≤ `flush_period_ms`, 10 s by default) or the writer calls
  `SyncTag`.
- **Same node:** always coherent; a write drops the local entry.
- **Bypass:** a process that writes straight to the core (or any pool below
  568.0) is invisible to the cache; cached copies stay stale until evicted.

## Guarantees

- **Ack, data mode:** bytes are in the node segment and logged. They reach the
  chain within about one flusher pass (50 µs cadence). Durability is then the
  policy cache's. Errors of buffered writes surface at the next `SyncTag` of
  that tag (rc 28 for no space, else the failing rc), like fsync reporting EIO.
- **Ack, `md_only`:** exactly the pool below's ack.
- **`SyncTag`:** waits until every range logged before it has landed (and its
  invalidations finished), syncs the tag on the remote homes that received this
  node's ranges, then forwards down the chain.

| failure | data mode | `md_only` |
|---|---|---|
| client process crash | nothing acked (the bytes are in the runtime's segment) | nothing |
| runtime crash (`kill -9`) | logged ranges not yet flushed (usually under 1 ms of writes; at most what fits in `capacity`), plus the policy cache's loss | the policy cache's loss only |
| power loss | as above, plus the chain's | the chain's |
| disk loss | the chain's | the chain's |
| temporary node loss | flushes of ranges homed there fail, are retried `flush_retries` times, then the error sticks to the tag | the chain's |
| permanent node loss | its unflushed log | nothing extra |
| `clio-run start --fresh` / any restart | the segment is rebuilt empty; cached entries are not persistent | same |

## Pitfalls and failure signatures

| symptom | cause | fix |
|---|---|---|
| `client.HasCache()` is False | client on another node, a transport that cannot map the segment, or no shm cache in the compose | expected on remote nodes; otherwise check the compose and the start log |
| Log `core_shm_cache_writethru 568.0: no segment; serving tasks only` | segment could not be created (shared-memory limit, e.g. a small `/dev/shm` in Docker) | raise `shm_size`/`/dev/shm`, or lower `capacity` |
| Other nodes report an old file size for seconds | `invalidate_on_flush: true` (by design until the flush) | `SyncTag` on the writer, lower write-back `flush_period_ms`, or `invalidate_on_flush: false` |
| Other nodes see stale metadata indefinitely | `invalidate_on_flush: true` but write-back `flush_pool` unset, or write-thru below | set `flush_pool: "568.0"` on write-back, or `invalidate_on_flush: false` |
| Puts stall about 250 ms then slow down | segment full of dirty buffers; puts wait `nospace_wait_ms`, then write around | larger `capacity`, or `md_only: true` |
| `SyncTag` returns an error for writes that "succeeded" | a buffered range failed to flush after `flush_retries` | check the chain below (space, dead node) |
| Wrong homes after failover or a placement change | homes assume plain hash-mod-N placement (known gap) | avoid with `failover_to_successor`; known issue |
| Writes from a tool bound to 512.0 not seen by cached readers | bypass | bind tools to the chain top (`CLIO_CTE_POOL=564.0`) |
| DESIGN-style keys `max_name`, `inval_timeout_ms` have no effect | not parsed on dev | none (fixed limits) |

## Verify it works

1. Start log: `core_shm_cache_writethru 568.0: segment 256 MB, log 65536
   ranges, next 567.0`.
2. Python snippet in SKILL.md: `HasCache()` is True. Then
   `client.GetBlobSize(tag.GetTagId(), "b0")` (the synchronous call) is served
   from the segment after the first miss.
3. At `clio-run stop` the runtime logs `[shm-cache stats] node=... puts=...
   get_hits=... get_misses=... invalidations=... evictions=...`. Hits rising
   with misses flat means the cache works.
4. Two nodes: write on A, `SyncTag` (or fsync) on A, read the size on B: it
   must be the new size.
