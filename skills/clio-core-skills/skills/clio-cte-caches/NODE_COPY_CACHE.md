# Node-copy cache (`clio_cte_cache`, 563.0)

## What it is

A write-through cache for **single-core** chains on several nodes. Each blob
has one owner node. This module keeps an untransformed (uncompressed) copy of
blobs owned by *other* nodes on the node that reads or writes them, in the CTE
core's `REPLICA_CACHE` slot, so repeat reads stay local. Every write goes down
the chain and is acked only after the authoritative write succeeds. The owner
invalidates every other node's copy before it acks a write. It tracks no dirty
data and has no flush. It costs extra space on the fast tier and, on a miss, a
second read of the blob to populate the copy.

## Availability

- `mod_name: clio_cte_cache`, default `pool_id` 563.0.
- In the pip wheel. Source builds: `CLIO_CTE_ENABLE_CACHE=ON` (default).
- **Not in the default compose.** The write-back cache (567.0) and the shm
  cache (568.0) replaced it there.

## When, if ever, to use it

Use it only when all of these hold:
- the deployment has **one** CTE core at 512.0 with a memory tier (no
  volatile/persistent split, no write-back cache);
- several nodes, and reads of blobs owned by other nodes dominate;
- often with the compressor below, so hot reads stay uncompressed.

Do not use it:
- in the default two-core layout. It always stores its copies in the core at
  **512.0** (fixed, not configurable), which there is the disk core, so the
  copies land on disk.
- together with write-back or write-thru. Both keep their own copies.
- on a single node without compression. Every blob is owner-local, so it adds
  nothing except a raw mirror of batched puts.

## Compose entry

```yaml
compose:
  - mod_name: clio_cte_core        # single core with a DRAM tier and a disk tier
    pool_name: cte_main
    pool_query: local
    pool_id: "512.0"
    storage:
      - { path: "ram::cte_dram", bdev_type: "ram", capacity_limit: "8GB", score: 1.0 }
      - { path: "/mnt/nvme/cte.dat", bdev_type: "file", capacity_limit: "200GB",
          score: 0.2, persistence_level: "temporary" }
    performance:
      metadata_log_path: "/mnt/nvme/cte_metadata_log"
  - mod_name: clio_cte_replication
    pool_name: clio_cte_replication
    pool_query: local
    pool_id: "561.0"
    next_pool_id: "512.0"
  - mod_name: clio_cte_cache
    pool_name: clio_cte_cache
    pool_query: local
    pool_id: "563.0"
    next_pool_id: "561.0"
    min_score: 0.5
    md_only: false
```

Clients: set `CLIO_CTE_POOL=563.0` (no shm cache in this chain, so the client
would otherwise bind to 512.0 and skip the cache).

### Keys (`CacheConfig::LoadConfig`, best-effort: malformed values are ignored)

| key | type | default | meaning |
|---|---|---|---|
| `next_pool_id` | pool id `"major.minor"` | the core pool (512.0) | where authoritative writes and reads go (replication, compressor or indexer). |
| `min_score` | float 0.0-1.0 | 0.5 | score floor for cache copies. The organizer never demotes a copy below it; only capacity pressure on the tier frees one. |
| `md_only` | bool | false | keep no node-local data copies: puts, gets and size probes go straight to the chain. Tag write-through and routing are unchanged. Use for write-heavy runs where copies crowd the fast tier. |

No size key: copies use the core's tiers, bounded by `capacity_limit`. Bytes in
cache copies are reported as free capacity.

## Where it goes in the chain

- Top of a single-core chain: `cache 563.0 -> [compressor 562.0 ->] replication
  561.0 -> core 512.0`. Compose it after `next_pool_id`.
- It intercepts PutBlob, MultiPutBlob, GetBlob, GetBlobSize, the GPU pager's
  `PodMultiGetBlob`, GetOrCreateTag and DelTag; everything else passes through.
- Puts, gets and size probes run on the **submitting** node; only the
  authoritative hop crosses the network. Ops with an explicit replica bypass it.

## Guarantees

- **Ack:** the chain below acked the authoritative write. A failure to update
  the local copy degrades reads, never the put; a failed authoritative put
  deletes the local copy.
- **Consistency:** a read served from a copy reflects every write acked before
  it, on any node (owner invalidation; read-path registration is version
  checked). Tag create/delete is applied to every core container before it
  returns.
- **Loss:** the same as the chain below. Copies are never the only copy.

| failure | effect |
|---|---|
| client crash | nothing acked lost |
| runtime crash / restart | copies on that node are dropped (it may have missed invalidations); refilled on demand |
| power loss, disk loss, node loss, `--fresh` | as the chain below |
| owner node down | a node standing in for it drops its old copy and serves the replicated primary |

## Pitfalls

| symptom | cause | fix |
|---|---|---|
| No speedup | single node, or reads are mostly owner-local | expected; remove it |
| Copies appear on the disk tier | two-core layout: copies go to core 512.0 | use write-back + shm cache instead |
| Batched puts are not cached on a cluster | by design: batch copies have no owner registration | reads populate copies |
| Partial writes do not create a copy | by design; only a whole-blob write at offset 0 or a read does | n/a |
| Cache bypassed | client bound to 512.0 | `CLIO_CTE_POOL=563.0` |

## Verify it works

1. Start log: `cache: md-only mode OFF (no node-local blob data copies)`.
2. Two nodes: write a blob owned by node B from node A, read it twice on A
   (the second read is local), overwrite it from B, read on A: the new bytes
   must come back.
