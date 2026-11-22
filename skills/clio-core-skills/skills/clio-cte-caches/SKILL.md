---
name: clio-cte-caches
description: 'Use when choosing, composing, tuning or debugging a CLIO Core CTE cache module: write-back (567.0), write-thru, the shared-memory cache (568.0) or the node-copy cache (563.0). Covers keys, chain order, ack and fsync semantics, staleness and loss per failure. Triggers on "write-back cache", "flush_period_ms", "fsync is slow", "lost writes after crash", "md_only", "invalidate_on_flush", "stale size on another node". Not for the core or chain design; use clio-cte-core or storage-compose.'
metadata:
  bundle: clio-core
  servers: none
  provenance: designed
  eval-status: scenarios-recorded
---

# CTE cache modules

The Context Transfer Engine (CTE) ships four cache modules. All four are
**interposers**: they speak the CTE core's own task interface (PutBlob, GetBlob,
SyncTag, ...) and forward to the pools they name. A client never calls a
cache-specific data API. It points an ordinary CTE core client at the cache's
pool id (or at a pool above it), and the cache is transparent.

Facts here were checked against clio-core `dev` (bd1dc3183) on 2026-11-21.

## Which cache for which requirement

| requirement | use | why |
|---|---|---|
| Fast writes, fsync-level durability (the ext4 contract) | **write-back** (567.0) | ack after the DRAM write. Data is durable after `SyncTag`/fsync, a periodic flush, or a graceful stop. This is the default. |
| Every acked write must already be on persistent storage | **write-thru** | ack after the persistent chain wrote it. Nothing is dirty. Every put pays disk latency. |
| Many processes on one node doing small I/O (HDF5 VOL, Python loops) | **shm cache** (568.0) above a policy cache | clients map one node segment. Metadata hits (and data hits, unless `md_only`) need no task. |
| Volatile scratch only, no persistent tier | write-back with `next_persistent` omitted | puts go straight to the volatile core. No flushing. `SyncTag` makes nothing durable. |
| Single CTE core (no volatile/persistent split), multi-node, hot remote reads | **cache** (563.0) | keeps a raw copy of remote-owned blobs on the reading node, owner-invalidated. Not in the default chain. |
| Many nodes reading one blob written elsewhere, with a two-core layout | write-back plus the shm cache | the 563.0 cache stores its copies in core 512.0, which is the disk core in a two-core layout. |

Rules of thumb:
- Compose **exactly one** policy cache (write-back or write-thru) over the two
  cores. Do not stack write-back and write-thru.
- The shm cache is a front end, not a policy. It always sits above a policy
  cache (or the compressor above one).
- Do not compose the 563.0 cache over a write-back cache. Both keep node-local
  copies, and 563.0 assumes a single core at 512.0.

## Availability

| module | `mod_name` | default `pool_id` | in the pip wheel | build option (source/conda/Spack) |
|---|---|---|---|---|
| write-back | `clio_cte_core_persist_write_back` | 567.0 | yes | `CLIO_CTE_ENABLE_POLICY_CACHES` (ON) |
| write-thru | `clio_cte_core_persist_write_thru` | 566.0 (**collides with checkpoint 566.0**) | yes | `CLIO_CTE_ENABLE_POLICY_CACHES` (ON) |
| shm cache | `clio_cte_core_shm_cache_writethru` | 568.0 | yes | `CLIO_CTE_ENABLE_SHM_WTC` (ON) |
| node-copy cache | `clio_cte_cache` | 563.0 | yes | `CLIO_CTE_ENABLE_CACHE` (ON) |

Write-back and write-thru share one implementation (`cache_base`) and one key
set. The default pool id is only a convention: the compose entry's `pool_id`
wins. Give write-thru an explicit free id (for example `"569.0"`) whenever the
checkpoint module could also be composed.

## The default chain (pip install, `~/.clio/clio.yaml`)

```
client --> indexer 564.0 --> shm cache 568.0 (md_only) --> write-back 567.0
                                                            |-- next_volatile:   core 540.0 (DRAM, no WAL)
                                                            '-- next_persistent: replication 561.0 --> core 512.0 (disk + WAL)
```

- A pool must be composed **after** every pool it names. Order in the file:
  512.0, 540.0, 561.0, 567.0, 568.0, 564.0.
- Clients get here automatically. `CLIO_CTE_CLIENT_INIT` (Python:
  `initialize_cte`) asks core 512.0 which shm cache fronts it, binds the
  process to that cache's `entry_pool_id` (564.0) and maps the segment.
- Without an shm cache in the compose, the client stays bound to **512.0** and
  skips every cache. Then set `CLIO_CTE_POOL=<top pool>` (for example `567.0`).

## What an acknowledged put means

| top policy | ack means | durable when |
|---|---|---|
| write-back | bytes are in volatile DRAM on the **writer's** node (on the owner's node for metadata blobs and special puts) | `SyncTag`/fsync returns 0, or a flush has landed **and** the WAL was fsynced (≤ `flush_metadata_period_ms`, 5000 ms) |
| write-thru | the persistent chain (replication + core 512.0) wrote the bytes to its device | the WAL fsync (≤ 5000 ms) or `SyncTag` |
| shm cache, data mode | bytes copied into the node segment and the range logged | after the flusher lands them below (normally within ms), then as the policy below says |
| shm cache, `md_only` | same as the pool below it (the put is a task down the chain) | as below |
| 563.0 cache | the chain below acked the authoritative write | as below |

Full semantics, loss per failure and pitfalls are in each module's file.

## Files in this skill

| file | read when |
|---|---|
| **WRITE_BACK.md** | composing or tuning the default policy cache; flush triggers, fsync, stop flush, every shared policy key, loss per failure. |
| **WRITE_THRU.md** | every ack must be persistent; the 566.0 id collision; which keys it ignores. |
| **SHM_CACHE.md** | the node shared-memory front end: `md_only`, `invalidate_on_flush`, sizes, coherence, what is stale and when. |
| **NODE_COPY_CACHE.md** | the 563.0 `clio_cte_cache` (single-core chains); when, if ever, to use it. |
| **evals.md** | scenarios to check the skill against. |

## Client API (all four modules)

Interposer: clients call the **core's** CTE API on the cache's pool id, or on a
pool above it. No cache adds data verbs a client should call.

Python (`clio_cte_core_ext`, shipped in the wheel):

```python
import iowarp_core                 # sets library paths
import clio_cte_core_ext as cte

assert cte.clio_init(cte.RuntimeMode.kClient, False)    # attach to a running runtime
assert cte.initialize_cte("", cte.PoolQuery.Dynamic())  # binds to the chain top
client = cte.get_cte_client()
print("shm cache mapped:", client.HasCache())

tag = cte.Tag("demo")                 # uses the bound client
tag.PutBlob("b0", b"x" * 4096)        # off=0
assert tag.GetBlob("b0", 4096) == b"x" * 4096

# Address one pool explicitly (e.g. bypass the shm cache):
wb = cte.Client(cte.PoolId(567, 0))
rc = wb.AsyncPutBlob(tag.GetTagId(), "b1", b"y" * 100).wait()   # 0 = success
```

C++ (source, conda or Spack builds; the wheel has no headers):
`clio::cte::core::Client c(clio::run::PoolId(567, 0));` then the usual
`AsyncPutBlob`, `AsyncGetBlob`, `AsyncSyncTag(tag_id)` (Broadcast by default;
rc 28 = ENOSPC, 5 = EIO). Python exposes no `SyncTag`: use `fsync` on a clio-fs
mount, or C++.

## Environment variables

| variable | effect |
|---|---|
| `CLIO_SERVER_CONF` | config file the runtime reads (default `~/.clio/clio.yaml`) |
| `CLIO_CTE_POOL=major.minor` | bind this process's CTE client to that pool instead of asking the core. It still maps the shm cache if that pool fronts it. |
| `CLIO_WTC_TRACE=1` | shm cache: log sync and chain-read-miss traces at info level |

## Verify the chain

1. `clio-run start`, then read the runtime log. Expect one line per cache:
   `policy cache 567.0: write-back over volatile 540.0 (persistent chain)` and
   `core_shm_cache_writethru 568.0: segment 256 MB, log ... ranges, next 567.0`.
   `no segment; serving tasks only` means the shm segment failed to map.
2. `clio-run compose list` shows the live pools (or the dashboard's Pools page).
3. Run the Python snippet above. `HasCache()` is `True` for a same-node client.
   The client log line `CTE ClientInit: bound to 564.0 over shm cache 568.0`
   confirms the binding.
4. Durability: write, `clio-run stop` (graceful; flushes within
   `--grace-period`, default 5000 ms), `clio-run start` (no `--fresh`), read
   back. Each module file has its own check.
