---
name: storage-compose
description: 'Use when designing, configuring or reviewing a CLIO Core storage deployment: choosing which shipped modules to compose (bdev, safe_bdev, CTE cores, write-back/write-thru caches, replication, shm cache, indexer, compressor, stream, filesystem, CAE), ordering the chain, and stating its consistency and fault tolerance. Triggers on "compose section", "clio_default.yaml", "next_pool_id", "replication_factor", "is this durable", "what survives a crash". Not for writing a new module; use chimod-dev.'
metadata:
  bundle: clio-core
  servers: none
  provenance: designed
  eval-status: scenarios-recorded
---

# Composing a storage system from CLIO Core modules

A CLIO storage system is a list of **pools** in the `compose:` section of the
runtime config. Each pool is an instance of a module (a "chimod"). The data
path is a **chain**: the modules above the CTE cores all use the core's own
Put/Get task interface and forward to the pool named by `next_pool_id`. A client
addresses the top of the chain and does not need to know what is below it.

Facts here were checked against clio-core `dev` at bd1dc3183 (2026-11-21).
The runtime reads the file named by `CLIO_SERVER_CONF`, else
`~/.clio/clio.yaml`. A pip install copies the shipped, commented default
(`<site-packages>/iowarp_core/data/clio_default.yaml`) to `~/.clio/clio.yaml`
the first time Python imports `iowarp_core`. Start from that file.

## Files in this skill

| file | read when |
|---|---|
| **MODULES.md** | choosing modules. Purpose, default pool id, key settings, and caveats for each module. |
| **GUARANTEES.md** | before telling anyone what is durable, consistent or available. What an ack means, what fsync means, and what is lost to each failure. |
| **RECIPES.md** | writing the YAML. Six complete deployments, from scratch-only to replicated filesystem. |

## Module skills

This skill designs the chain. Each module's full key table, client API and
failure behavior is in its own skill; load it when you write that module's
entry.

| skill | covers |
|---|---|
| `clio-runtime-config` | the config file outside `compose:`, environment variables, the `clio-run` CLI, multi-node and Docker, the admin module |
| `clio-bdev` | `clio_bdev` (301.0 and every tier device) and the erasure-coded `clio_safe_bdev` |
| `clio-cte-core` | the CTE core (512.0 persistent, 540.0 volatile), and the C++ and Python blob API every chain speaks |
| `clio-cte-caches` | write-back (567.0), write-thru, the shm cache (568.0) and the plain cache (563.0) |
| `clio-replication` | `clio_cte_replication` (561.0) and node-loss availability |
| `clio-compressor` | `clio_cte_compressor` (562.0) |
| `clio-indexer` | `clio_cte_indexer` (564.0) and search |
| `clio-fs` | filesystem (560.0), stream (565.0), checkpoint, and mounting with FUSE |
| `clio-cae` | ingest: CAE core (400.0), the summarizer (401.0) and OMNI files |
| `chimod-dev` | writing a module the shipped ones do not cover |

## The default chain (dev)

```
clients / CAE 400 / filesystem 560 / stream 565
        |
   indexer 564            search index (BM25), read-your-writes for search
        |
   shm cache 568          node shared-memory metadata cache (md_only)
        |
  [compressor 562]        optional, needs CLIO_CTE_ENABLE_COMPRESS
        |
   write-back 567         owns durability: acks on the volatile write, flushes later
     |            \
 volatile core 540  replication 561 -> persistent core 512
 (DRAM, no WAL)                        (file/DAOS tiers, metadata WAL)
        \___________ bdev pools, created by each core per storage[] entry
```

## Workflow

1. **State the requirements as answers, not adjectives.** For each one, give
   the failure it must survive and the acceptable loss window:
   - Must data acked by a put survive a runtime crash? A power loss? The loss
     of a node?
   - Must it stay *available* while a node is down, or only come back later?
   - Must readers on other nodes see a write as soon as it is acked?
   - Must every acked byte be durable at once, or only after fsync?
   - Capacity per tier, and whether the workload is scratch, a checkpoint, or
     a dataset.
2. **Pick the core layout.**
   - Scratch only: one volatile core.
   - Durable data: a persistent core plus a volatile core, joined by a policy
     cache. Use write-back by default. Use write-thru when every ack must be
     durable.
   - Never mix volatile and persistent devices in one core (GUARANTEES.md).
3. **Add a layer only for a requirement from step 1.**
   - Node loss: `replication`.
   - Disk loss without a second node: `safe_bdev` under the persistent core.
   - Search: `indexer`.
   - Many local client processes: the shm cache.
   - POSIX: `filesystem` plus `stream`.
   - Ingest: CAE.
   - Smaller footprint: `compressor`.
4. **Order the compose list bottom-up.** Every entry must come *after* each pool
   it names (`next_pool_id`, `next_volatile`, `next_persistent`, `flush_pool`,
   `staging_pool_id`, `existing_pool_id`). Give every pool a unique `pool_id`
   (MODULES.md lists the collisions in the defaults).
5. **Write down the guarantees** of the result, in the form GUARANTEES.md uses:
   what an ack means, when data becomes durable, and what is lost on a runtime
   crash, a power loss, or the loss of a node.
6. **Validate by killing things.** A deployment is not durable until a
   restart test shows it. See "Validation" below.

## Non-negotiable rules

- `clio_bdev` at 301.0 (`ram::chi_default_bdev`) is required: other modules
  allocate from it.
- **Metadata durability is opt-in.** A persistent core without
  `performance.metadata_log_path` keeps the bytes on disk but not the record of
  which blob owns them. After a restart, the data is unreachable.
- Set `persistence_level` on every durable tier. Omitting it makes the tier
  **volatile**, even for a `file` tier. Its blocks are then dropped at replay,
  and a persistent core refuses the device.
- Do not put CAE (or anything with its own method ids) at 512.0, and do not
  point a client at a pool whose method ids are not the core's. CAE's
  `ParseOmni` is id 10, which is also the core's `RegisterTarget`.
- Interposer verbs are forwarded **in process on the same node**. Every chain
  pool must exist on every node the core does: compose them all
  `pool_query: local`.
- Keep `num_replicas: 0` over a `safe_bdev` tier. The array already survives
  disk loss, so a second copy on the same array only doubles each fsynced byte.
- **Point clients at the top of the chain.** A client attached by shared
  memory binds to the shm cache's entry pool (564.0 by default). A TCP or IPC
  client binds to 512.0 and **skips every layer above it** (caches,
  replication, indexer). Set `CLIO_CTE_POOL=564.0` (or your chain top) before
  the client initializes.
- `MultiPutBlob` routes by its *first* put. A multi-node caller must ensure
  every put in the batch maps to the same owner container.
- SWIM failure detection is **off** by default, and that is the
  recommendation for batch jobs. Turn it on only for long-lived services, with
  timeouts sized for the widest collective.
- Scope the indexer (`tag_re`, `blob_re`) and the summarizer (`label_matches`)
  tightly. The indexer's in-memory index grows with every matching blob. The
  summarizer runs inference synchronously on the worker that handles the put.

## Validation

- **Resident check.** Write, read back, compare checksums. Do this from a second
  node too if the deployment is multi-node.
- **Durability check.** Write, fsync (`SyncTag`, or `fsync` on clio-fs), then
  kill the runtime with `kill -9`, not `clio-run stop`. Restart *without*
  `--fresh` and read back. Then repeat the test without the fsync, and record
  what was lost: it should match the loss window you stated.
- **Node-loss check.** This applies only with `replication_factor > 1`. Stop
  one node and read its blobs with `failover_to_successor: true`. Write
  through the successor, restart the owner, and confirm the hand-back.
- **Disk-loss check** (safe_bdev). Create `<pool_name>.fail` for one member.
  Reads must still return correct bytes. Then fail more than `max_failures`
  members: requests must fail with EIO, never return wrong bytes.
- Use `CLIO_REQUIRE_RUNTIME=1` in any measurement, so a run that never reached
  the runtime fails instead of reporting a clean zero.
- Watch for these signatures:
  - `PutBlob` rc=11 with `available_targets` empty means no target
    registered. Check the tier config, or the build tree is stale.
  - `RegisterTarget` rc=1 with `Permission denied` on a `/tmp` path means
    root-owned leftovers from an earlier run.
