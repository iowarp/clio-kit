---
name: clio-replication
description: 'Use when configuring, reasoning about or testing clio_cte_replication (pool 561.0): replication_factor and remote copies on successor nodes, failover_to_successor, hand-back after a node returns, handoff_log_path, on-node replicas, and node or disk loss. Triggers on "replication_factor", "remote_copies", "failover_to_successor", "hand-back", "survive a node", "561.0". Not for choosing or ordering the whole chain; use storage-compose. Not for core tiers or WAL; use clio-cte-core.'
metadata:
  bundle: clio-core
  servers: none
  provenance: designed
  eval-status: scenarios-recorded
---

# clio-replication: durable copies and node failover

## What it is

The replication module is the reliability layer of the CTE chain. It sits
directly above a CTE core and speaks the core's own task interface. Every put
it sees is written to the blob's owner (the primary) and then copied: to the
next `remote_copies` nodes by hash, and optionally to `num_replicas` extra
persistent copies on the owner's own node. With the core's
`failover_to_successor`, a dead owner's blobs are served by its first live
successor, and the changes made meanwhile are handed back when it returns. The
cost is disk space (about one full copy per extra copy) and one network round
trip per remote copy on every put.

Facts here were checked against clio-core `dev` (bd1dc3183), 2026-11-21.

## Files in this skill

| file | read when |
|---|---|
| **FAILOVER.md** | explaining what happens during and after a node outage, or running the multi-node test |
| **evals.md** | checking the skill still closes its gaps |

## Availability

| item | value |
|---|---|
| `mod_name` | `clio_cte_replication` |
| default `pool_id` | `561.0` (pool name `clio_cte_replication`) |
| pip wheel `iowarp-core` | shipped, and composed by the default config |
| conda / Spack / source | built by default (CMake `CLIO_CTE_ENABLE_REPLICATION`, default ON) |

## When to compose it, and when not to

- Compose it with `replication_factor >= 2` **and** the core's
  `failover_to_successor: true` when data must stay readable and writable
  while a node is down, or must survive losing a node for good.
- Keep it in the chain at `replication_factor: 1` (the default config does).
  That costs nothing extra, and a stand-in still records and hands back writes
  it takes for a dead owner.
- Use `num_replicas` only to survive a dead device on a node with no
  redundant array. **Leave `num_replicas: 0` over a `clio_safe_bdev` tier**:
  the array already survives disk loss, so a second copy on it protects
  nothing and doubles every fsynced byte.
- Skip it for a single-node deployment that only needs crash durability. The
  core's WAL and fsync give that.

## Compose entry

```yaml
compose:
  # ... persistent CTE core first (512.0). It MUST carry:
  #   targets:
  #     failover_to_successor: true
  - mod_name: clio_cte_replication
    pool_name: clio_cte_replication
    pool_query: local                     # required: one container per node
    pool_id: "561.0"
    next_pool_id: "512.0"                 # the persistent CTE core
    replication_factor: 2                 # primary + 1 copy on the next node
    cache_score: 1.0
    replica_score: 0.2
    handoff_log_path: "/var/lib/clio/cte_handoff_log"   # one file per node
    # Advanced overrides (rarely needed):
    # remote_copies: 1                    # default: replication_factor - 1
    # num_replicas: 0                     # extra copies on the owner's node
    # replicate_period_ms: 50             # 0 = write num_replicas before ack
```

Every key `LoadConfig` reads:

| key | type | default | meaning and constraints |
|---|---|---|---|
| `next_pool_id` | string `"major.minor"` | none (falls back to the core at 512.0) | Pool every put/get is forwarded to. Must be a CTE core or a pool that speaks its interface. A value without a dot is ignored. |
| `replication_factor` | int | `1` | Total durable copies of each blob. Values below 1 become 1. Sets `remote_copies = N - 1` and `num_replicas = 0` unless those keys are also given. |
| `remote_copies` | int | `replication_factor - 1` | Copies on the next containers by hash (other nodes). Capped at node count minus 1. Written before the put returns. |
| `num_replicas` | int | `0` | Extra persistent copies on the owner's own node (pinned, never placed on a volatile tier). Protects against a dead device, not a dead node. Each one adds a full copy to the disk footprint. |
| `replicate_period_ms` | int (ms) | `50` | Period of the sweep that writes `num_replicas` copies after the ack. `0` = synchronous (written before the ack). Does not affect remote copies. The sweep only runs when `num_replicas > 0`. |
| `cache_score` | float 0-1 | `1.0` | Score given to a primary re-created from a replica, so it lands back on the fast tier. |
| `replica_score` | float 0-1 | `0.2` | Score stamped on on-node replicas and remote copies. Also the organizer's drop threshold: a primary scoring below the best persistent replica is dropped, not migrated. |
| `handoff_log_path` | string path | `""` (memory only) | Base path of the hand-back log. Each container appends `.<container_id>`. Without it, a stand-in that restarts before the owner returns forgets what to hand back. The directory must exist and be writable. |

Parsing is best-effort: a malformed value silently leaves that key and every
later key at its default. Check the startup log for the line
`replication: async write-through sweep every N ms` when you set
`num_replicas`.

The pairing key lives on the **core**, not here:

| key (on the CTE core named by `next_pool_id`) | type | default | meaning |
|---|---|---|---|
| `targets.failover_to_successor` | bool | `false` | While a blob's owner node is dead, route the blob to the first live successor container. Off: operations on the dead owner's blobs fail. |

Environment variables (runtime process) that decide when a node counts as dead:

| variable | default | effect |
|---|---|---|
| `CLIO_PROBE_SILENCE_S` | `30` s | A peer silent this long is declared dead. Failover starts only after this. |
| `CLIO_NET_RETRY_TIMEOUT_S` | `30` s | How long a send to an unreachable peer is retried before its task fails. |
| `CLIO_NET_DEAD_FAIL_FAST` | `0` | `1` = a task addressed to a peer already declared dead fails at once instead of waiting out the retry window. |
| `CLIO_CTE_POOL` | unset | Client side. `561.0` binds a plain CTE client to this pool. |

## Where it goes in the chain

- It must sit **directly above the persistent CTE core** and be composed after
  it. The default chain is
  `indexer 564 -> shm cache 568 -> [compressor 562] -> write-back 567 ->
  (volatile core 540 | replication 561 -> persistent core 512)`.
  The write-back cache names it as `next_persistent: "561.0"`; the stream
  module may name it as `staging_pool_id`.
- Put the compressor **above** it, never below, so the copies are the stored
  (compressed) form and replication sees whole blobs.
- `pool_query: local` and the same entry on **every node**. Interposed verbs
  are forwarded in process on the node that runs them, and remote copies are
  sent to the replication container on the successor node. A node without the
  pool cannot hold copies.
- All nodes must share one hostfile. Container id = node id, and the container
  count is fixed by the hostfile, so "the next node by hash" is
  `(owner + i) mod N` in hostfile order. There is no rack or failure-domain
  awareness: put nodes that fail together far apart in the hostfile.

## Guarantees

**What an ack from this pool means.** The owner's primary is written to its
core. Then, before the put returns, each of the next `remote_copies` successors
that is alive receives a copy through its own replication pool (so its local
`num_replicas` apply too). A successor that is down, or whose copy fails, is
**skipped**: the put still succeeds with one copy fewer, a warning
(`remote copy of ... failed`) is logged, and the blob goes on an in-memory
re-copy list. A sweep every 500 ms re-copies it from the primary once the
successor is back. On-node `num_replicas` are written by a sweep after the ack
(every `replicate_period_ms`), or before the ack when that is `0`.

**In the default chain, the ack a client sees is not this ack.** Replication
sits below the write-back cache, so a put acked at the top is in the owner's
volatile core only. It reaches replication on the write-back flush
(`flush_period_ms`, default 10 s), on fsync (`SyncTag`), or at a graceful
`clio-run stop`. Use the write-thru cache, or address 561.0 directly, if every
ack must already be on N nodes.

Durability of each copy is the durability of the core that stores it: device
plus metadata WAL (see `clio-cte-core`). An interposed `SyncTag` first writes
every pending on-node replica of the tag, then runs the core's sync.

| event | with `replication_factor: 2` and failover on |
|---|---|
| client crash | nothing acked is lost |
| runtime `kill -9` on one node | its blobs are unavailable for up to `CLIO_PROBE_SILENCE_S`, then served by the successor. On restart the node pulls back what changed and heals primaries from the remote copy |
| power loss on one node | as above. Writes the successor took are safe; the node's own un-synced device/WAL state is lost, then healed from the copy on first read |
| one disk lost | reads fall back to an on-node replica, else to the remote copy |
| node down temporarily | after it is declared dead, reads and writes go to the successor. **Writes taken during the outage have one copy** (on the stand-in) until handed back |
| node lost permanently | its blobs survive on the successor and stay served there. **Nothing rebuilds the lost copy**: redundancy stays reduced, and a later loss of that successor loses them |
| `clio-run start --fresh` on a node | that node's data and its hand-back log are discarded; changes it held for other owners are never handed back |

With `replication_factor: 1` and failover on, a stand-in can take **new**
writes for a dead owner, but older blobs of that owner are not on it: reads of
them fail or find nothing until the owner returns.

**Partitions.** A node that sees fewer than half the pool alive does not stand
in for anyone. It keeps serving the blobs it owns and fails the rest (EIO).
Two-node clusters still fail over (exactly half counts). A hand-back push older
than a write the owner took after it came back is dropped as stale.

## Client API

Clients call the **CTE core API on pool 561.0**. Nothing else changes: set
`CLIO_CTE_POOL=561.0` before initialization, or (C++) construct
`clio::cte::core::Client(clio::run::PoolId(561, 0))`. Every core verb works;
`PutBlob`, `MultiPutBlob`, `GetBlob`, `GetBlobSize`, `DelBlob`, `TruncateBlob`
and `SyncTag` are interposed, everything else is forwarded untouched.

Module method ids (all above the core's id space):

| id | verb | who calls it |
|---|---|---|
| 100 | `ReplicateBlob(tag_id, blob_name, replica, context)` | user, C++ only: bring replica index `replica` (>= 1) of one blob up to date. Out: `bytes_copied_`. rc 2 alloc failure, 10+x size probe, 20+x read, 30+x write |
| 101 | `FlushTag(tag_id, replica, min_score, context)` | user, C++ only: `ReplicateBlob` every blob of a tag with score >= `min_score`. Out: `blobs_replicated_`, `bytes_copied_` |
| 102 | `ReplicateSweep` | internal periodic (num_replicas) |
| 103 | `HandoffPull(owner)` | internal: a restarted owner asks peers for hand-back |
| 104 | `HandoffSweep` | internal periodic, every 500 ms |

`Monitor` answers the query `replicate_period_ms` (clio-fs reads it at mount).

```cpp
// C++ (source/conda/Spack builds; the wheel has no headers)
#include <clio_cte/replication/replication_client.h>
clio::cte::replication::Client repl(clio::run::PoolId(561, 0),
                                    clio::run::PoolId(512, 0));
clio::cte::core::Context ctx;
ctx.min_persistence_level_ = 1;               // replica blocks on durable tiers
auto f = repl.AsyncFlushTag(tag_id, /*replica=*/1, /*min_score=*/0.0f, ctx);
f.Wait();                                     // f->blobs_replicated_, f->bytes_copied_
```

Python has no replication-specific calls; use `clio_cte_core_ext` against 561.0
(see "Verify it works").

## Pitfalls and failure signatures

| symptom | cause | fix |
|---|---|---|
| Copies exist on the successor, but reads of a dead node's blobs hang ~30 s then fail | `failover_to_successor` not set on the core under replication | Set `targets.failover_to_successor: true` on the core named by `next_pool_id`, on every node |
| Failover on, but the dead node's old blobs read as missing | `replication_factor: 1`: there is no copy to serve | `replication_factor >= 2` |
| Reads stall for 30 s at the start of every outage | node not yet declared dead (`CLIO_PROBE_SILENCE_S`) | Lower `CLIO_PROBE_SILENCE_S` / `CLIO_NET_RETRY_TIMEOUT_S`, set `CLIO_NET_DEAD_FAIL_FAST=1` |
| Some nodes hold no copies; puts log `remote copy ... failed` | pool missing on some nodes, or not `pool_query: local` | Same compose entry on every node, `pool_query: local` |
| Acked data lost when a node died although `replication_factor: 2` | write-back above replication: data was still dirty in the volatile core | fsync before relying on it, lower `flush_period_ms`, or use write-thru |
| After a stand-in restart, the owner returns without the outage's writes | no `handoff_log_path` | Set it to a persistent, writable directory |
| Restarted node's writes wait up to 120 s | it is waiting for peers' hand-back pulls (5 s in Create, then up to 120 s) | Expected; check peers are alive. Log: `never answered the hand-back pull` |
| `cannot open handoff log` at start | directory missing or not writable | Create it before `clio-run start` |
| Disk usage roughly doubles | expected: about `1 + num_replicas + remote_copies` bytes per fsynced byte, plus cache copies | Size tiers for it; keep `num_replicas: 0` over safe_bdev |
| A key seems ignored | YAML parse error stops parsing silently | Quote ids (`"512.0"`), use plain ints |

SWIM gossip is off by default (`swim.enabled: false`). Replication does not
need it: liveness comes from probe silence. If you enable SWIM, its default
timeouts declare a node dead after about an hour.

## Verify it works

1. `clio-run compose list` on each node shows `clio_cte_replication` at 561.0.
2. Single node, write through the pool and read back:

```python
import os
os.environ["CLIO_CTE_POOL"] = "561.0"         # set before init
import clio_cte_core_ext as cte
assert cte.clio_init(cte.RuntimeMode.kClient)
assert cte.initialize_cte("", cte.PoolQuery.Dynamic())
tag = cte.Tag("repl_check")
data = os.urandom(1 << 20)
tag.PutBlob("b0", data)
assert tag.GetBlobSize("b0") == len(data)
assert tag.GetBlob("b0", len(data)) == data
```

3. Multi-node: the outage and hand-back test in FAILOVER.md. A deployment is
   not "replicated" until that test passes.
