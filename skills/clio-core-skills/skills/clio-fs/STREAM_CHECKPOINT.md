# Stream and checkpoint ChiMods

## clio_cte_stream (565.0)

A byte-stream layer over CTE tags. A stream is a tag whose bytes live in
1 MiB page blobs named by decimal page index (`"0"`, `"1"`, ...). The stream
pool owns each stream's **authoritative logical size** and merges deferred
appends. clio-fs uses one stream per file; other code may use it for logs.

| item | value |
|---|---|
| `mod_name` | `clio_cte_stream` |
| default pool id / name | 565.0 / `clio_cte_stream` (no fixed id in code; owners name the pool) |
| pip wheel | yes |
| build option | `CLIO_CTE_ENABLE_STREAM` (default ON); required by the filesystem |

### Compose keys

```yaml
  - mod_name: clio_cte_stream
    pool_name: clio_cte_stream
    pool_query: local
    pool_id: "565.0"
    next_pool_id: "564.0"
    staging_pool_id: "561.0"
    log_path: "${CLIO_STORAGE_ROOT}/cte_stream_log"
```

| key | type | default | meaning |
|---|---|---|---|
| `next_pool_id` | `"major.minor"` | none; **required** | Pool pages are read from and merged into (the chain top). Missing: pool creation fails (rc 1). |
| `staging_pool_id` | `"major.minor"` | `next_pool_id` | Pool staged appends are written to. Point it below every cache (replication 561.0): staged bytes are written once and read at most once. |
| `log_path` | path | unset | Base of each container's size/plan log; `.<container_id>` is appended. Records reach the OS per write and are fsynced at least every 5000 ms and on `fsync(2)` of a file. Unset: sizes, merge plans and merged-append names are not persisted (clio-fs then recovers sizes from inode records). A fresh start (`--fresh`) rewrites it empty. |

Compose it **before** the filesystem and after the chain it names. If it is
not composed, the filesystem creates its own stream pool at its own id plus
one minor (560.1), with `log_path = <metadata_log_path>.stream`.

### How appends work

1. **Append** (on the writer's node): the bytes are written as a staged blob
   named `sa.<tag major>.<tag minor>.<home>.<clock>.<origin>.<counter>` in the
   tag `_clio_stream_staging` on `staging_pool_id`. The call returns when that
   put returns: from then on a crash cannot lose the append.
2. **Sequence** (every 1 ms per node): each node ships its batch for each
   stream, with the bytes inline, to that stream's **home** container.
3. **Plan** (at the home): the combined batch is sorted by (origin clock,
   origin node, counter), its range reserved from the size, and copied into
   the pages in slices of at most 16 MiB. The plan is logged so a restart
   finishes it.

Ordering: one node's appends keep their submission order (its clock is
forced monotonic even if the wall clock steps back). Appends from different
nodes are ordered by their origin's wall clock; skew between nodes can
reorder near-simultaneous appends, but each append lands whole and once.

**Size authority:** only the home changes a stream's size. The caller chooses
the home and must use the same one for the stream's life (clio-fs uses the
inode's home; others use `Client::DefaultHome(tag, num_containers)`, a hash
of the tag id).

**Flush** waits only for appends this node accepted before the flush, then
returns the size. It fails with `ETIMEDOUT` after 120 s, or with `EIO` after
5 s if the home's node is down (the appends stay queued and merge when it
returns). `close()` and `fsync()` on an `O_APPEND` file call it.

### Client API (C++, source builds only; no Python binding)

`clio::cte::stream::Client` (header `clio_cte/stream/stream_client.h`):

| method | meaning |
|---|---|
| `AsyncCreate(pool_query, pool_name, pool_id, StreamConfig)` | create/bind the pool (use `PoolQuery::Dynamic()`) |
| `AsyncSizeOp(tag, home, op, value=0)` | `StreamSizeOp::kGet`, `kMax` (size = max(size, value)), `kReserve` (old = size; size += value), `kSet` (truncate/extend), `kDrop` (forget; discard pending appends), `kSync` (fsync the home's log). Results in `old_size_`, `new_size_` |
| `Append(tag, home, data, size)` | synchronous deferred append from private memory; returns 0 or an error (EIO if staging failed) |
| `AsyncAppend(tag, home, shm_buffer, size)` | same from a runtime buffer |
| `AsyncFlush(tag, home)` | wait for this node's earlier appends; `size_` after `Wait()` |

```cpp
clio::cte::stream::Client s(clio::run::PoolId(565, 0));
auto tag = /* TagId from the CTE core's GetOrCreateTag */;
clio::run::u32 home = s.DefaultHome(tag, num_containers);
s.Append(tag, home, line.data(), line.size());
auto f = s.AsyncFlush(tag, home); f.Wait();   // f->size_ = merged size
```

Writes to pages that do not go through the stream (`PutBlob` on the tag) do
not change its size; push it with `kMax`.

## clio_cte_checkpoint (566.0)

The fault handler behind lazy copies (`Vector::Copy(new_name)` in the CPU
and GPU vector adapters). A copy tag is created with this pool registered as
its fault handler and the source tag name as the fault parameter. Nothing is
copied up front. When a get on the copy misses a blob, the core sends the get
here; the handler reads the **whole** source blob, writes it into the copy,
and serves the request from the copy. A source blob that is absent becomes an
empty tombstone in the copy, so later source writes never leak in.
`Copy(new_name, true)` (or `MaterializeAll()`) materializes every page before
returning.

| item | value |
|---|---|
| `mod_name` | `clio_cte_checkpoint` |
| pool id / name | 566.0 / `clio_cte_checkpoint` (hard-coded in `Vector::Copy`) |
| pip wheel | **no**. Built by default (`CLIO_CTE_ENABLE_CHECKPOINT`, ON) and shipped in deb/rpm/source installs |
| key | `next_pool_id` (`"major.minor"`, optional): pool sources are read through and copies written to. Unset: the CTE core |

You do not need to compose it: `Vector::Copy` creates it on every node
(`PoolQuery::Dynamic()`, idempotent) with `next_pool_id` = the vector's
storage pool. Clients call the core's API on the copy tag; the handler is
invisible.

**Collision:** `clio_cte_core_persist_write_thru` also defaults to 566.0. If
you compose write-thru, give **write-thru** another id (for example 569.0).
Moving the checkpoint pool does not help: `Vector::Copy` always uses 566.0.

Guarantees: the copy is a snapshot of the source at `Copy` time (the vector
flushes its resident pages first). When the source is written later, the
core's copy-on-write hook first materializes that blob into the copy from
the pre-write bytes; a blob the copy already holds is never replaced.
Materialized blobs have the durability of `next_pool_id`; blobs not yet
materialized depend on the source tag still existing. Failures: a fault with an empty source name returns rc 1;
an allocation failure returns rc 2.
