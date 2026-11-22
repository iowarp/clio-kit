# API.md: calling the CTE core

The same API reaches a core and every module above it (indexer, shm cache,
write-back, replication, compressor): they all accept the core's tasks. What
changes is the **pool id** the client is bound to. Checked against clio-core
`dev` at bd1dc3183 (`core/include/clio_cte/core/core_client.h`,
`wrapper/python/core_bindings.cc`).

## Choosing the pool

| how the client is made | pool it talks to |
|---|---|
| `CLIO_CTE_CLIENT_INIT()` (C++) / `initialize_cte()` (Python), `CLIO_CTE_POOL` set | that pool, e.g. `564.0` |
| same, `CLIO_CTE_POOL` unset, client attached by shared memory and a shm cache composed | the cache's `entry_pool_id` (564.0 in the default config), with the shm fast path |
| same, but TCP/IPC client, or no shm cache | **512.0, the persistent core directly** (bypasses caches above it) |
| `Client(PoolId(major, minor))` | exactly that pool; no shm fast path unless you call `ConnectCache()` and the pool is the cache or its entry |

The `Tag` wrapper (C++ and Python) always uses the process-wide client
(`CLIO_CTE_CLIENT` / `get_cte_client()`), so set `CLIO_CTE_POOL` to aim it.
To talk to the top of a chain at 564.0 instead of 512.0:

```cpp
// C++: either before CLIO_CTE_CLIENT_INIT: setenv("CLIO_CTE_POOL", "564.0", 1);
clio::cte::core::Client top(clio::run::PoolId(564, 0));
```

```python
os.environ["CLIO_CTE_POOL"] = "564.0"      # before initialize_cte(); aims Tag too
top = cte.Client(cte.PoolId(564, 0))       # or an explicit client
core = cte.Client(cte.PoolId(512, 0))      # the persistent core itself
```

## Pool queries and routing

`PoolQuery` values: `Local()` (this node), `Dynamic()` (let the pool route),
`Broadcast()` (every container, results merged), plus `DirectHash`/`DirectId`
in C++. The core reroutes by method:

| method | where it runs |
|---|---|
| PutBlob, GetBlob, DelBlob, GetBlobSize, GetBlobScore, GetBlobInfo, ReorganizeBlob, TruncateBlob, GetResidency | the blob's owner, `hash(tag, blob)`, whatever you pass |
| MultiPutBlob, MultiDelBlob | the owner of the **first** record. MultiDelBlob then splits by owner; MultiPutBlob does not, so group puts per owner |
| GetOrCreateTag, DelTag/RenameTag by name, GetOrCreateTagAlias | `hash(tag name)` |
| DelTag/RenameTag/GetTagName by id | the tag's home, `hash(tag id)` |
| TagQuery, BlobQuery, GetContainedBlobs, GetTagSize, Evict, ReorganizeHint, ListLocalBlobs | each container answers for itself: pass **Broadcast** for a cluster-wide answer (TagQuery/BlobQuery/Evict default to it) |
| RegisterTarget, ListTargets, StatTargets, GetTargetInfo | local node |
| SyncTag, MultiDelTag | default Broadcast |
| GetCapacity | default Local; Broadcast sums the cluster |

## C++ client (`clio::cte::core::Client`)

Header `<clio_cte/core/core_client.h>`, CMake target `clio::cte::core_client`
(from `find_package(clio-core)`). Not in the pip wheel. Types: `TagId` =
`clio::run::UniqueId {major_, minor_}`; `u32`/`u64`/`i32` are
`clio::run::` integers. Every `Async*` returns `clio::run::Future<Task>`:
call `.Wait()`, then read `->GetReturnCode()` and the task's output fields.

```cpp
#include <clio_cte/core/core_client.h>
using namespace clio::cte::core;

int main() {
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false)) return 1;
  if (!CLIO_CTE_CLIENT_INIT()) return 1;              // binds CLIO_CTE_CLIENT
  Tag tag("demo/run1");                                // GetOrCreateTag
  std::string v = "hello";
  tag.PutBlob("step0", v.data(), v.size());            // throws on rc != 0
  std::string out(tag.GetBlobSize("step0"), '\0');
  tag.GetBlob("step0", out.data(), out.size());
  auto *c = CLIO_CTE_CLIENT;
  auto sync = c->AsyncSyncTag(tag.GetTagId());         // power-loss point
  sync.Wait();
  return sync->GetReturnCode();                        // 0, 5 (EIO) or 28 (ENOSPC)
}
```

### Synchronous helpers (use the shm cache when connected)

```cpp
u32 PutBlob(const TagId &tag_id, const std::string &blob_name, u64 offset,
            u64 size, const char *src, float score = -1.0f,
            const Context *ctx = nullptr, u32 flags = 0,
            const PoolQuery &pool_query = PoolQuery::Dynamic());
u32 GetBlob(const TagId &tag_id, const std::string &blob_name, u64 offset,
            u64 size, char *dst, u32 flags = 0, const Context *ctx = nullptr,
            const PoolQuery &pool_query = PoolQuery::Dynamic());
u32 GetBlobSize(const TagId &tag_id, const std::string &blob_name, u64 *size,
                const PoolQuery &pool_query = PoolQuery::Dynamic());
u32 GetOrCreateTag(const std::string &tag_name, TagId *tag_id,
                   u64 *tag_size = nullptr,
                   const PoolQuery &pool_query = PoolQuery::Dynamic());
bool ConnectCache();          // fetch shm cache keys and connect
bool HasCache() const;
```

They return 0 or the task's return code (`EINVAL` for a null buffer,
`ENOMEM` when staging failed). `PutBlob` with size 0 returns 0.

### Blobs (`Async*`)

```cpp
// Private buffer (copied once into shared memory in a client process,
// zero-copy inside the runtime). Empty Future for size 0 / null data.
Future<PutBlobTask> AsyncPutBlob(const TagId&, const std::string &blob_name,
    u64 offset, u64 size, const char *priv_data, float score = -1.0f,
    const Context &context = Context(), u32 flags = 0,
    const PoolQuery & = PoolQuery::Dynamic());
// Shared-memory buffer (ShmPtr from CLIO_IPC->AllocateBuffer).
Future<PutBlobTask> AsyncPutBlob(const TagId&, const std::string &blob_name,
    u64 offset, u64 size, ctp::ipc::ShmPtr<> blob_data, float score = -1.0f,
    const Context & = Context(), u32 flags = 0,
    const PoolQuery & = PoolQuery::Dynamic());
Future<GetBlobTask> AsyncGetBlob(const TagId&, const std::string &blob_name,
    u64 offset, u64 size, u32 flags, char *priv_data,
    const PoolQuery & = PoolQuery::Dynamic(), const Context & = Context());
Future<GetBlobTask> AsyncGetBlob(const TagId&, const std::string &blob_name,
    u64 offset, u64 size, u32 flags, ctp::ipc::ShmPtr<> blob_data,
    const PoolQuery & = PoolQuery::Dynamic(), const Context & = Context());
Future<DelBlobTask> AsyncDelBlob(const TagId&, const std::string &blob_name,
    const PoolQuery & = PoolQuery::Dynamic(), u32 del_flags = 0);
    // del_flags kDelCacheCopyOnly (1): drop only cache copies
Future<GetBlobSizeTask> AsyncGetBlobSize(const TagId&, const std::string&,
    const PoolQuery & = PoolQuery::Dynamic(), int replica = 0);   // ->size_
Future<GetBlobScoreTask> AsyncGetBlobScore(const TagId&, const std::string&,
    const PoolQuery & = PoolQuery::Dynamic());                    // ->score_
Future<GetBlobInfoTask> AsyncGetBlobInfo(const TagId&, const std::string&,
    const PoolQuery & = PoolQuery::Dynamic());  // ->score_, total_size_, blocks_
Future<ReorganizeBlobTask> AsyncReorganizeBlob(const TagId&,
    const std::string &blob_name, float new_score,
    const PoolQuery & = PoolQuery::Dynamic(), int replica = 0);
Future<TruncateBlobTask> AsyncTruncateBlob(const TagId&, const std::string&,
    u64 new_size, const PoolQuery & = PoolQuery::Dynamic());
Future<MultiPutBlobTask> AsyncMultiPutVectored(ctp::ipc::ShmPtr<> data,
    u64 data_len, const std::vector<MultiPutDesc> &descs,
    const PoolQuery & = PoolQuery::Local(), const Context & = Context());
    // MultiPutDesc {tag_id_, blob_name_, offset_, size_, payload_off_,
    //               flags_ (only kCtePutDroppable), score_ = -1}
Future<MultiDelBlobTask> AsyncMultiDelBlob(const std::vector<MultiDelKey>&,
    const PoolQuery & = PoolQuery::Dynamic(), u32 del_flags = 0);
Future<EvictTask> AsyncEvict(float min_tier_score, u64 bytes,
    const PoolQuery & = PoolQuery::Broadcast(), u32 droppable_only = 0);
    // ->bytes_evicted_, blobs_evicted_ (lowest score first, tiers with
    //   score >= min_tier_score)
```

Put flags (`flags`): `kCtePutReplace` 0x1 (resize the blob to exactly
`offset + size`; default puts only grow, like POSIX write),
`kCtePutDroppable` 0x2 (expendable cache data), `kCtePutAppend` 0x4 (write at
the current end; landed offset returned in `task->offset_`). Score: -1 keeps
the existing score (1.0 for a new blob); 0.0-1.0 explicit. The `Tag::PutBlob`
private-buffer form defaults to score **1.0**, not -1. Conditional puts go
through `Context::op_flags_` (`kPutIfAbsent`, `kPutIfVersion` with
`version_`).

### Tags

```cpp
Future<GetOrCreateTagTask<CreateParams>> AsyncGetOrCreateTag(
    const std::string &tag_name, const TagId &tag_id = TagId::GetNull(),
    const PoolQuery & = PoolQuery::Dynamic());       // ->tag_id_, tag_size_
Future<DelTagTask> AsyncDelTag(const TagId&, const PoolQuery & = Dynamic());
Future<DelTagTask> AsyncDelTag(const std::string &tag_name,
    const PoolQuery & = PoolQuery::Dynamic(), bool posix_unlink = false);
Future<MultiDelTagTask> AsyncMultiDelTag(const std::vector<TagId>&,
    const PoolQuery & = PoolQuery::Broadcast());
Future<RenameTagTask> AsyncRenameTag(const std::string &old_name,
    const std::string &new_name, const TagId & = TagId::GetNull(),
    const PoolQuery & = PoolQuery::Dynamic());
Future<GetTagSizeTask> AsyncGetTagSize(const TagId&,
    const PoolQuery & = PoolQuery::Dynamic());
    // ->tag_size_, ctime_, mtime_, atime_. Runs Local: pass Broadcast() for
    //   the whole tag.
Future<GetContainedBlobsTask> AsyncGetContainedBlobs(const TagId&,
    const PoolQuery & = PoolQuery::Dynamic());
    // ->blob_names_. Runs Local: pass Broadcast() for the whole tag.
Future<SyncTagTask> AsyncSyncTag(const TagId&,
    const PoolQuery & = PoolQuery::Broadcast(), int min_persistence = -1);
    // ->deferred_, bdevs_synced_; rc kSyncIoRc 5, kSyncNoSpaceRc 28
```

### Queries

```cpp
Future<TagQueryTask> AsyncTagQuery(const std::string &tag_regex,
    u32 max_tags = 0, const PoolQuery & = PoolQuery::Broadcast()); // ->results_
Future<BlobQueryTask> AsyncBlobQuery(const std::string &tag_regex,
    const std::string &blob_regex, u32 max_blobs = 0,
    const PoolQuery & = PoolQuery::Broadcast());  // ->tag_names_, blob_names_
```

Regexes are full-string `std::regex_match`: use `.*part.*` for a substring.
`SemanticSearch` is served by the indexer (564.0), not by a bare core.

### Targets and durability

```cpp
Future<RegisterTargetTask> AsyncRegisterTarget(const std::string &target_name,
    clio::run::bdev::BdevType bdev_type, u64 total_size,
    const PoolQuery &target_query = PoolQuery::Local(),
    const PoolId &bdev_id = PoolId::GetNull(),
    const PoolQuery &pool_query = PoolQuery::Dynamic(),
    u32 attach_existing = 0, u64 growth_unit = 1ull << 30,
    i32 persistence_level = -1);   // 0 volatile, 1 temporary, 2 long_term,
                                   // -1 = from the storage config, else volatile
Future<UnregisterTargetTask> AsyncUnregisterTarget(const std::string&,
    const PoolQuery & = PoolQuery::Dynamic());
Future<ListTargetsTask> AsyncListTargets(const PoolQuery & = Dynamic());
    // ->target_names_
Future<GetTargetInfoTask> AsyncGetTargetInfo(const std::string &target_name,
    const PoolQuery & = PoolQuery::Dynamic());
Future<GetCapacityTask> AsyncGetCapacity(const PoolQuery & = Local());
    // ->total_capacity_, remaining_capacity_
Future<SetDurabilityTask> AsyncSetDurability(u32 durability,
    const PoolQuery & = PoolQuery::Local());
    // kCteDurabilityVolatile 0 / kCteDurabilityPersistent 1; until restart
```

Compose registers every `storage[]` item; call `RegisterTarget` only to add
a tier at runtime. A registration is not saved: it is gone after a restart
unless the tier is also in the compose entry.

## Return codes

| rc | from | meaning |
|---|---|---|
| 0 | all | success |
| 1 | GetBlob, GetBlobSize, DelBlob | no such blob (authoritative absence) |
| 2 / 3 / 4 / 5 | PutBlob | size 0 / null data / empty blob name / score > 1.0 |
| 1 / 3 | ReorganizeBlob | score outside 0.0-1.0 / blob not found |
| **11** | PutBlob | no target with space. **Also what you get when no target is registered at all** (empty target list, `GetCapacity` 0) |
| 12 / 13 | PutBlob | no target able to hold the request / tier exhausted (all three are out-of-space, retryable) |
| 14 | PutBlob | droppability conflict: blob is droppable, this put is not |
| 15 | PutBlob | `kCtePutAppend` with replace, a replica, or a vectored put. (Create: invalid `durability`) |
| 16 | RegisterTarget, SetDurability | a volatile device in a persistent core |
| 21 | PutBlob | device I/O error during the write |
| 28 | PutBlob | the device itself refused the write for space (ENOSPC). SyncTag: ENOSPC |
| 5 | SyncTag | a device sync failed (EIO) |
| 41-43, 58 | PutBlob through replication | 30 + the core's code for the durable copy |
| 60 / 61 | conditional PutBlob | blob exists / version mismatch |
| 62 | GetBlob | bytes the blob has could not be read (device failed, or range lost on a volatile tier after restart) |
| 63 | GetBlob | owner down and the stand-in holds no copy; existence unknown |
| -1000 (as u32) | any | node lost with the task in flight; resend idempotent ops after failover |

## Python: `clio_cte_core_ext`

Ships in the `iowarp-core` wheel. Import `iowarp_core` first (it sets library
paths). Payloads are `bytes`. Every blocking call releases the GIL.

### Module functions and constants

| name | signature / value |
|---|---|
| `clio_init` | `clio_init(mode, default_with_runtime=False, is_restart=False) -> bool`. `CLIO_WITH_RUNTIME` overrides the second argument. Waits up to `CLIO_WAIT_SERVER` s for the runtime |
| `initialize_cte` | `initialize_cte(config_path, pool_query) -> bool`. Both required; `config_path` is unused, pass `""` and `PoolQuery.Dynamic()` |
| `get_cte_client` | `get_cte_client() -> Client`: a copy of the process-wide client (call after `initialize_cte`) |
| `PUT_EXISTS_RC`, `PUT_VERSION_MISMATCH_RC` | 60, 61 |

### Types

| type | members |
|---|---|
| `UniqueId` (aliases `TagId`, `BlobId`, `PoolId`) | `UniqueId()`, `UniqueId(major, minor)`, `major_`, `minor_`, `IsNull()`, `ToU64()`, static `GetNull()` |
| `PoolQuery` | static `Broadcast(net_timeout=-1.0)`, `Dynamic(net_timeout=-1.0)`, `Local(parallelism=32)` |
| `RuntimeMode` | `kClient`, `kServer`, `kRuntime` |
| `BdevType` | `kFile`, `kRam` |
| `CteOp` | `kPutBlob`, `kGetBlob`, `kDelBlob`, `kGetOrCreateTag`, `kDelTag`, `kGetTagSize` |
| `CapacityInfo` | `total_capacity`, `remaining_capacity` (bytes) |
| `TargetInfo` | `target_score`, `remaining_space`, `bytes_read`, `bytes_written`, `ops_read`, `ops_written` |
| `SemanticSearchResult` | `tag_id`, `blob_name`, `score` |
| `TemporalSearchResult` | `tag_id`, `blob_name`, `last_modified` (steady-clock ns, an ordering key, not Unix time) |
| `CteTelemetry` | `op_`, `off_`, `size_`, `tag_id_`, `mod_time_`, `read_time_`, `logical_time_` |
| `Future` | `wait(max_sec=-1) -> int or None` (rc; None on timeout), `done() -> bool`, `result(max_sec=-1)` (value; raises `TimeoutError`) |

### `Tag` (uses the process-wide client)

| call | returns |
|---|---|
| `Tag(tag_name)` | get-or-create; raises on failure |
| `Tag(tag_id)` | wraps an id, no RPC |
| `.PutBlob(blob_name, data, off=0)` | None; score 1.0 for a new blob; raises `RuntimeError` with the rc |
| `.GetBlob(blob_name, data_size, off=0)` | `bytes` of `data_size`; raises on failure |
| `.GetBlobSize(blob_name)` | int |
| `.GetBlobScore(blob_name)` | float |
| `.GetContainedBlobs()` | `list[str]` (broadcast) |
| `.ReorganizeBlob(blob_name, new_score)` | None |
| `.GetTagId()` | `UniqueId` |

### `Client`

`Client()` (unbound; use `get_cte_client()`), `Client(pool_id)`.

| call | returns |
|---|---|
| `PutBlob(tag_id, blob_name, data, off=0)` | int rc |
| `GetBlob(tag_id, blob_name, data_size, off=0)` | `bytes`, or None on failure |
| `GetBlobSize(tag_id, blob_name)` | int, or None if absent |
| `DelBlob(tag_id, blob_name)` | bool |
| `ReorganizeBlob(tag_id, blob_name, new_score)` | bool |
| `TagQuery(tag_regex, max_tags=0, pool_query=Broadcast())` | `list[str]` |
| `BlobQuery(tag_regex, blob_regex, max_blobs=0, pool_query=Broadcast())` | `list[(tag, blob)]` |
| `SemanticSearch(tag_regex, blob_regex, query_text, k=10, pool_query=Broadcast())` | `list[SemanticSearchResult]` (bind to 564.0) |
| `TemporalSearch(tag_regex, blob_regex, time_begin=0, time_end=0, max_entries=0, pool_query=Broadcast())` | `list[TemporalSearchResult]` |
| `RegisterTarget(target_name, bdev_type, total_size[, target_query, bdev_id])` | int rc |
| `PollTelemetryLog(minimum_logical_time)` | `list[CteTelemetry]` |
| `ConnectCache()` / `HasCache()` | bool |
| `AsyncGetOrCreateTag(tag_name, pool_query=Dynamic())` | Future -> `TagId` |
| `AsyncPutBlob(tag_id, blob_name, data, off=0)` | Future -> None (`wait()` = rc) |
| `AsyncPutBlobIfAbsent(tag_id, blob_name, data, off=0)` | Future -> `{"code", "version"}` |
| `AsyncPutBlobIfVersion(tag_id, blob_name, data, version, off=0)` | Future -> `{"code", "version"}` |
| `AsyncGetBlob(tag_id, blob_name, data_size, off=0)` | Future -> `bytes` |
| `AsyncGetBlobVersioned(tag_id, blob_name, data_size, off=0)` | Future -> `{"code", "version", "data"}` |
| `AsyncGetBlobSize(tag_id, blob_name, pool_query=Dynamic())` | Future -> int |
| `AsyncGetContainedBlobs(tag_id, pool_query=Broadcast())` | Future -> `list[str]` |
| `AsyncDelBlob(tag_id, blob_name)` | Future (`wait()` = rc) |
| `AsyncReorganizeBlob(tag_id, blob_name, new_score)` | Future |
| `AsyncRegisterTarget(target_name, bdev_type, total_size, target_query=Local(), bdev_id=GetNull(), pool_query=Dynamic(), attach_existing=0, growth_unit=1<<30, persistence_level=-1)` | Future (rc 16 = volatile device in persistent core) |
| `AsyncGetCapacity(pool_query=Local())` | Future -> `CapacityInfo` |
| `AsyncListTargets(pool_query=Local())` | Future -> `list[str]` |
| `AsyncStatTargets(pool_query=Local())` | Future |
| `AsyncGetTargetInfo(target_name, pool_query=Local())` | Future -> `TargetInfo` |
| `AsyncTagQuery`, `AsyncBlobQuery`, `AsyncSemanticSearch`, `AsyncTemporalSearch`, `AsyncPollTelemetryLog` | Futures with the blocking forms' arguments and results |

**Not bound in Python on dev:** `SyncTag`, `DelTag`, `GetTagSize`, `Evict`,
`MultiPutBlob`, `TruncateBlob`, `SetDurability`. For a power-loss point from
Python, write through a clio-fs mount and `os.fsync`, or call `SyncTag` from
C++; otherwise rely on the periodic flushes.

### Complete Python example

```python
import os
os.environ.setdefault("CLIO_WITH_RUNTIME", "0")   # attach to `clio-run start`
os.environ.setdefault("CLIO_CTE_POOL", "564.0")   # chain top (default config)
import iowarp_core
import clio_cte_core_ext as cte

assert cte.clio_init(cte.RuntimeMode.kClient, False), "runtime not reachable"
assert cte.initialize_cte("", cte.PoolQuery.Dynamic())
client = cte.get_cte_client()

tag = cte.Tag("experiment/run1")
tag.PutBlob("note0", b"Arctic sea ice extent fell to a record low.", 0)
tag.PutBlob("note1", b"Atmospheric CO2 passed 420 ppm.", 0)

for t, b in client.BlobQuery("experiment/.*", "note.*"):   # broadcast
    print(t, b)
print(tag.GetContainedBlobs())

tid = tag.GetTagId()
futs = [client.AsyncGetBlob(tid, n, tag.GetBlobSize(n)) for n in ("note0", "note1")]
print([f.result() for f in futs])

tag.ReorganizeBlob("note1", 0.0)                 # demote to the slowest tier
won = client.AsyncPutBlobIfAbsent(tid, "lock", b"owner=me").result()
print("lock", "taken" if won["code"] == 0 else "held by another writer")
assert client.DelBlob(tid, "note0")
```
