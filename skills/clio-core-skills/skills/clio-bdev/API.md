# Client API

Most users never call these. The CTE allocates, writes, reads and syncs bdev
blocks for you; use the CTE client (`clio_cte_core_ext` in Python) instead.
Call a bdev directly only in a custom ChiMod, a benchmark, or a test. The C++
headers and CMake package are **not** in the pip wheel; they come with a
source, conda or Spack install (`find_package(clio-core)`, target
`clio::run::bdev_client`).

All methods return `clio::run::Future<Task>`. Call `.Wait()`, then read
`->GetReturnCode()` (0 = success) and the output fields.

## bdev (`#include <clio_runtime/bdev/bdev_client.h>`, namespace `clio::run::bdev`)

```cpp
class Client {
 public:
  explicit Client(const clio::run::PoolId &pool_id);

  Future<CreateTask> AsyncCreate(const PoolQuery &pool_query,
      const std::string &pool_name, const PoolId &custom_pool_id,
      BdevType bdev_type, u64 total_size = 0, u32 io_depth = 32,
      u32 alignment = 4096, const PerfMetrics *perf_metrics = nullptr,
      const std::string &alloc_log_path = "", u64 growth_unit = 1ull << 30,
      int persistence_level = -1);   // -1 unset, 0 volatile, 1 temporary, 2 long_term
  Future<AllocateBlocksTask> AsyncAllocateBlocks(const PoolQuery &q, u64 size);
  Future<FreeBlocksTask> AsyncFreeBlocks(const PoolQuery &q, const std::vector<Block> &blocks);
  Future<WriteTask> AsyncWrite(const PoolQuery &q, const priv::vector<Block> &blocks,
                               ctp::ipc::ShmPtr<> data, size_t length);
  Future<ReadTask> AsyncRead(const PoolQuery &q, const priv::vector<Block> &blocks,
                             ctp::ipc::ShmPtr<> data, size_t buffer_size);
  Future<SyncTask> AsyncSync(const PoolQuery &q);          // fdatasync (file); no-op for memory
  Future<GetStatsTask> AsyncGetStats(const PoolQuery &q = PoolQuery::Local());
  Future<FlushAllocLogTask> AsyncFlushAllocLog(const PoolQuery &q, double period_us = 0);
  Future<SetLifespanTask> AsyncSetLifespan(const PoolQuery &q, u32 lifespan_days);
  Future<UpdateTask> AsyncUpdate(const PoolQuery &q);      // publish hbm/pinned pointers
  Future<MonitorTask> AsyncMonitor(const PoolQuery &q, const std::string &query); // "stats"
};
```

| Type / field | Meaning |
|---|---|
| `BdevType` | `kFile`=0, `kRam`=1, `kHbm`=2, `kPinned`=3, `kNoop`=4, `kS3`=5, `kGcs`=6, `kDaos`=7 |
| `Block` | `offset_`, `size_` (bytes), `block_type_` (size class). One allocation may return several |
| `AllocateBlocksTask::blocks_` | the extents; rc 1 = out of space or growth failed |
| `WriteTask::bytes_written_`, `io_error_` | `io_error_`: 0 ok, 1 transient, 2 device fault, 3 disconnected, 4 no space, 5 permission, 6 invalid, 7 timeout, 8 unknown |
| `ReadTask::bytes_read_`, `length_`, `io_error_` | `length_` comes back as the bytes actually read |
| `GetStatsTask` | `metrics_` (PerfMetrics), `total_size_`, `remaining_size_`, `predicted_ttl_days_` (999999 = healthy) |
| `SyncTask` | rc != 0: the device could not guarantee durability |

Create return codes: 0 ok, 1 transport not available for this type (not
compiled in), 2 transport init failed (cannot open file, missing
credentials, cudaMalloc failure).

`CreateTask` always goes to the admin pool; `AsyncCreate` does that for you.
Use `PoolQuery::Local()` for node-local pools.

Example (inside a client process that called `CLIO_INIT`):

```cpp
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>
using namespace clio::run;

bdev::Client disk;
PoolId id(302, 0);
auto c = disk.AsyncCreate(PoolQuery::Local(), "/mnt/nvme/test.bin", id,
                          bdev::BdevType::kFile, 1ull << 30);
c.Wait();
if (c->GetReturnCode() != 0) return 1;
disk.pool_id_ = c->new_pool_id_;

auto a = disk.AsyncAllocateBlocks(PoolQuery::Local(), 1 << 20);
a.Wait();                                   // a->blocks_ holds the extents
auto buf = CLIO_IPC->AllocateBuffer(1 << 20);
std::memset(buf.ptr_, 7, 1 << 20);
auto w = disk.AsyncWrite(PoolQuery::Local(), a->blocks_,
                         buf.shm_.template Cast<void>(), 1 << 20);
w.Wait();
auto s = disk.AsyncSync(PoolQuery::Local());
s.Wait();                                   // now durable on disk
```

## safe_bdev (`#include <clio_runtime/safe_bdev/safe_bdev_client.h>`, namespace `clio::run::safe_bdev`)

The data plane reuses the bdev task types and method ids (AllocateBlocks 10,
FreeBlocks 11, Write 12, Read 13, GetStats 14, Sync 18), so a
`bdev::Client` pointed at the array's pool id works unchanged; that is how
the CTE uses it.

```cpp
Future<CreateTask> AsyncCreate(const PoolQuery &q, const std::string &pool_name,
    const PoolId &custom_pool_id, u32 max_failures,
    const std::vector<MemberBdevDesc> &members,
    const std::string &alloc_log_path = "", bool distributed = false);
    // MemberBdevDesc(pool_name, node_id, pool_id, parity = false)
    // intent_sync is compose-only (YAML); the C++ create keeps the default true
Future<AddBdevTask> AsyncAddBdev(const PoolQuery &q, const std::string &pool_name,
    u32 node_id, const PoolId &member_pool_id, u32 as_parity = 0);
Future<RemoveBdevTask> AsyncRemoveBdev(const PoolQuery &q,
    const PoolId &target_pool_id, u32 was_faulty);
Future<RecoverBdevTask> AsyncRecoverBdev(const PoolQuery &q, const PoolId &old_bdev_id,
    const std::string &pool_name, u32 node_id, const PoolId &new_pool_id);
Future<BuildParityTask> AsyncBuildParity(const PoolQuery &q, u32 max_batch = 0,
    double period_us = 0);         // started automatically every 50 ms
Future<FlushAllocLogTask> AsyncFlushAllocLog(const PoolQuery &q, double period_us = 0);
// plus AsyncAllocateBlocks / AsyncFreeBlocks / AsyncWrite / AsyncRead /
// AsyncGetStats() / AsyncMonitor(q, "stats") with the bdev signatures
```

The member pool for `AddBdev` / `RecoverBdev` must already exist. Return
code 40 means the task reached a node that is not the array's home
(`distributed: true`); send it to the first member's node.

## Python (`clio_runtime_ext`, in the wheel)

There is no Python binding for bdev I/O. Three helpers drive safe_bdev
membership (they fail in a wheel-only install because the safe_bdev module
itself is absent):

```python
import clio_runtime_ext as rt
rt.clio_init(0)                                   # 0 = client
rt.safe_bdev_remove_bdev("350.0", "302.0", was_faulty=1)          # -> rc
new_id = rt.safe_bdev_add_bdev("350.0", "/mnt/nvme4/d4.bin",
                               capacity="1TB", node_id=0, as_parity=0)  # -> "major.minor" or ""
new_id = rt.safe_bdev_replace_bdev("350.0", "302.0", "/mnt/nvme3/d0b.bin",
                                   capacity="1TB", node_id=0)     # remove + create + recover
```

These create the new member as a file bdev with `alloc_log: <path>.alog` at
a pool id from 40000 upward, **not** marked restartable: add the member to
the compose file before the next restart, or use `clio_safe_bdev_recover`.

`clio_cte_core_ext.Client.AsyncRegisterTarget(target_name, bdev_type,
total_size, target_query, bdev_id, pool_query, attach_existing, growth_unit,
persistence_level)` registers a tier at runtime; its `BdevType` enum exposes
only `kFile` and `kRam`.
