# INTERPOSER.md: writing a CTE data-path layer

An interposer is a ChiMod that speaks the **CTE core's own task interface** on
its own pool id. Clients point an ordinary `clio::cte::core::Client` at it. It
handles the core verbs it cares about (usually Put and Get) and forwards every
other verb, unchanged, to `next_pool_id`. Every shipped layer works this way:
replication, the write-back and write-thru caches, the shm cache, the indexer,
the compressor, checkpoint and the CAE summarizer.

Below is a complete minimal interposer, `myorg::bytecount`, that counts the
bytes put and got through it and forwards everything. It follows the
replication module (clio-core `context-transfer-engine/replication`, the
reference implementation) and `core_interposer.h`, checked at `dev`
(bd1dc3183). It needs an install built with the CTE (the default), because it
uses the `clio::cte::core_client` target and the `clio_cte/core/*.h` headers.

## 1. What the base class gives you

`clio::cte::core::CoreInterposer` (`<clio_cte/core/core_interposer.h>`,
header-only) derives from `clio::run::Container` and provides:

| member | what it does |
|---|---|
| `interposer_next_pool_` (protected `PoolId`) | the pool you forward to. Set it in `Create`. Null falls back to the canonical core, 512.0 (`kCtePoolId`) |
| `CorePoolId()` | `interposer_next_pool_`, or 512.0 if null |
| `GetNextPool()` | override that reports the next pool to the runtime |
| `ScheduleTask(task)` | delegates to the next pool's routing, so blob verbs hash to the blob's owner exactly as at the core. **Do not override it for core verbs**: chain operations would then resolve on the submitting node, and readers on other nodes would never see the writer's blobs |
| `GetTaskStats(task)` | the core's cost model. Override only for your own verbs |
| `ForwardToCore(method, task)` | runs the task on the next pool's **local** container, verbatim; sets rc 1 if that pool does not exist |
| `ForwardSaveTask`, `ForwardLoadTask`, `ForwardLocalLoadTask`, `ForwardLocalSaveTask`, `ForwardNewTask`, `ForwardNewCopyTask`, `ForwardAggregateOut`, `ForwardAggregateIn` | the next pool's serialization and allocation, for the `default:` cases of your dispatch |

## 2. Rules

1. **Interposed core verbs keep the core's method ids and the core's task
   structs.** Take the ids from `clio::cte::core::Method::k...`
   (`<clio_cte/core/autogen/core_methods.h>`). Data verbs: `kPutBlob` 15,
   `kGetBlob` 16, `kDelBlob` 18, `kGetBlobSize` 23, `kTruncateBlob` 37,
   `kPodPutBlob` 43, `kPodGetBlob` 44, `kMultiPutBlob` 48,
   `kPodMultiPutBlob` 51, `kPodMultiGetBlob` 52, `kSyncTag` 56,
   `kMultiDelBlob` 60. Tag verbs: `kGetOrCreateTag` 14, `kDelTag` 19.
   The core uses ids up to 62.
2. **Your own verbs are 100 and up.** A collision shadows a forwarded core
   verb, and the bug looks like the wrong handler crashing.
3. **If you transform data, interpose every verb that carries it.** A layer
   that handles only `kPutBlob`/`kGetBlob` is bypassed by vectored puts
   (`segments_`), `kMultiPutBlob`, the `kPod*` GPU verbs and truncation, which
   reach the core untransformed. Either handle them, or document the gap.
4. **The dispatch file is written by hand.** `clio-run refresh` cannot
   generate the forwarding `default:` cases. Keep the interposer out of the
   `modules:` list of any `clio_repo.yaml`, or the generator overwrites your
   dispatch. `clio_mod.yaml` is still required (CMake reads `module_name` and
   `namespace` from it).
5. **Compose with `pool_query: local`, after the next pool, on every node.**
   `ForwardToCore` calls the next pool's container on the same node: routing
   already sent the task to the blob owner's node. Container counts must match
   the core's (one per node), or `ScheduleTask`'s answer is wrong for your pool.
6. **Reaching another node** (for example a replica) needs a client and a
   `DirectId`/`DirectHash` query that you await; `ForwardToCore` never leaves
   the node.
7. **Preserve the core's semantics for what you do not change.** Pass return
   codes through. A Put you transform still reports the logical size.
8. **State your guarantees** where users will read them: what an ack means,
   what you hold in memory and how it becomes durable, what a crash loses,
   how you stay coherent across nodes. Anything cached is rebuilt from a log
   on `Restart` or dropped; never trust memory across a restart.
9. **Test on two runtimes.** On one node every verb is local and routing bugs
   hide. Write on one runtime, read on the other, then kill and restart one
   for anything that claims durability.

## 3. Files

Same layout as TEMPLATE.md, with `bytecount` for `counter`. `modules/
clio_repo.yaml` must NOT list `bytecount`.

### `clio_mod.yaml`

```yaml
module_name: bytecount
namespace: myorg
version: 1.0.0
kCreate: 0
kDestroy: 1
kMonitor: 9
kPutBlob: 15     # interposed core verb: same id as the core
kGetBlob: 16     # interposed core verb: same id as the core
kGetCounts: 100  # own verb
```

### `include/myorg/bytecount/autogen/bytecount_methods.h` (hand-written)

```cpp
#ifndef MYORG_BYTECOUNT_METHODS_H_
#define MYORG_BYTECOUNT_METHODS_H_

#include <string>
#include <vector>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/autogen/core_methods.h>

namespace myorg::bytecount::Method {
GLOBAL_CROSS_CONST clio::run::u32 kCreate = 0;
GLOBAL_CROSS_CONST clio::run::u32 kDestroy = 1;
GLOBAL_CROSS_CONST clio::run::u32 kMonitor = 9;
GLOBAL_CROSS_CONST clio::run::u32 kPutBlob = clio::cte::core::Method::kPutBlob;
GLOBAL_CROSS_CONST clio::run::u32 kGetBlob = clio::cte::core::Method::kGetBlob;
GLOBAL_CROSS_CONST clio::run::u32 kGetCounts = 100;
GLOBAL_CROSS_CONST clio::run::u32 kMaxMethodId = 101;

/** Method names by id, for the runtime's per-method statistics. */
inline const std::vector<std::string> &GetMethodNames() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v(kMaxMethodId);
    v[kCreate] = "Create";
    v[kDestroy] = "Destroy";
    v[kMonitor] = "Monitor";
    v[kPutBlob] = "PutBlob";
    v[kGetBlob] = "GetBlob";
    v[kGetCounts] = "GetCounts";
    return v;
  }();
  return names;
}
}  // namespace myorg::bytecount::Method

#endif  // MYORG_BYTECOUNT_METHODS_H_
```

### `include/myorg/bytecount/bytecount_tasks.h`

```cpp
#ifndef MYORG_BYTECOUNT_TASKS_H_
#define MYORG_BYTECOUNT_TASKS_H_

#include <string>

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/admin/admin_tasks.h>
#include <clio_cte/core/core_tasks.h>
#include <yaml-cpp/yaml.h>

#include "autogen/bytecount_methods.h"

namespace myorg::bytecount {

static constexpr clio::run::PoolId kDefaultPoolId(701, 0);

/** Compose: next_pool_id is the pool every verb is forwarded to. */
struct CreateParams {
  static constexpr const char *chimod_lib_name = "myorg_bytecount";
  clio::run::PoolId next_pool_id_;

  CreateParams() : next_pool_id_(clio::run::PoolId::GetNull()) {}

  template <class Archive>
  void serialize(Archive &ar) { ar(next_pool_id_); }

  /** @param pool_config whole compose entry as YAML in config_ */
  void LoadConfig(const clio::run::PoolConfig &pool_config) {
    if (pool_config.config_.empty()) return;
    try {
      YAML::Node node = YAML::Load(pool_config.config_);
      if (node["next_pool_id"]) {
        next_pool_id_ = clio::run::PoolId::FromString(
            node["next_pool_id"].as<std::string>());
      }
    } catch (const std::exception &e) {
      HLOG(kError, "bytecount: bad compose entry: {}", e.what());
    }
  }
};

using CreateTask = clio::run::admin::GetOrCreatePoolTask<CreateParams>;
using DestroyTask = clio::run::admin::DestroyTask;
using MonitorTask = clio::run::admin::MonitorTask;

/** Own verb: bytes counted by the containers it reaches (sum over replicas). */
struct GetCountsTask : public clio::run::Task {
  OUT clio::run::u64 put_bytes_;
  OUT clio::run::u64 get_bytes_;

  GetCountsTask() : clio::run::Task(), put_bytes_(0), get_bytes_(0) {}
  explicit GetCountsTask(const clio::run::TaskId &task_id,
                         const clio::run::PoolId &pool_id,
                         const clio::run::PoolQuery &pool_query)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kGetCounts),
        put_bytes_(0), get_bytes_(0) {}

  template <typename Ar> void SerializeIn(Ar &ar) { Task::SerializeIn(ar); }
  template <typename Ar> void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar);
    ar(put_bytes_, get_bytes_);
  }
  void Copy(const ctp::ipc::FullPtr<GetCountsTask> &o) {
    Task::Copy(o.template Cast<clio::run::Task>());
    put_bytes_ = o->put_bytes_;
    get_bytes_ = o->get_bytes_;
  }
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &b) {
    Task::AggregateOut(b);
    auto o = b.template Cast<GetCountsTask>();
    put_bytes_ += o->put_bytes_;
    get_bytes_ += o->get_bytes_;
  }
};

}  // namespace myorg::bytecount
#endif  // MYORG_BYTECOUNT_TASKS_H_
```

### `include/myorg/bytecount/bytecount_client.h`

Only Create and the own verb. Blob traffic goes through the core's client.

```cpp
#ifndef MYORG_BYTECOUNT_CLIENT_H_
#define MYORG_BYTECOUNT_CLIENT_H_

#include <string>
#include "bytecount_tasks.h"

namespace myorg::bytecount {

class Client : public clio::run::ContainerClient {
 public:
  Client() = default;
  explicit Client(const clio::run::PoolId &pool_id) { Init(pool_id); }

  /** Create over next_pool_id (API create; compose is the usual route). */
  clio::run::Future<CreateTask> AsyncCreate(
      const clio::run::PoolQuery &pool_query, const std::string &pool_name,
      const clio::run::PoolId &pool_id, const CreateParams &params);

  /** Sum of the counters; use Broadcast() to cover every node. */
  clio::run::Future<GetCountsTask> AsyncGetCounts(
      const clio::run::PoolQuery &pool_query);
};

}  // namespace myorg::bytecount
#endif  // MYORG_BYTECOUNT_CLIENT_H_
```

`src/bytecount_client.cc`:

```cpp
#include "myorg/bytecount/bytecount_client.h"

namespace myorg::bytecount {

clio::run::Future<CreateTask> Client::AsyncCreate(
    const clio::run::PoolQuery &pool_query, const std::string &pool_name,
    const clio::run::PoolId &pool_id, const CreateParams &params) {
  auto *ipc = CLIO_IPC;
  auto task = ipc->NewTask<CreateTask>(
      clio::run::CreateTaskId(), clio::run::kAdminPoolId, pool_query,
      CreateParams::chimod_lib_name, pool_name, pool_id, this, params);
  return ipc->Send(task);
}

clio::run::Future<GetCountsTask> Client::AsyncGetCounts(
    const clio::run::PoolQuery &pool_query) {
  auto *ipc = CLIO_IPC;
  auto task = ipc->NewTask<GetCountsTask>(clio::run::CreateTaskId(), pool_id_,
                                          pool_query);
  return ipc->Send(task);
}

}  // namespace myorg::bytecount
```

### `include/myorg/bytecount/bytecount_runtime.h`

```cpp
#ifndef MYORG_BYTECOUNT_RUNTIME_H_
#define MYORG_BYTECOUNT_RUNTIME_H_

#include <atomic>
#include <string>

#include <clio_runtime/clio_runtime.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_interposer.h>

#include "bytecount_client.h"
#include "bytecount_tasks.h"

namespace myorg::bytecount {

class Runtime : public clio::cte::core::CoreInterposer {
 public:
  using CreateParams = myorg::bytecount::CreateParams;  // for CLIO_TASK_CC
  ~Runtime() override = default;

  clio::run::TaskResume Create(clio::run::shared_ptr<CreateTask> &task);
  clio::run::TaskResume Destroy(clio::run::shared_ptr<DestroyTask> &task);
  clio::run::TaskResume Monitor(clio::run::shared_ptr<MonitorTask> &task);
  clio::run::TaskResume PutBlob(
      clio::run::shared_ptr<clio::cte::core::PutBlobTask> &task);
  clio::run::TaskResume GetBlob(
      clio::run::shared_ptr<clio::cte::core::GetBlobTask> &task);
  clio::run::TaskResume GetCounts(clio::run::shared_ptr<GetCountsTask> &task);

  // Dispatch, in src/bytecount_lib_exec.cc (hand-written).
  void Init(const clio::run::PoolId &pool_id, const std::string &pool_name,
            clio::run::u32 container_id = 0) override;
  clio::run::TaskResume Run(clio::run::u32 method,
                            clio::run::shared_ptr<clio::run::Task> task_ptr) override;
  clio::run::u64 GetWorkRemaining() const override { return 0; }
  void SaveTask(clio::run::u32 method, clio::run::SaveTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task> &task_ptr) override;
  void LoadTask(clio::run::u32 method, clio::run::LoadTaskArchive &archive,
                clio::run::shared_ptr<clio::run::Task> &task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> AllocLoadTask(
      clio::run::u32 method, clio::run::LoadTaskArchive &archive) override;
  void LocalLoadTask(clio::run::u32 method, clio::run::DefaultLoadArchive &archive,
                     clio::run::shared_ptr<clio::run::Task> &task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> LocalAllocLoadTask(
      clio::run::u32 method, clio::run::DefaultLoadArchive &archive) override;
  void LocalSaveTask(clio::run::u32 method, clio::run::DefaultSaveArchive &archive,
                     clio::run::shared_ptr<clio::run::Task> &task_ptr) override;
  clio::run::shared_ptr<clio::run::Task> NewCopyTask(
      clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig,
      bool deep) override;
  clio::run::shared_ptr<clio::run::Task> NewTask(clio::run::u32 method) override;
  void AggregateOut(clio::run::u32 method,
                    clio::run::shared_ptr<clio::run::Task> &orig_task,
                    const clio::run::shared_ptr<clio::run::Task> &replica_task) override;
  void AggregateIn(clio::run::u32 method,
                   clio::run::shared_ptr<clio::run::Task> &agg_task,
                   const clio::run::shared_ptr<clio::run::Task> &member_task) override;

 private:
  std::atomic<clio::run::u64> put_bytes_{0};
  std::atomic<clio::run::u64> get_bytes_{0};
};

}  // namespace myorg::bytecount
#endif  // MYORG_BYTECOUNT_RUNTIME_H_
```

### `src/bytecount_runtime.cc`: the handlers

```cpp
#include "myorg/bytecount/bytecount_runtime.h"

#include <clio_ctp/serialize/msgpack_wrapper.h>

namespace myorg::bytecount {

namespace core = clio::cte::core;

clio::run::TaskResume Runtime::Create(clio::run::shared_ptr<CreateTask> &task) {
  CLIO_TASK_BODY_BEGIN
  auto &fields = task.template Cast<clio::run::admin::CreatePoolFields>();
  CreateParams params = fields->GetParamsAs<CreateParams>();
  interposer_next_pool_ = params.next_pool_id_;   // where everything goes
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Destroy(clio::run::shared_ptr<DestroyTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Monitor(clio::run::shared_ptr<MonitorTask> &task) {
  CLIO_TASK_BODY_BEGIN
  msgpack::sbuffer sbuf;
  msgpack::packer<msgpack::sbuffer> pk(sbuf);
  pk.pack_map(2);
  pk.pack("put_bytes");
  pk.pack(static_cast<uint64_t>(put_bytes_.load()));
  pk.pack("get_bytes");
  pk.pack(static_cast<uint64_t>(get_bytes_.load()));
  task->results_[container_id_] = std::string(sbuf.data(), sbuf.size());
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::PutBlob(
    clio::run::shared_ptr<core::PutBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // segments_ non-empty = vectored put; then offset_/size_ are ignored.
  clio::run::u64 bytes = task->size_;
  if (!task->segments_.empty()) {
    bytes = 0;
    for (size_t i = 0; i < task->segments_.size(); ++i) {
      bytes += task->segments_[i].size_;
    }
  }
  CLIO_CO_AWAIT(ForwardToCore(core::Method::kPutBlob,
                              task.template Cast<clio::run::Task>()));
  if (task->GetReturnCode() == 0) put_bytes_.fetch_add(bytes);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::GetBlob(
    clio::run::shared_ptr<core::GetBlobTask> &task) {
  CLIO_TASK_BODY_BEGIN
  CLIO_CO_AWAIT(ForwardToCore(core::Method::kGetBlob,
                              task.template Cast<clio::run::Task>()));
  if (task->GetReturnCode() == 0) get_bytes_.fetch_add(task->size_);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::GetCounts(
    clio::run::shared_ptr<GetCountsTask> &task) {
  CLIO_TASK_BODY_BEGIN
  task->put_bytes_ = put_bytes_.load();
  task->get_bytes_ = get_bytes_.load();
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

}  // namespace myorg::bytecount

CLIO_TASK_CC(myorg::bytecount::Runtime)
```

To transform data instead of counting it, change the task's buffer before
`ForwardToCore` (Put) or after it (Get). To send work elsewhere, build a
`clio::cte::core::Client(CorePoolId())` member in `Create` and await its
`Async*` futures instead of forwarding.

### `src/bytecount_lib_exec.cc`: the hand-written dispatch

One X-macro lists the methods you implement; every `default:` forwards.

```cpp
#include "myorg/bytecount/bytecount_runtime.h"

namespace myorg::bytecount {

#define BYTECOUNT_METHODS(X)                                  \
  X(kCreate, CreateTask, Create)                              \
  X(kDestroy, DestroyTask, Destroy)                           \
  X(kMonitor, MonitorTask, Monitor)                           \
  X(kPutBlob, clio::cte::core::PutBlobTask, PutBlob)          \
  X(kGetBlob, clio::cte::core::GetBlobTask, GetBlob)          \
  X(kGetCounts, GetCountsTask, GetCounts)

void Runtime::Init(const clio::run::PoolId &pool_id,
                   const std::string &pool_name, clio::run::u32 container_id) {
  clio::run::Container::Init(pool_id, pool_name, container_id);
  DefineModel(Method::kMaxMethodId);
  SetMethodNames(Method::GetMethodNames());
}

clio::run::TaskResume Runtime::Run(clio::run::u32 method,
                                   clio::run::shared_ptr<clio::run::Task> task_ptr) {
  CLIO_TASK_BODY_BEGIN
  switch (method) {
#define X(MID, TASK, HANDLER)                       \
    case Method::MID: {                             \
      auto &typed = task_ptr.template Cast<TASK>(); \
      CLIO_CO_AWAIT(HANDLER(typed));                \
      break;                                        \
    }
    BYTECOUNT_METHODS(X)
#undef X
    default:
      CLIO_CO_AWAIT(ForwardToCore(method, task_ptr));
      break;
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

void Runtime::SaveTask(clio::run::u32 method, clio::run::SaveTaskArchive &archive,
                       clio::run::shared_ptr<clio::run::Task> &task_ptr) {
  switch (method) {
#define X(MID, TASK, HANDLER) \
    case Method::MID: archive << *task_ptr.template Cast<TASK>(); break;
    BYTECOUNT_METHODS(X)
#undef X
    default: ForwardSaveTask(method, archive, task_ptr); break;
  }
}

void Runtime::LoadTask(clio::run::u32 method, clio::run::LoadTaskArchive &archive,
                       clio::run::shared_ptr<clio::run::Task> &task_ptr) {
  switch (method) {
#define X(MID, TASK, HANDLER) \
    case Method::MID: archive >> *task_ptr.template Cast<TASK>(); break;
    BYTECOUNT_METHODS(X)
#undef X
    default: ForwardLoadTask(method, archive, task_ptr); break;
  }
}

clio::run::shared_ptr<clio::run::Task> Runtime::AllocLoadTask(
    clio::run::u32 method, clio::run::LoadTaskArchive &archive) {
  clio::run::shared_ptr<clio::run::Task> t = NewTask(method);
  if (!t.IsNull()) LoadTask(method, archive, t);
  return t;
}

void Runtime::LocalLoadTask(clio::run::u32 method,
                            clio::run::DefaultLoadArchive &archive,
                            clio::run::shared_ptr<clio::run::Task> &task_ptr) {
  switch (method) {
#define X(MID, TASK, HANDLER) \
    case Method::MID: archive >> *task_ptr.template Cast<TASK>(); break;
    BYTECOUNT_METHODS(X)
#undef X
    default: ForwardLocalLoadTask(method, archive, task_ptr); break;
  }
}

clio::run::shared_ptr<clio::run::Task> Runtime::LocalAllocLoadTask(
    clio::run::u32 method, clio::run::DefaultLoadArchive &archive) {
  clio::run::shared_ptr<clio::run::Task> t = NewTask(method);
  if (!t.IsNull()) LocalLoadTask(method, archive, t);
  return t;
}

void Runtime::LocalSaveTask(clio::run::u32 method,
                            clio::run::DefaultSaveArchive &archive,
                            clio::run::shared_ptr<clio::run::Task> &task_ptr) {
  switch (method) {
#define X(MID, TASK, HANDLER) \
    case Method::MID: archive << *task_ptr.template Cast<TASK>(); break;
    BYTECOUNT_METHODS(X)
#undef X
    default: ForwardLocalSaveTask(method, archive, task_ptr); break;
  }
}

clio::run::shared_ptr<clio::run::Task> Runtime::NewCopyTask(
    clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig,
    bool deep) {
  auto *ipc = CLIO_IPC;
  switch (method) {
#define X(MID, TASK, HANDLER)                                              \
    case Method::MID: {                                                    \
      auto t = ipc->NewTask<TASK>();                                       \
      if (!t.IsNull()) {                                                   \
        t->Copy(ctp::ipc::FullPtr<TASK>(orig.template Cast<TASK>().get())); \
        return t.template Cast<clio::run::Task>();                         \
      }                                                                    \
      break;                                                               \
    }
    BYTECOUNT_METHODS(X)
#undef X
    default: return ForwardNewCopyTask(method, orig, deep);
  }
  return clio::run::shared_ptr<clio::run::Task>();
}

clio::run::shared_ptr<clio::run::Task> Runtime::NewTask(clio::run::u32 method) {
  auto *ipc = CLIO_IPC;
  switch (method) {
#define X(MID, TASK, HANDLER) \
    case Method::MID: return ipc->NewTask<TASK>().template Cast<clio::run::Task>();
    BYTECOUNT_METHODS(X)
#undef X
    default: return ForwardNewTask(method);
  }
}

void Runtime::AggregateOut(clio::run::u32 method,
                           clio::run::shared_ptr<clio::run::Task> &orig_task,
                           const clio::run::shared_ptr<clio::run::Task> &replica_task) {
  switch (method) {
#define X(MID, TASK, HANDLER)                                 \
    case Method::MID:                                         \
      orig_task.template Cast<TASK>()->AggregateOut(          \
          ctp::ipc::FullPtr<clio::run::Task>(replica_task.get())); \
      break;
    BYTECOUNT_METHODS(X)
#undef X
    default: ForwardAggregateOut(method, orig_task, replica_task); break;
  }
}

void Runtime::AggregateIn(clio::run::u32 method,
                          clio::run::shared_ptr<clio::run::Task> &agg_task,
                          const clio::run::shared_ptr<clio::run::Task> &member_task) {
  ForwardAggregateIn(method, agg_task, member_task);  // no collectives of our own
}

#undef BYTECOUNT_METHODS
}  // namespace myorg::bytecount
```

### `CMakeLists.txt` (module directory)

```cmake
add_clio_module_client(
  SOURCES src/bytecount_client.cc
  LINK_LIBRARIES clio::run::admin_client clio::cte::core_client)
add_clio_module_runtime(
  SOURCES src/bytecount_runtime.cc src/bytecount_lib_exec.cc
  LINK_LIBRARIES clio::run::admin_runtime clio::cte::core_client)
```

## 4. Compose and use it

```yaml
compose:
  # ... the CTE core at 512.0 (and anything else you forward to) first ...
  - mod_name: myorg_bytecount
    pool_name: bytecount
    pool_query: local          # required: one container per node
    pool_id: "701.0"
    next_pool_id: "512.0"      # the persistent core; or the chain top
```

Where to put it in the shipped chain (default config, top to bottom):
filesystem 560.0 -> indexer 564.0 -> shm cache 568.0 -> write-back 567.0 ->
replication 561.0 -> core 512.0. To sit between two layers, set your
`next_pool_id` to the lower one and change the upper one's `next_pool_id` (or
`next_persistent`) to your id, and compose yours between them. Anything that
does not name your pool bypasses it.

A client uses the core's API on your pool id:

```cpp
clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false);
clio::cte::core::Client cte(myorg::bytecount::kDefaultPoolId);
auto tag = cte.AsyncGetOrCreateTag("demo");      // forwarded to 512.0
tag.Wait();
auto *ipc = CLIO_IPC;
auto buf = ipc->AllocateBuffer(4096);
auto put = cte.AsyncPutBlob(tag->tag_id_, "b0", 0, 4096,
                            ctp::ipc::ShmPtr<>(buf.shm_));   // counted
put.Wait();
ipc->FreeBuffer(buf);
myorg::bytecount::Client bc(myorg::bytecount::kDefaultPoolId);
auto counts = bc.AsyncGetCounts(clio::run::PoolQuery::Broadcast());
counts.Wait();            // counts->put_bytes_ == 4096 (summed over nodes)
```

A process whose CTE client is set up by `CLIO_CTE_CLIENT_INIT` can be bound to
your pool instead of 512.0 with `CLIO_CTE_POOL=701.0`.
