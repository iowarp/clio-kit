# REFERENCE.md: chimod mechanics as they exist in source

The template module is `context-runtime/modules/MOD_NAME/`. The line numbers
below are approximate; search for the symbol.

## 1. Layout

| file | role | written by |
|---|---|---|
| `clio_mod.yaml` | module name, namespace, method ids | you |
| `CMakeLists.txt` | client and runtime library targets | you |
| `include/<prefix>/<mod>/<mod>_tasks.h` | `CreateParams`, task structs, Create/Destroy/Monitor aliases | you |
| `include/<prefix>/<mod>/<mod>_client.h` | `Client : clio::run::ContainerClient`, `AsyncX` methods | you |
| `include/<prefix>/<mod>/<mod>_runtime.h` | `Runtime : clio::run::Container`, handler and autogen-override declarations | you |
| `include/<prefix>/<mod>/autogen/<mod>_methods.h` | `Method::kX`, `kMaxMethodId`, `GetMethodNames()` | codegen |
| `src/<mod>_client.cc` | out-of-line `chimod_lib_name` definition | you |
| `src/<mod>_runtime.cc` | handler bodies; ends with `CLIO_TASK_CC(<ns>::Runtime)` | you |
| `src/autogen/<mod>_lib_exec.cc` | `Init`, `Run`, `SaveTask`, `LoadTask`, `AllocLoadTask`, `Local*Task`, `NewTask`, `NewCopyTask`, `AggregateOut`, `AggregateIn` | codegen |
| `test/` | simple_test.h tests | you |

`<prefix>` is the first `include/<dir>` that already contains `<mod>/`.
Otherwise it is the namespace with `::` replaced by `_`, for example
`clio_runtime`, `clio_cte` or `clio_cae`.

## 2. YAML

`clio_repo.yaml` sits in the directory that holds the modules. Only
`namespace` and `modules` are read:

```yaml
name: modules
namespace: clio::run
version: 1.0.0
modules:
  - MOD_NAME
  - admin
  - bdev
```

`clio_mod.yaml` has `module_name` and `namespace` (CMake requires both; the
module's namespace overrides the repo's). Methods are **top-level keys that
start with `k` and have integer values**:

```yaml
module_name: my_mod
namespace: clio::run
version: 1.0.0
kCreate: 0
kDestroy: 1
kNodeFailure: -1     # -1 = not implemented (no dispatch generated)
kRecover: -1
kMigrate: -1
kUpgrade: -1
kMonitor: 9
kCustom: 10
```

Any other schema is ignored silently. That includes a `methods:` list or
`tasks:`, both of which appear in old docs. `kRestart >= 0` generates a
`Runtime::Restart` override instead of a dispatch case.

## 3. Codegen

```bash
<build>/bin/clio-run refresh <repo_dir>       # e.g. context-runtime/modules
```

The nested spelling `clio-run repo refresh <dir>` also works. It takes exactly
one argument. For each module in `modules:`, it rewrites `_methods.h` and
`_lib_exec.cc`. Commit both.

**Naming contract.**
- Method `kFoo` dispatches to `Runtime::Foo(clio::run::shared_ptr<FooTask>&)`.
- `kCreate` maps to `CreateTask`, and `kDestroy` to `DestroyTask`.
- So the tasks header must provide `CreateTask`, `DestroyTask` and
  `MonitorTask`, usually as aliases:

```cpp
using CreateTask  = clio::run::admin::GetOrCreatePoolTask<CreateParams>;
using DestroyTask = clio::run::admin::DestroyTask;
using MonitorTask = clio::run::admin::MonitorTask;
```

The generated `Init` constructs `client_ = Client(pool_id)`, so `Runtime` must
have a `Client client_` member.

## 4. Tasks

**Base class.** `clio::run::Task` (`context-runtime/include/clio_runtime/task.h`).
- IN fields: `pool_id_`, `task_id_`, `pool_query_`, `method_`, `task_flags_`,
  `period_ns_`, `task_group_`.
- OUT fields: atomic `return_code_`, `completer_`.
- Annotation macros `IN`, `OUT`, `INOUT` and `TEMP` are empty; they document
  intent.

```cpp
struct FooTask : public clio::run::Task {
  IN    clio::run::u32 op_;
  INOUT clio::run::priv::string data_;

  FooTask() : clio::run::Task(), op_(0), data_(CLIO_PRIV_ALLOC) {}
  explicit FooTask(const clio::run::TaskId &id, const clio::run::PoolId &pool,
                   const clio::run::PoolQuery &q, const std::string &data,
                   clio::run::u32 op)
      : clio::run::Task(id, pool, q, Method::kFoo),
        op_(op), data_(CLIO_PRIV_ALLOC, data) {}

  template <typename Ar> CTP_CROSS_FUN void SerializeIn(Ar &ar) {
    Task::SerializeIn(ar); ar(op_, data_);
  }
  template <typename Ar> CTP_CROSS_FUN void SerializeOut(Ar &ar) {
    Task::SerializeOut(ar); ar(data_);
  }
  void Copy(const ctp::ipc::FullPtr<FooTask> &o) {
    Task::Copy(o.template Cast<clio::run::Task>());
    op_ = o->op_; data_ = o->data_;
  }
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &b) {
    Task::AggregateOut(b);                       // OUT/INOUT only; never Copy()
    data_ = b.template Cast<FooTask>()->data_;
  }
};
```

**Field types.**
- `clio::run::priv::string`, `priv::vector<T>` and `priv::unordered_map`,
  constructed with `CLIO_PRIV_ALLOC`.
- `std::string`/`std::vector` work in CPU-only tasks.
- `priv::fixed_string<N>` and plain PODs are required in tasks a GPU kernel
  touches.

**Bulk data.** Use the `bulk()` serialization path. Under SHM it ships a
`ShmPtr`; under TCP/IPC it inlines the bytes.

**Other task hooks.**
- `AggregateIn` matters only for ManyToOne/AllToOne collectives.
- **Return codes:** 0 means success. Set them with `task->SetReturnCode(n)`.

**`CreateParams`.**
- It needs `static constexpr const char *chimod_lib_name` and a cereal
  `serialize()`.
- It needs `void LoadConfig(const clio::run::PoolConfig&)` when the pool is
  created from `compose:`. Every compose key except
  `mod_name/pool_name/pool_query/pool_id` arrives there as YAML. See bdev's
  `LoadConfig` for a full example.

## 5. Runtime container

```cpp
class Runtime : public clio::run::Container {
 public:
  Client client_;
  clio::run::TaskResume Create(clio::run::shared_ptr<CreateTask> &task);
  clio::run::TaskResume Foo(clio::run::shared_ptr<FooTask> &task);
  clio::run::u64 GetWorkRemaining() const override;
  // + declarations of every override autogen defines (copy from MOD_NAME_runtime.h)
};

clio::run::TaskResume Runtime::Foo(clio::run::shared_ptr<FooTask> &task) {
  CLIO_TASK_BODY_BEGIN
  auto sub = other_client_.AsyncBar(clio::run::PoolQuery::Local(), ...);
  CLIO_CO_AWAIT(sub);                    // suspends this task, not the worker
  if (sub->GetReturnCode() != 0) { task->SetReturnCode(sub->GetReturnCode()); CLIO_CO_RETURN; }
  task->data_ = ...;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}
```

**Handler shape.** There is no `RunContext` parameter. The current task is
available from `clio::run::GetCurrentTask()`.

**Optional overrides.**
- `ScheduleTask`: per-task routing, for example hashing a key to a container.
  The CTE core's version is the model.
- `GetTaskStats`: cost features for the scheduler.
- `InlineOp`, `Restart`, `Recover`, `Migrate`, `Expand`.
- `RegisterViz`: a dashboard page under `viz/`.

**Periodic work.** Submit a task with `period_ns_` set from Create. The
interposers' sweeps are examples.

**Calling other pools.** Keep their `Client` as a member, set its `pool_id_`
in Create, and await its futures. The callee must be composed **before** your
pool.

**Locks.** `CoMutex` is reentrant per OS thread; take it with
`ScopedCoMutex lock(mutex_);`. `CoRwLock` has read and write variants.

**Affinity.** `task_group_ = TaskGroup(id)` keeps related tasks on one worker.

## 6. Client

```cpp
class Client : public clio::run::ContainerClient {
 public:
  clio::run::Future<CreateTask> AsyncCreate(const clio::run::PoolQuery &q,
                                            const std::string &pool_name,
                                            const clio::run::PoolId &custom_id) {
    auto *ipc = CLIO_IPC;
    auto t = ipc->NewTask<CreateTask>(clio::run::CreateTaskId(),
        clio::run::kAdminPoolId, q, CreateParams::chimod_lib_name,
        pool_name, custom_id, this);
    return ipc->Send(t);
  }
  clio::run::Future<FooTask> AsyncFoo(const clio::run::PoolQuery &q, ...) {
    auto *ipc = CLIO_IPC;
    auto t = ipc->NewTask<FooTask>(clio::run::CreateTaskId(), pool_id_, q, ...);
    return ipc->Send(t);
  }
};
```

**Futures.** `fut.Wait()` blocks a client thread; inside the runtime, await the
future instead. `fut->field_` reads the task. After Create completes, set
`client.pool_id_ = create_fut->new_pool_id_`.

**Pool queries.**
- `Local()`, `DirectHash(h)`, `DirectId(c)`, `Range`, `Broadcast()`.
- `Dynamic()` for Create: it creates the pool once and reuses it if it exists.

## 7. CMake

```cmake
add_clio_module_client(SOURCES src/my_mod_client.cc)
add_clio_module_runtime(SOURCES src/my_mod_runtime.cc src/autogen/my_mod_lib_exec.cc)
```

**Arguments.**
- The name and namespace come from `clio_mod.yaml`.
- Optional arguments: `LIB_NAME`, `COMPILE_DEFINITIONS`, `LINK_LIBRARIES`,
  `INCLUDE_DIRECTORIES`.

**Targets.**
- `<ns_with_underscores>_<mod>_{client,runtime}`, with the alias
  `<ns>::<mod>_{client,runtime}`.
- The runtime target gets `CLIO_RUNTIME=1` and links its client.
- Modules other than admin and bdev get those two linked automatically.

**Install and packaging.**
- Installation and the package config are automatic.
- There is no `add_clio_repo`: add one `add_subdirectory` per module.

**Linking.** Link CTP through the modular targets (`ctp::cxx`,
`ctp::configure`, `ctp::serialize`, ...), never yaml-cpp or cereal directly.

**External module repos.** Use `find_package(clio-core CONFIG REQUIRED)`,
then the same two functions.

## 8. Loading and compose

`ModuleManager::ScanForChiMods` searches these locations, in order:
1. the directory of `libclio_run_cxx`
2. `CLIO_REPO_PATH` (`:`-separated; `;` on Windows)
3. `LD_LIBRARY_PATH`
4. `./lib`, `../lib`, `/usr/local/lib`

It loads every `*_runtime.{so,dylib,dll}` that exports `alloc_chimod` and
`get_chimod_name`. Each module is keyed by `chimod_lib_name`, and the first one
found wins. A stale copy earlier on the path shadows your build.

```yaml
compose:
  - mod_name: clio_my_mod          # == CreateParams::chimod_lib_name
    pool_name: my_mod
    pool_query: local
    pool_id: "600.0"               # unique; check the id map in storage-compose
    my_key: 42                     # -> CreateParams::LoadConfig
```

Pools are created in list order.

## 9. Tests

The model test is `context-runtime/modules/MOD_NAME/test/`.

```cpp
#include <simple_test.h>
class Fixture {
 public:
  Fixture() {
    REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));
    SimpleTest::g_test_finalize = clio::run::CLIO_RUNTIME_FINALIZE;
  }
};
TEST_CASE("my_mod create and foo", "[my_mod]") {
  Fixture f;
  my_mod::Client c;
  auto cr = c.AsyncCreate(clio::run::PoolQuery::Dynamic(), "my_mod_test", my_mod::kPoolId);
  cr.Wait();
  REQUIRE(cr->GetReturnCode() == 0);
  c.pool_id_ = cr->new_pool_id_;
  ...
}
SIMPLE_TEST_MAIN()
```

**Framework.** Do not use Catch2 with runtime initialization (it segfaults).
`simple_test.h` has no `CHECK`; use `REQUIRE`.

**Registration** (see `MOD_NAME/test/CMakeLists.txt`):
- `add_test(... WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin)`.
- Set the environment to `CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin`,
  `CLIO_TEST_MODE=1`, `CLIO_BIND_ADDR=127.0.0.1`, and a unique `CLIO_PORT`.
- Set a `TIMEOUT`.
- Add a `clio_add_force_net_test` variant so that serialization is exercised.

## 10. GPU-visible tasks

GPU kernels are task **producers only**:
- The host pre-allocates POD tasks in a registered backend
  (`AllocateAndRegisterGpuBackend`).
- The kernel fills fields and calls `CLIO_IPC->Send(task)`.
- The CPU worker runs the handler. There are no GPU-side handlers, and
  `*_runtime_gpu` libraries no longer exist.

A task a kernel touches must be POD with an identical layout on host and
device:
- `CTP_CROSS_FUN` constructors and hooks
- `priv::fixed_string<N>`, not `priv::string`
- `CLIO_VIRTUAL` is always `virtual`

The models are `GpuSubmitTask` in `MOD_NAME_tasks.h` and the CTE's `PodPutBlob`
family. Inside a kernel, `NewTask`, `NewObj` and `AllocateBuffer` are
forbidden.
