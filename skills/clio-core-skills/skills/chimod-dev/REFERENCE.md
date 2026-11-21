# REFERENCE.md: the ChiMod API as the installed headers define it

Every name below was checked against clio-core `dev` (bd1dc3183) headers under
`include/clio_runtime/` and `include/clio_runtime/admin/` of an install. All
types are in namespace `clio::run` unless shown otherwise.

## 1. Headers to include

| header | gives you |
|---|---|
| `<clio_runtime/clio_runtime.h>` | everything below except admin tasks: `Task`, `PoolQuery`, `Future`, `Container`, `ContainerClient`, `CLIO_IPC`, `CLIO_INIT`, `HLOG` |
| `<clio_runtime/admin/admin_tasks.h>` | `admin::GetOrCreatePoolTask<P>`, `admin::CreatePoolFields`, `admin::DestroyTask`, `admin::MonitorTask`, `PoolConfig` |
| `<clio_runtime/container.h>` | `Container`, `CLIO_TASK_CC`, `MonitorModeId` |
| `<clio_runtime/comutex.h>`, `<clio_runtime/corwlock.h>` | `CoMutex`, `ScopedCoMutex`, `CoRwLock`, `ScopedCoRwReadLock`, `ScopedCoRwWriteLock` |
| `<clio_ctp/serialize/msgpack_wrapper.h>` | msgpack, for `Monitor` replies |
| `<clio_cte/core/core_client.h>`, `<clio_cte/core/core_interposer.h>` | CTE client and interposer base (INTERPOSER.md) |

## 2. Types

| name | what |
|---|---|
| `u32`, `u64`, `i32`... | fixed-width integers |
| `PoolId` (= `UniqueId`) | `PoolId(major, minor)`, constexpr. `PoolId::FromString("700.0")`, `.ToString()`, `.IsNull()`, `PoolId::GetNull()` |
| `kAdminPoolId` | `PoolId(1, 0)`. Every create task is sent here |
| `TaskId` | from `CreateTaskId()`; pass it as a task's first argument |
| `ContainerId` | `u32`; on every pool the container id equals the node id |
| `priv::string`, `priv::vector<T>`, `priv::unordered_map<K,V>` | allocator-backed containers for task fields; construct with `CLIO_PRIV_ALLOC` |
| `priv::fixed_string<N>` | inline POD string; use in tasks a GPU kernel touches |
| `TaskResume` | return type of every handler (a coroutine or fiber handle) |
| time units | `kNano` 1, `kMicro` 1e3, `kMilli` 1e6, `kSec` 1e9 (nanoseconds) |

`std::string`, `std::vector` and `std::unordered_map` also work as task fields
in CPU-only tasks (`admin::MonitorTask` uses them).

## 3. The Task base class

Fields every task inherits (annotations `IN`/`OUT`/`INOUT`/`TEMP` are empty
macros that document direction):

| field | dir | meaning |
|---|---|---|
| `pool_id_` | IN | target pool |
| `task_id_` | IN | from `CreateTaskId()` |
| `pool_query_` | IN | routing (section 4) |
| `method_` | IN | method id from `Method::k...` |
| `task_flags_` | IN | bitfield: `TASK_PERIODIC`, `TASK_DATA_OWNER`, `TASK_REMOTE`, `TASK_FIRE_AND_FORGET` ... |
| `period_ns_` | IN | period of a periodic task |
| `task_group_` | IN | `TaskGroup(id)`: tasks with the same id run on one worker. `TaskGroup()` = none |
| `return_code_` | OUT | atomic; 0 = success |
| `completer_` | OUT | container id that ran it |

Methods: `GetReturnCode()`, `SetReturnCode(u32)`, `GetCompleter()`,
`SetPeriod(value, unit)`, `GetPeriod(unit)`, `IsPeriodic()`, `IsRemote()`,
`SetFlags(bits)`, `ClearFlags(bits)`, `IsComplete()`.

Constructor: `Task(task_id, pool_id, pool_query, method)` sets every base
field; a default `Task()` nulls them.

**Methods each task struct must have** (the generated dispatch calls them):

| member | rule |
|---|---|
| default constructor | used by `NewTask<T>()` before deserializing |
| `template <class Ar> void SerializeIn(Ar&)` | call `Task::SerializeIn(ar)` first, then `ar(in_and_inout_fields...)` |
| `template <class Ar> void SerializeOut(Ar&)` | call `Task::SerializeOut(ar)` first, then `ar(out_and_inout_fields...)` |
| `void Copy(const ctp::ipc::FullPtr<T>&)` | whole-task copy: `Task::Copy(o.template Cast<Task>())`, then every field |
| `void AggregateOut(const ctp::ipc::FullPtr<Task>&)` | merge one replica's result: `Task::AggregateOut(o)`, then **OUT/INOUT fields only, never `Copy()`**. Pick a reduction (sum, max, concat, first non-empty) such that a single replica lands on its own value |
| `void AggregateIn(const ctp::ipc::FullPtr<Task>&)` | optional; fold a batched member's IN fields into the aggregate (ManyToOne/AllToOne). Default no-op |

Mark a task's GPU-visible members `CTP_CROSS_FUN` only if a GPU kernel submits
it (section 13).

## 4. PoolQuery: where a task runs

| factory | runs on | notes |
|---|---|---|
| `PoolQuery::Local(u32 parallelism = 32)` | this node's container | the default; use when unsure |
| `PoolQuery::DirectId(ContainerId c, float net_timeout = -1)` | container `c` | |
| `PoolQuery::DirectHash(u32 h, float net_timeout = -1)` | container `h % num_containers` | stable key-to-node mapping |
| `PoolQuery::Range(u32 offset, u32 count, float net_timeout = -1)` | `count` containers from `offset` | replicas merged with `AggregateOut` |
| `PoolQuery::Broadcast(float net_timeout = -1)` | every container | replicas merged with `AggregateOut` |
| `PoolQuery::Physical(u32 node_id, float net_timeout = -1)` | the container on node `node_id` | |
| `PoolQuery::Dynamic(float net_timeout = -1)` | resolved by the pool's monitor | use for Create: reuses a pool that exists locally, otherwise broadcasts the create |
| `PoolQuery::ManyToOne(u32 container_hash, u64 batch_key = 0, u64 batch_for_ns = 10000)` | neighborhood leader | batches matching tasks for 10 us, folds them with `AggregateIn`, runs once, returns the result to all |
| `PoolQuery::AllToOne(u32 container_hash, u64 batch_key = 0)` | leader, after every container contributed | a barrier / all-reduce |
| `PoolQuery::Null()` | nowhere | the task completes with rc -1001. Never use it |

`net_timeout` is in seconds; -1 = runtime default. Compose files accept
`local`, `broadcast`, `dynamic`, `direct_id:N`, `direct_hash:N`,
`range:OFF:CNT`, `physical:N` (case-insensitive).

A container can rewrite routing per task by overriding
`PoolQuery ScheduleTask(const shared_ptr<Task>&)`; the default returns
`task->pool_query_`.

## 5. Client side: create, send, wait

```cpp
auto *ipc = CLIO_IPC;                                     // store the pointer
auto task = ipc->NewTask<FooTask>(CreateTaskId(), pool_id_, query, args...);
Future<FooTask> fut = ipc->Send(task);
fut.Wait();                     // bool Wait(float max_sec = -1, bool reuse_task = false)
u32 rc = fut->GetReturnCode();  // operator-> and get() reach the task
```

- `Wait(-1)` blocks forever, `Wait(0)` polls once, `Wait(s)` returns false
  after `s` seconds. The task is freed when the last `Future` goes away.
- Create: `NewTask<CreateTask>(CreateTaskId(), kAdminPoolId, query,
  CreateParams::chimod_lib_name, pool_name, pool_id, this, params_args...)`.
  The arguments after `this` construct `CreateParams`. After `Wait()`,
  `fut->new_pool_id_` is the pool id; also set `client.pool_id_` from it.
- Pool ids are explicit: a null id fails with `EINVAL`. Create is
  get-or-create by **pool name**: a second create with the same name returns
  the existing pool's id and does not run your `Create` again. A create
  whose **pool id** is already taken also returns rc 0 and that existing
  pool, even under another name (logged `already exists by ID`). Give every
  pool its own id.
- Data buffers: `ctp::ipc::FullPtr<char> b = ipc->AllocateBuffer(n);` pass
  `ctp::ipc::ShmPtr<>(b.shm_)` in the task; `ipc->FreeBuffer(b)` after `Wait`.
- Process setup: `bool CLIO_INIT(RuntimeMode mode, bool default_with_runtime =
  false, bool is_restart = false)`, then `CLIO_RUNTIME_FINALIZE()` before exit.
  `RuntimeMode::kClient` with `true` becomes the runtime in-process when no
  runtime answers on `CLIO_PORT`, and attaches otherwise. `CLIO_WITH_RUNTIME=0|1`
  overrides that second argument.

## 6. Runtime side: handlers and coroutines

```cpp
TaskResume Runtime::Foo(shared_ptr<FooTask> &task) {
  CLIO_TASK_BODY_BEGIN
  auto sub = other_client_.AsyncBar(PoolQuery::Local(), ...);
  CLIO_CO_AWAIT(sub);                                 // suspends this task only
  if (sub->GetReturnCode() != 0) {
    task->SetReturnCode(sub->GetReturnCode());
    CLIO_CO_RETURN;
  }
  CLIO_CO_AWAIT(clio::run::yield(25.0));              // cooperative yield, 25 us
  task->out_ = ...;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}
```

| macro / call | C++20 build | Boost-fiber build |
|---|---|---|
| `CLIO_TASK_BODY_BEGIN` / `_END` | empty | empty |
| `CLIO_CO_AWAIT(x)` | `co_await (x)` | suspend the task's fiber |
| `CLIO_CO_RETURN` | `co_return` | `return TaskResume{}` |

- Always use the macros, never raw `co_await`/`co_return`: the same source
  then builds under both coroutine back ends.
- `CLIO_CO_AWAIT` accepts a `Future`, a `yield(us)` and another handler's
  `TaskResume` (a helper coroutine).
- `yield(us)`: **pass a positive delay**. `yield()` with 0 waits for an event
  and may never resume.
- There is no `RunContext` parameter and no public accessor for it. The task
  being run is `clio::run::GetCurrentTask()` (a `shared_ptr<Task>&`);
  execution state is reached through the `Task` accessors.
- Never block a worker thread: no `sleep`, no blocking syscall, no spin. Long
  blocking I/O goes on a thread pool you own; the handler polls it with
  `yield(us)`.
- Calling another pool: keep its `Client` as a member, set it up in
  `Create`, and await its futures. That pool must be composed before yours.
  Calling your own pool: the generated `Init` gives you `client_`.

## 7. Locks

| type | acquire | notes |
|---|---|---|
| `CoMutex` | `ScopedCoMutex lock(m);` or `m.Lock()`/`m.TryLock()`/`m.Unlock()` | reentrant for the same task, and for a subtask on the same worker |
| `CoRwLock` | `ScopedCoRwReadLock r(l);` / `ScopedCoRwWriteLock w(l);` or `ReadLock()`/`ReadUnlock()`/`WriteLock()`/`WriteUnlock()` | not copyable; movable |

Both are fair ticket locks that wait by spinning with back-off on the
worker's OS thread; they do not suspend the coroutine. So keep critical
sections short, and do not hold one across `CLIO_CO_AWAIT`: another task on
the same worker that tries to take it spins on the thread the holder needs to
resume. Never log inside a lock or unlock path.

## 8. Return codes

| value | meaning |
|---|---|
| 0 | success |
| `EINVAL` (22 on Linux) | create rejected: module not loaded (no library with that `chimod_lib_name`), empty name, or null pool id |
| `ENODEV`, `ENOENT`, `EIO` | create failed further in (container could not be built or created) |
| -1000 (`kRun2RunNetworkTimeoutRC`, as u32 4294966296) | a remote node did not answer in time |
| -1001 (`kRouteNullPoolRc`, as u32 4294966295) | task sent to the null pool |
| anything else | your module's own codes; document them |

`return_code_` is `u32`: compare against the cast constant, not a negative int.

## 9. Loading: where the runtime finds modules

At startup the runtime scans these directories, in order, and loads every
file whose name ends in `_runtime.so` (`.dylib` on macOS, `.dll` on Windows;
`_runtime_gpu.*` is skipped):

1. the directory holding the runtime library itself (`libclio_run_cxx`):
   `<prefix>/lib` on Linux and macOS, `<prefix>/bin` on Windows, and
   `<site-packages>/iowarp_core/lib` for the pip wheel;
2. each entry of `CLIO_REPO_PATH` (`:`-separated; `;` on Windows);
3. each entry of `LD_LIBRARY_PATH` (`:`-separated);
4. `./lib`, `../lib`, `/usr/local/lib` (relative to the working directory).

A library loads only if it exports `alloc_chimod`, `new_chimod`,
`get_chimod_name`, `destroy_chimod` (all from `CLIO_TASK_CC`) **and** its
`get_chimod_abi()` matches the runtime's ABI fingerprint. The fingerprint is a
hash of the runtime's public headers plus the coroutine and GPU build options,
so a module built against one install loads only into a runtime of the same
version and options. A mismatch logs `Skipping ChiMod ...: ABI fingerprint
... does not match` and the create then fails with `EINVAL`.
`CLIO_CHIMOD_ABI_CHECK=0` disables the check (unsafe).

Modules are keyed by `chimod_lib_name`; **the first one found wins**. A stale
copy in an earlier directory shadows your build, logged as `Skipping
duplicate ChiMod`. Success is logged as `Loaded ChiMod: <name> from <path>`.
Raise the log level with `CTP_LOG_LEVEL=debug` (or `info`).

Scanning happens when the runtime starts. Start the runtime with the
environment that finds your library; then compose the pool with
`clio-run compose start <file.yaml>` or let a client call `AsyncCreate`.

**Windows.** The runtime loads a module with `LoadLibraryA(<full path>)`,
which does not search the module's own directory for its dependencies. The
CLIO Core DLLs are found because they sit next to `clio-run.exe`. Your
`*_runtime.dll` imports your `*_client.dll` only if it calls a function
defined in the client .cc (the linker drops unused imports; TEMPLATE.md's
counter does not, and loads from `CLIO_REPO_PATH` alone). If yours does, or
it needs any other DLL of yours, install them next to `clio_run_cxx.dll`
(`<prefix>/bin`), or put their directory on `PATH` as well as on
`CLIO_REPO_PATH`. A DLL that fails to load is logged only at debug level;
the visible symptom is Create rc `EINVAL` and `ChiMod '<name>' not found`.
Your own programs (tests, clients) need `<prefix>/bin` on `PATH`: Windows has
no RPATH.

## 10. Compose entry

```yaml
compose:
  - mod_name: myorg_counter     # CreateParams::chimod_lib_name
    pool_name: counter          # unique; create is get-or-create by this name
    pool_query: local           # local = one container per node
    pool_id: "700.0"            # unique "major.minor"
    restart: false              # true: re-created by the next `clio-run start`
    container_visibility: public          # or private: reject external clients
    container_rpc_acl: {Increment: private}   # per-method override, by name
    initial_value: 100          # any other key -> your LoadConfig
```

`LoadConfig(const PoolConfig&)` receives the **whole** entry as YAML text in
`PoolConfig::config_` (`mod_name_`, `pool_name_`, `pool_id_`, `pool_query_`
are also parsed for you). Pools are created in list order, so a pool must come
after every pool it names. Ids in use by shipped modules: 1.0 (admin), 301.0
to 350.0 (bdevs), 400.0 to 401.0 (CAE), 512.0 to 568.0 (CTE and its layers).
Use 700.0 and up for your own.

## 11. Optional Container overrides

| override | when |
|---|---|
| `TaskStat GetTaskStats(const Task*) const` | cost hints for the scheduler: `TaskStat{io_size_, compute_ (us), wall_time_}` |
| `PoolQuery ScheduleTask(const shared_ptr<Task>&)` | per-task routing, e.g. hash a key to a container |
| `u64 GetWorkRemaining() const` | **required**. The runtime waits for 0 before it unplugs a container (destroy, migration). Return 0 if you track nothing |
| `void Restart(pool_id, pool_name, container_id)` | warm restart on the same node; default calls `Init` |
| `u32 ScheduleRecover()`, `Recover(...)`, `Expand(host)`, `Migrate(node)` | failure handling, elastic membership |
| `bool InlineOp(u32 method, void *in, void *out)` | a fast path that runs without a task |

Members you can read: `pool_id_`, `pool_name_`, `container_id_`.

## 12. Serialization, bulk data, periodic tasks, Monitor

**Archive calls.** Inside `SerializeIn/Out`, `ar(a, b, c)` serializes fields
(PODs, `priv::*`, `std::` containers, anything with a cereal `serialize`).
Large payloads go by **bulk transfer**, which ships a shared-memory pointer
when client and runtime share memory, and inlines the bytes over TCP/IPC:

```cpp
// input buffer (client -> runtime), e.g. a put:
template <class Ar> void SerializeIn(Ar &ar) {
  Task::SerializeIn(ar); ar(size_, data_);
  ar.bulk(data_, size_, BULK_XFER);
}
// output buffer (runtime -> client), e.g. a get:
template <class Ar> void SerializeIn(Ar &ar) {
  Task::SerializeIn(ar); ar(size_, data_);
  ar.bulk(data_, size_, BULK_EXPOSE);        // caller's buffer, exposed
}
template <class Ar> void SerializeOut(Ar &ar) {
  Task::SerializeOut(ar);
  ar.bulk(data_, size_, BULK_XFER);          // filled bytes come back
}
```

`data_` is a `ctp::ipc::ShmPtr<>` from `AllocateBuffer`. The local path skips
serialization entirely, so a missing field shows up only across nodes or with
`CLIO_FORCE_NET=1`.

**Periodic task.** Submit from `Create`; it re-runs every period until the
pool is destroyed:

```cpp
auto *ipc = CLIO_IPC;
auto t = ipc->NewTask<SweepTask>(CreateTaskId(), pool_id_, PoolQuery::Local());
t->SetPeriod(500.0, kMilli);
t->SetFlags(TASK_PERIODIC);   // SetPeriod alone does not make it periodic
ipc->Send(t);                 // do not Wait on it
```

**Monitor.** Every module implements `kMonitor: 9` with `admin::MonitorTask`:
`IN std::string query_`, `OUT std::unordered_map<ContainerId, std::string>
results_`. Write `results_[container_id_]` as msgpack bytes; its
`AggregateOut` merges the per-container maps, so `Broadcast()` collects all
nodes. The query string is yours to define (`"stats"` is the convention).
`MonitorModeId { kLocalSchedule = 0, kGlobalSchedule = 1, kEstLoad = 2 }` is
the runtime's internal scheduling mode; `Dynamic()` routes through
`kGlobalSchedule`. You do not handle it.

## 13. GPU-submitted tasks

GPU kernels only submit tasks; handlers always run on CPU workers. The host
allocates POD tasks in a registered backend (`IpcManager::
AllocateAndRegisterGpuBackend`), the kernel fills fields, calls
`CLIO_IPC->Send(task)` and waits. A task a kernel touches must have the same
layout on host and device: `CTP_CROSS_FUN` constructors and hooks,
`priv::fixed_string<N>` instead of `priv::string`, no `std::` containers.
`NewTask`, `NewObj` and `AllocateBuffer` are host-only. Use the
`gpu-vector` skill for kernels.

## 14. CMake functions (from `find_package(clio-core)`)

`add_clio_module_client(...)` and `add_clio_module_runtime(...)` take:

| argument | kind | meaning |
|---|---|---|
| `SOURCES` | list | source files |
| `LIB_NAME` | one | override the `<pkg>_<module>` prefix of target and file names |
| `LINK_LIBRARIES` | list | extra link targets (PUBLIC) |
| `INCLUDE_DIRECTORIES` | list | extra include dirs (build tree only) |
| `COMPILE_DEFINITIONS` | list | extra definitions |
| `LINK_DIRECTORIES` | list | extra link dirs |

Both read `module_name` and `namespace` from `clio_mod.yaml` in the current
source directory (fatal error if missing), build a SHARED library with C++20,
add `include/` as a PUBLIC include dir, link `clio::run::cxx`, set
`WINDOWS_EXPORT_ALL_SYMBOLS`, and install the library, headers and a
`<pkg>_<module>Config.cmake` package. The runtime target also gets
`CLIO_RUNTIME=1`, `Threads::Threads`, `rt` (Linux) and its own client.
Out of tree they do **not** add admin or bdev: link `clio::run::admin_client`
/ `clio::run::admin_runtime` yourself.

Imported targets the package provides: `clio::run::cxx`, `ctp::cxx`,
`ctp::configure`, `ctp::serialize`, other `ctp::*`, `clio::run::admin_client`,
`clio::run::admin_runtime`, `clio::run::bdev_client`,
`clio::run::bdev_runtime`, and, if built, `clio::cte::core_client`,
`clio::cte::core_runtime`, `clio::cae::core_client`, `clio::cae::core_runtime`.
Link `ctp::*` targets, never yaml-cpp or cereal directly.
