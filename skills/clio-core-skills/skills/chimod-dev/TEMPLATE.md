# TEMPLATE.md: a complete out-of-tree ChiMod

Every file of a working module, `myorg::counter`, built outside the clio-core
source tree against an installed CLIO Core. It has one custom method
(`Increment`) plus the inherited `Create`, `Destroy` and `Monitor`. Copy the
files, then rename `myorg` and `counter`.

Written against clio-core `dev` (bd1dc3183): the in-tree template module
(`context-runtime/modules/MOD_NAME`) and the CMake functions in
`cmake/ClioCoreCommon.cmake`. These files were built and run, as written,
against an installed CLIO Core at that commit on Linux (GCC 13) and Windows
(MSVC, `windows-release`): the ctest, a composed pool and a separate client
all passed. The out-of-tree example that lives in the repo
(`context-runtime/test/unit/external-chimod`) uses an old YAML schema, `void`
handlers and a synchronous `Create()`; do not copy it.

## 1. Layout

```
counter-mod/
├── CMakeLists.txt                  top level: find_package, bin/ output dir, test
├── counter_compose.yaml            compose entry for a running runtime
├── modules/
│   ├── clio_repo.yaml              lists the modules `clio-run refresh` generates
│   └── counter/                    directory name == module_name == header stem
│       ├── clio_mod.yaml           module name, namespace, method ids
│       ├── CMakeLists.txt
│       ├── include/myorg/counter/
│       │   ├── counter_tasks.h     CreateParams + task structs
│       │   ├── counter_client.h    Client : ContainerClient
│       │   ├── counter_runtime.h   Runtime : Container
│       │   └── autogen/counter_methods.h    GENERATED
│       └── src/
│           ├── counter_client.cc
│           ├── counter_runtime.cc  handlers + CLIO_TASK_CC
│           └── autogen/counter_lib_exec.cc  GENERATED
└── test/counter_test.cc            plain main(), no test framework
```

Naming chain (all four must agree, or something silently fails):

| thing | value here | rule |
|---|---|---|
| directory under `modules/` | `counter` | listed in `clio_repo.yaml`; the generator uses it as the C++ namespace leaf and file stem |
| `module_name` in `clio_mod.yaml` | `counter` | keep equal to the directory name |
| `namespace` in `clio_mod.yaml` | `myorg` | C++ namespace is `<namespace>::<directory>`, here `myorg::counter` |
| include prefix | `include/myorg/` | `namespace` with `::` replaced by `_` |
| library files | `libmyorg_counter_client.so`, `libmyorg_counter_runtime.so` (`myorg_counter_*.dll` on Windows) | `<namespace with _>_<module_name>_{client,runtime}` |
| CMake aliases | `myorg::counter_client`, `myorg::counter_runtime` | `<namespace>::<module_name>_{client,runtime}` |
| `CreateParams::chimod_lib_name` | `"myorg_counter"` | the `mod_name` in compose; any unique string, but use the library stem |

## 2. `modules/clio_repo.yaml`

```yaml
name: myorg-modules
namespace: myorg
version: 1.0.0
modules:
  - counter
```

## 3. `modules/counter/clio_mod.yaml`

```yaml
# Counter module: one counter per container.
module_name: counter
namespace: myorg
version: 1.0.0

kCreate: 0
kDestroy: 1
kNodeFailure: -1
kRecover: -1
kMigrate: -1
kUpgrade: -1
kMonitor: 9

kIncrement: 10
```

Do not write the words `namespace:` or `module_name:` inside a comment in this
file: CMake reads both keys with a regex that takes the first match, comments
included.

## 4. `modules/counter/include/myorg/counter/counter_tasks.h`

```cpp
/* BSD-3 license header here if you contribute upstream */
#ifndef MYORG_COUNTER_TASKS_H_
#define MYORG_COUNTER_TASKS_H_

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/admin/admin_tasks.h>
#include <yaml-cpp/yaml.h>

#include "autogen/counter_methods.h"

namespace myorg::counter {

/** Pool id this module is composed at by default. Pick one no other pool uses. */
static constexpr clio::run::PoolId kDefaultPoolId(700, 0);

/**
 * Parameters a pool is created with. Serialized with cereal when a client
 * calls AsyncCreate; filled by LoadConfig when the pool comes from compose.
 */
struct CreateParams {
  /** Name the runtime registers this module under (compose `mod_name`). */
  static constexpr const char *chimod_lib_name = "myorg_counter";

  clio::run::u64 initial_value_;  /**< counter value after Create */

  /**
   * @param initial_value counter value after Create
   */
  explicit CreateParams(clio::run::u64 initial_value = 0)
      : initial_value_(initial_value) {}

  /** cereal serialization (client-created pools). */
  template <class Archive>
  void serialize(Archive &ar) {
    ar(initial_value_);
  }

  /**
   * Read compose keys. config_ holds the whole compose entry as YAML,
   * including mod_name / pool_name / pool_query / pool_id.
   * @param pool_config the compose entry
   */
  void LoadConfig(const clio::run::PoolConfig &pool_config) {
    if (pool_config.config_.empty()) return;
    try {
      YAML::Node node = YAML::Load(pool_config.config_);
      if (node["initial_value"]) {
        initial_value_ = node["initial_value"].as<clio::run::u64>();
      }
    } catch (const std::exception &e) {
      HLOG(kError, "counter: bad compose entry: {}", e.what());
    }
  }
};

/** Create is the admin's get-or-create; Destroy and Monitor are the admin's. */
using CreateTask = clio::run::admin::GetOrCreatePoolTask<CreateParams>;
using DestroyTask = clio::run::admin::DestroyTask;
using MonitorTask = clio::run::admin::MonitorTask;

/** Add amount_ to the container's counter and return the new value. */
struct IncrementTask : public clio::run::Task {
  IN clio::run::u64 amount_;
  OUT clio::run::u64 value_;

  /** Default constructor: required by NewTask / deserialization. */
  IncrementTask() : clio::run::Task(), amount_(0), value_(0) {}

  /**
   * Emplace constructor used by the client.
   * @param task_id from clio::run::CreateTaskId()
   * @param pool_id the counter pool
   * @param pool_query where to run it
   * @param amount value to add
   */
  explicit IncrementTask(const clio::run::TaskId &task_id,
                         const clio::run::PoolId &pool_id,
                         const clio::run::PoolQuery &pool_query,
                         clio::run::u64 amount)
      : clio::run::Task(task_id, pool_id, pool_query, Method::kIncrement),
        amount_(amount),
        value_(0) {}

  /** Fields sent client -> runtime (IN and INOUT). Base first. */
  template <typename Archive>
  void SerializeIn(Archive &ar) {
    Task::SerializeIn(ar);
    ar(amount_);
  }

  /** Fields sent runtime -> client (OUT and INOUT). Base first. */
  template <typename Archive>
  void SerializeOut(Archive &ar) {
    Task::SerializeOut(ar);
    ar(value_);
  }

  /** Whole-task copy (used to make replicas). Base first. */
  void Copy(const ctp::ipc::FullPtr<IncrementTask> &other) {
    Task::Copy(other.template Cast<clio::run::Task>());
    amount_ = other->amount_;
    value_ = other->value_;
  }

  /**
   * Merge one replica's OUT fields into this origin task (Broadcast / Range).
   * OUT fields only; never call Copy(). Sum: one replica lands on its value.
   */
  void AggregateOut(const ctp::ipc::FullPtr<clio::run::Task> &other_base) {
    Task::AggregateOut(other_base);
    value_ += other_base.template Cast<IncrementTask>()->value_;
  }
};

}  // namespace myorg::counter

#endif  // MYORG_COUNTER_TASKS_H_
```

## 5. `modules/counter/include/myorg/counter/counter_client.h`

```cpp
#ifndef MYORG_COUNTER_CLIENT_H_
#define MYORG_COUNTER_CLIENT_H_

#include <string>

#include <clio_runtime/clio_runtime.h>

#include "counter_tasks.h"

namespace myorg::counter {

/** Client API: every call returns a Future; call Wait() on it. */
class Client : public clio::run::ContainerClient {
 public:
  Client() = default;

  /** @param pool_id pool this client talks to */
  explicit Client(const clio::run::PoolId &pool_id) { Init(pool_id); }

  /**
   * Get or create the pool. Sent to the admin pool, never to pool_id_.
   * @param pool_query use PoolQuery::Dynamic()
   * @param pool_name unique pool name chosen by the caller
   * @param pool_id explicit pool id (a null id is rejected with EINVAL)
   * @param initial_value counter value after Create
   */
  clio::run::Future<CreateTask> AsyncCreate(
      const clio::run::PoolQuery &pool_query, const std::string &pool_name,
      const clio::run::PoolId &pool_id, clio::run::u64 initial_value = 0);

  /**
   * Add to the counter.
   * @param pool_query Local(), DirectHash(h), Broadcast(), ...
   * @param amount value to add
   */
  clio::run::Future<IncrementTask> AsyncIncrement(
      const clio::run::PoolQuery &pool_query, clio::run::u64 amount);

  /**
   * Query container state; results_ maps container id -> msgpack bytes.
   * @param pool_query routing
   * @param query free-form query string
   */
  clio::run::Future<MonitorTask> AsyncMonitor(
      const clio::run::PoolQuery &pool_query, const std::string &query);
};

}  // namespace myorg::counter

#endif  // MYORG_COUNTER_CLIENT_H_
```

## 6. `modules/counter/src/counter_client.cc`

The client methods live in a .cc on purpose: on Windows a DLL with no exported
symbol produces no import library, and the runtime DLL, which links the client,
then fails to link.

```cpp
#include "myorg/counter/counter_client.h"

namespace myorg::counter {

clio::run::Future<CreateTask> Client::AsyncCreate(
    const clio::run::PoolQuery &pool_query, const std::string &pool_name,
    const clio::run::PoolId &pool_id, clio::run::u64 initial_value) {
  auto *ipc = CLIO_IPC;
  // Arguments after `this` are forwarded to the CreateParams constructor.
  auto task = ipc->NewTask<CreateTask>(
      clio::run::CreateTaskId(), clio::run::kAdminPoolId, pool_query,
      CreateParams::chimod_lib_name, pool_name, pool_id, this, initial_value);
  return ipc->Send(task);
}

clio::run::Future<IncrementTask> Client::AsyncIncrement(
    const clio::run::PoolQuery &pool_query, clio::run::u64 amount) {
  auto *ipc = CLIO_IPC;
  auto task = ipc->NewTask<IncrementTask>(clio::run::CreateTaskId(), pool_id_,
                                          pool_query, amount);
  return ipc->Send(task);
}

clio::run::Future<MonitorTask> Client::AsyncMonitor(
    const clio::run::PoolQuery &pool_query, const std::string &query) {
  auto *ipc = CLIO_IPC;
  auto task = ipc->NewTask<MonitorTask>(clio::run::CreateTaskId(), pool_id_,
                                        pool_query, query);
  return ipc->Send(task);
}

}  // namespace myorg::counter
```

## 7. `modules/counter/include/myorg/counter/counter_runtime.h`

```cpp
#ifndef MYORG_COUNTER_RUNTIME_H_
#define MYORG_COUNTER_RUNTIME_H_

#include <atomic>
#include <string>

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/container.h>

#include "autogen/counter_methods.h"
#include "counter_client.h"
#include "counter_tasks.h"

namespace myorg::counter {

/** Server side: one instance per container (one per node for `local`). */
class Runtime : public clio::run::Container {
 public:
  using CreateParams = myorg::counter::CreateParams;  // read by CLIO_TASK_CC

  Runtime() = default;
  ~Runtime() override = default;

  // ---- Handlers: one per k<Name> in clio_mod.yaml, named <Name> ----------
  clio::run::TaskResume Create(clio::run::shared_ptr<CreateTask> &task);
  clio::run::TaskResume Destroy(clio::run::shared_ptr<DestroyTask> &task);
  clio::run::TaskResume Monitor(clio::run::shared_ptr<MonitorTask> &task);
  clio::run::TaskResume Increment(clio::run::shared_ptr<IncrementTask> &task);

  /** Outstanding work; the runtime waits for 0 before unplugging a container. */
  clio::run::u64 GetWorkRemaining() const override;

  // ---- Defined by `clio-run refresh` in src/autogen/counter_lib_exec.cc --
  void Init(const clio::run::PoolId &pool_id, const std::string &pool_name,
            clio::run::u32 container_id = 0) override;
  clio::run::TaskResume Run(clio::run::u32 method,
                            clio::run::shared_ptr<clio::run::Task> task_ptr) override;
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
      clio::run::u32 method, clio::run::shared_ptr<clio::run::Task> &orig_task_ptr,
      bool deep) override;
  clio::run::shared_ptr<clio::run::Task> NewTask(clio::run::u32 method) override;
  void AggregateOut(clio::run::u32 method,
                    clio::run::shared_ptr<clio::run::Task> &orig_task,
                    const clio::run::shared_ptr<clio::run::Task> &replica_task) override;
  void AggregateIn(clio::run::u32 method,
                   clio::run::shared_ptr<clio::run::Task> &agg_task,
                   const clio::run::shared_ptr<clio::run::Task> &member_task) override;

 private:
  Client client_;  /**< assigned Client(pool_id) by the generated Init() */
  std::atomic<clio::run::u64> value_{0};  /**< this container's counter */
};

}  // namespace myorg::counter

#endif  // MYORG_COUNTER_RUNTIME_H_
```

## 8. `modules/counter/src/counter_runtime.cc`

```cpp
#include "myorg/counter/counter_runtime.h"

#include <clio_ctp/serialize/msgpack_wrapper.h>

namespace myorg::counter {

clio::run::TaskResume Runtime::Create(clio::run::shared_ptr<CreateTask> &task) {
  CLIO_TASK_BODY_BEGIN
  // Read params through CreatePoolFields: a composed pool arrives as a
  // different create-task instantiation, so never call task->GetParams().
  auto &fields = task.template Cast<clio::run::admin::CreatePoolFields>();
  CreateParams params = fields->GetParamsAs<CreateParams>();
  value_.store(params.initial_value_);
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
  pk.pack_map(1);
  pk.pack("value");
  pk.pack(static_cast<uint64_t>(value_.load()));
  task->results_[container_id_] = std::string(sbuf.data(), sbuf.size());
  task->SetReturnCode(0);
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::TaskResume Runtime::Increment(
    clio::run::shared_ptr<IncrementTask> &task) {
  CLIO_TASK_BODY_BEGIN
  const clio::run::u64 amount = task->amount_;
  task->value_ = value_.fetch_add(amount) + amount;
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}

clio::run::u64 Runtime::GetWorkRemaining() const { return 0; }

}  // namespace myorg::counter

// Exports alloc_chimod / new_chimod / get_chimod_name / destroy_chimod /
// get_chimod_abi. Outside every namespace, once per module.
CLIO_TASK_CC(myorg::counter::Runtime)
```

## 9. `modules/counter/CMakeLists.txt`

```cmake
# Name and namespace come from clio_mod.yaml in this directory.
# Client first: the runtime links its own client automatically if it exists.
add_clio_module_client(
  SOURCES src/counter_client.cc
  LINK_LIBRARIES clio::run::admin_client      # admin_tasks.h (Create/Destroy/Monitor)
)
add_clio_module_runtime(
  SOURCES
    src/counter_runtime.cc
    src/autogen/counter_lib_exec.cc
  LINK_LIBRARIES clio::run::admin_runtime
)
```

Out of tree, the functions do NOT add admin for you: they only auto-link
targets named `clio_admin_client` / `clio_admin_runtime`, which exist only
inside the clio-core build. Name the imported `clio::run::admin_*` targets.

## 10. Top-level `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.22)
project(counter_mod LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Every library and exe in one directory: one CLIO_REPO_PATH entry, and on
# Windows the test exe finds the client DLL next to itself. The empty
# generator expression $<0:> stops Visual Studio from adding a Debug/ or
# Release/ subdirectory.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<0:>)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<0:>)

# Provides ctp::*, clio::run::cxx, clio::run::admin_*, clio::run::bdev_*,
# clio::cte::core_* (if CTE was built) and add_clio_module_client/_runtime.
# `iowarp-core` is the legacy spelling of the same package.
find_package(clio-core CONFIG REQUIRED)

add_subdirectory(modules/counter)

add_executable(counter_test test/counter_test.cc)
target_link_libraries(counter_test PRIVATE myorg::counter_client)

enable_testing()
add_test(NAME counter_test COMMAND counter_test
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_tests_properties(counter_test PROPERTIES
  TIMEOUT 120
  ENVIRONMENT "CLIO_REPO_PATH=${CMAKE_BINARY_DIR}/bin;CLIO_TEST_MODE=1;CLIO_BIND_ADDR=127.0.0.1;CLIO_PORT=10611")
if(WIN32)
  # Windows has no RPATH: put the CLIO Core DLLs (<prefix>/bin) on PATH.
  set_tests_properties(counter_test PROPERTIES ENVIRONMENT_MODIFICATION
    "PATH=path_list_prepend:$<TARGET_FILE_DIR:clio::run::cxx>")
endif()
```

`CLIO_TEST_MODE=1` makes the in-process runtime ignore `~/.clio/clio.yaml`
(a hermetic test) and bind to loopback. `CLIO_PORT` keeps it off a runtime you
may already be running on the default port.

## 11. `test/counter_test.cc`

`simple_test.h`, the runtime's own test header, is not installed by any
package, so the test is a plain `main()`. Do not use Catch2 with runtime
initialization (it segfaults).

```cpp
#include <iostream>

#include <clio_runtime/clio_runtime.h>
#include <myorg/counter/counter_client.h>

/** Fail with a message and a non-zero exit code. */
static int Fail(const char *what, long long rc) {
  std::cerr << "FAIL: " << what << " (rc=" << rc << ")\n";
  return 1;
}

int main() {
  // kClient + true: become the runtime in-process if none answers on
  // CLIO_PORT, else attach to the one that does.
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true)) {
    return Fail("CLIO_INIT", -1);
  }

  myorg::counter::Client client;
  auto create = client.AsyncCreate(clio::run::PoolQuery::Dynamic(),
                                   "counter_test",
                                   myorg::counter::kDefaultPoolId,
                                   /*initial_value=*/5);
  create.Wait();
  if (create->GetReturnCode() != 0) {
    return Fail("Create", create->GetReturnCode());
  }
  client.pool_id_ = create->new_pool_id_;

  auto inc = client.AsyncIncrement(clio::run::PoolQuery::Local(), 3);
  inc.Wait();
  if (inc->GetReturnCode() != 0) return Fail("Increment", inc->GetReturnCode());
  if (inc->value_ != 8) return Fail("Increment value != 8", inc->value_);

  std::cout << "PASS: counter = " << inc->value_ << "\n";
  clio::run::CLIO_RUNTIME_FINALIZE();
  return 0;
}
```

To also exercise serialization (the path a remote node takes), run the same
test with `CLIO_FORCE_NET=1` added to its environment: a field missing from
`SerializeIn`/`SerializeOut` then shows up as a wrong value on one node.

## 12. `counter_compose.yaml`

```yaml
compose:
  - mod_name: myorg_counter        # == CreateParams::chimod_lib_name
    pool_name: counter
    pool_query: local              # one container per node
    pool_id: "700.0"               # == kDefaultPoolId
    initial_value: 100             # -> CreateParams::LoadConfig
```

## 13. Generate, build, test

```bash
# 1. Generate the dispatch code. Argument = the directory holding clio_repo.yaml.
#    Re-run after EVERY change to any clio_mod.yaml. CMake never runs it.
clio-run refresh modules            # same as: clio-run repo refresh modules

# 1b. Fix the generated file. Generators older than clio-core PR #1418 write
#     SaveTask, LoadTask, LocalLoadTask and LocalSaveTask with task_ptr BY
#     VALUE; the Container base takes it by reference, so the build fails with
#     "no declaration matches ...SaveTask(...)". Run after every refresh; it
#     changes nothing when the generator is already fixed:
sed -i 's/^\( *clio::run::shared_ptr<clio::run::Task>\) task_ptr) {/\1 \&task_ptr) {/' \
  modules/counter/src/autogen/counter_lib_exec.cc

# 2. Configure against the install that has headers (see SKILL.md, Install).
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=<clio-core prefix>
cmake --build build -j

# 3. Test (in-process runtime; finds the module through CLIO_REPO_PATH).
ctest --test-dir build --output-on-failure
```

Run step 1 before the first configure: `CMakeLists.txt` lists
`src/autogen/counter_lib_exec.cc`, which does not exist until then. On macOS
write `sed -i ''` instead of `sed -i`.

**Windows** (Git Bash; Visual Studio generator):

```bash
export PATH="/c/path/to/prefix/bin:$PATH"     # clio-run.exe and the CLIO DLLs
clio-run refresh modules
sed -i 's/^\( *clio::run::shared_ptr<clio::run::Task>\) task_ptr) {/\1 \&task_ptr) {/' \
  modules/counter/src/autogen/counter_lib_exec.cc
cmake -S . -B build \
  "-DCMAKE_PREFIX_PATH=<prefix>;<clio-core build>/vcpkg_installed/x64-windows"
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

- The install copies the dependency DLLs into `<prefix>/bin` but not their
  CMake packages, so `find_package(clio-core)` fails with
  `Could not find a package configuration file provided by "yaml-cpp"`. Add
  the vcpkg tree the install was built with to `CMAKE_PREFIX_PATH`.
- Build the same configuration as the install (`Release` for
  `windows-release`). A Debug module against a Release install mixes C
  runtimes and fails to start (exit 127, missing debug DLLs).
- In Git Bash, write PATH entries as `/c/...`: a `C:/...` entry is split at
  the colon and `clio-run` is not found.

## 14. What the generator writes

`clio-run refresh modules` prints `Generating files for ChiMod: counter` and
writes two files. Commit them; never edit them by hand beyond the step 1b
fix.

`include/myorg/counter/autogen/counter_methods.h`:

```cpp
#ifndef MYORG_COUNTER_AUTOGEN_METHODS_H_
#define MYORG_COUNTER_AUTOGEN_METHODS_H_

#include <clio_runtime/clio_runtime.h>
#include <string>
#include <vector>

namespace myorg::counter {

namespace Method {
// Inherited methods
GLOBAL_CROSS_CONST clio::run::u32 kCreate = 0;
GLOBAL_CROSS_CONST clio::run::u32 kDestroy = 1;
GLOBAL_CROSS_CONST clio::run::u32 kMonitor = 9;

// counter-specific methods
GLOBAL_CROSS_CONST clio::run::u32 kIncrement = 10;

GLOBAL_CROSS_CONST clio::run::u32 kMaxMethodId = 11;

inline const std::vector<std::string>& GetMethodNames() { /* names by id */ }
}  // namespace Method

}  // namespace myorg::counter
#endif
```

`src/autogen/counter_lib_exec.cc` (abridged; one `case` per method in every
function):

```cpp
#include "myorg/counter/counter_runtime.h"
#include "myorg/counter/autogen/counter_methods.h"
#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/task.h>

namespace myorg::counter {

void Runtime::Init(const clio::run::PoolId &pool_id, const std::string &pool_name,
                   clio::run::u32 container_id) {
  clio::run::Container::Init(pool_id, pool_name, container_id);
  client_ = Client(pool_id);
  DefineModel(Method::kMaxMethodId);
  SetMethodNames(Method::GetMethodNames());
}

clio::run::TaskResume Runtime::Run(clio::run::u32 method,
                                   clio::run::shared_ptr<clio::run::Task> task_ptr) {
  CLIO_TASK_BODY_BEGIN
  switch (method) {
    case Method::kIncrement: {
      auto& typed_task = task_ptr.template Cast<IncrementTask>();
      CLIO_CO_AWAIT(Increment(typed_task));
      break;
    }
    // ... kCreate -> Create(CreateTask), kDestroy, kMonitor
    default: break;   // unknown method: nothing runs
  }
  CLIO_CO_RETURN;
  CLIO_TASK_BODY_END
}
// SaveTask / LoadTask: archive << / >> *typed_task
// AllocLoadTask / LocalAllocLoadTask: NewTask(method) then load
// LocalLoadTask / LocalSaveTask, NewCopyTask (NewTask<T>() + Copy),
// NewTask (NewTask<T>()), AggregateOut, AggregateIn
}  // namespace myorg::counter
```

What the generated code requires from you, per method `kFoo`:

- a struct `FooTask` in namespace `<namespace>::<directory>`, except
  `kCreate` -> `CreateTask` and `kDestroy` -> `DestroyTask`;
- a default constructor, `SerializeIn`, `SerializeOut`, `Copy(FullPtr<FooTask>)`
  and `AggregateOut`. `AggregateIn` is inherited from `Task` (a no-op);
- a handler `clio::run::TaskResume Runtime::Foo(clio::run::shared_ptr<FooTask>&)`;
- a `Client client_` member constructible from a `PoolId`.

Key rules of the generator:

- Method keys are top-level keys that start with `k` and have an integer
  value. A `methods:` or `tasks:` list (old schema) is silently ignored.
- `-1` means "not implemented": no constant, no dispatch case.
- Ids below 10 are labeled inherited; yours start at 10.
- `kRestart: <n>` with n >= 0 emits a `Restart()` override that sets
  `is_restart_ = true` and calls `Init`; declare `bool is_restart_ = false;`
  and the `Restart` override in your runtime if you use it.
- It takes exactly one argument and regenerates every module in `modules:`.

## 15. Use it from a running runtime

```bash
# Linux/macOS. The runtime scans CLIO_REPO_PATH when it starts.
CLIO_REPO_PATH=$PWD/build/bin clio-run start &
clio-run compose start counter_compose.yaml
```

`clio-run start` first runs the default compose (`~/.clio/clio.yaml`), which
names the CTE and CAE. If your install was built without one of them, start
fails with `ChiMod 'clio_cae_core' not found`; start bare with
`clio-run start --ephemeral` and compose what you need. Stop the runtime with
`clio-run stop`.

A separate client program attaches without starting its own runtime:

```cpp
clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, false);
myorg::counter::Client c(myorg::counter::kDefaultPoolId);   // composed pool
auto f = c.AsyncIncrement(clio::run::PoolQuery::Local(), 1);
f.Wait();   // f->value_ == 101 on the first call: initial_value 100 + 1
```

Link it to `myorg::counter_client` after `find_package(clio-core)` and
`find_package(myorg_counter)` (from `cmake --install build --prefix <p>`).

Do not point `counter_test` at this runtime. It creates `counter_test` at
700.0, and a create whose pool id is already taken returns rc 0 and the
**existing** pool, even under another name. Its Increment then lands on the
composed counter and the check `value_ != 8` fails.

Or install the module next to the runtime, where it is found with no
environment at all: `cmake --install build --prefix <clio-core prefix>`. See
REFERENCE.md section 9 for the full search order and Windows notes.
