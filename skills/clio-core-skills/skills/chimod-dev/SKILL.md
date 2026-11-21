---
name: chimod-dev
description: 'Use when writing your own CLIO Core runtime module (ChiMod) outside the clio-core repo: install with headers, clio_mod.yaml, tasks, runtime, client, codegen, CMake, loading, testing, or a CTE interposer layer. Triggers on "new chimod", "custom module", "add a method", "clio_mod.yaml", "clio-run refresh", "CreateParams", "CLIO_REPO_PATH", "CoreInterposer". Not for configuring shipped modules; use storage-compose. Not for GPU paged-vector kernels; use gpu-vector.'
metadata:
  bundle: clio-core
  servers: none
  provenance: designed
  eval-status: scenarios-recorded
---

# Developing a CLIO Core module (ChiMod)

A ChiMod is a shared library that the Context Runtime loads at startup. Each
instance of it is a **pool**, with one **container** per node when composed
`local`. Clients send **tasks**: structs that name a method id and carry IN
and OUT fields. The runtime routes each task to a container, which runs the
matching handler as a coroutine on a worker thread.

This skill is self-contained: you do not need a clio-core checkout. Facts were
checked against clio-core `dev` (bd1dc3183); source paths in parentheses are
provenance for maintainers only.

## Files in this skill

| file | read when |
|---|---|
| **TEMPLATE.md** | creating a module: every file of a compilable out-of-tree module (`clio_repo.yaml`, `clio_mod.yaml`, tasks, client, runtime, CMake, test, compose), the generated code, the build commands |
| **REFERENCE.md** | writing handler or client code: Task fields, PoolQuery kinds, Future, coroutine macros, yield, locks, return codes, bulk transfers, periodic tasks, Monitor, loading rules, CMake functions |
| **INTERPOSER.md** | the module sits in the CTE data path and speaks the core's Put/Get interface (cache, encoding, accounting, replication-like layers) |

## 1. Install CLIO Core with headers

The pip wheel (`pip install iowarp-core`) ships the libraries, the `clio-run`
CLI (including `clio-run refresh`, the code generator) and the Python
bindings, but **no C++ headers and no CMake package**. You cannot compile a
module against it. Use one of these, which run a full `cmake --install` and so
ship `include/clio_runtime/...`, `include/clio_ctp/...`, the admin and CTE
headers, and `lib/cmake/clio-core/clio-coreConfig.cmake`:

| route | command | notes |
|---|---|---|
| conda | `conda install -c iowarp -c conda-forge iowarp-core` | Linux, macOS. Use `$CONDA_PREFIX` as the prefix |
| Spack | `spack install iowarp` (package from the repo's `installers/spack` repo) | variants select optional parts |
| .deb / .rpm | release assets of github.com/iowarp/clio-core | installs to `/usr`; pulls `libzmq3-dev`, `libyaml-cpp-dev`, `libmsgpack-dev`, `libsodium-dev` |
| source | `cmake --preset=release -DCLIO_CTP_ENABLE_LIBPRESSIO=OFF && cmake --build build && cmake --install build --prefix <p>` | the only route on Windows (preset `windows-release`). Keep the LibPressio flag: without it a build without compression installs a broken package config (see Debugging) |

The `iowarp/deploy-cpu` Docker image is **not** enough: it copies
`/usr/local/lib`, `bin` and `share` but not `/usr/local/include`. Use
`iowarp/deps-cpu` (dependencies only) and a source install inside it.

**One install for both build and run.** A module records a fingerprint of the
runtime headers and coroutine/GPU options it was compiled against, and a
runtime with a different fingerprint refuses to load it. So run the module
with the `clio-run` from the same install whose headers you compiled against,
not with a pip `clio-run` beside a conda install.

## 2. Workflow

1. **Decide each method's contract first**: inputs, outputs, which container
   must run it (local, by key hash, one container, all), whether it awaits
   other pools, whether it is periodic.
2. **Copy TEMPLATE.md** and rename `myorg`/`counter`. Keep the directory name,
   `module_name` and header stem identical.
3. **Write `clio_mod.yaml`**: top-level `k<Name>: <id>` keys. Ids 0-9 are
   inherited (`kCreate: 0`, `kDestroy: 1`, `kMonitor: 9`, unused ones `-1`);
   yours start at 10. An interposer's own verbs start at 100.
4. **Generate**: `clio-run refresh <dir with clio_repo.yaml>`. It writes
   `include/<ns>/<mod>/autogen/<mod>_methods.h` and
   `src/autogen/<mod>_lib_exec.cc`. CMake never runs it: re-run after every
   `clio_mod.yaml` change, and commit the output. Then apply the one-line
   `sed` fix of TEMPLATE.md section 13 (step 1b): releases whose generator
   predates clio-core PR #1418 write four signatures that do not match the
   `Container` base. The fix is harmless on a fixed generator.
5. **Write the tasks header**: `CreateParams` (with `chimod_lib_name`,
   `serialize`, `LoadConfig`), `CreateTask`/`DestroyTask`/`MonitorTask`
   aliases, and one struct per method with a default constructor,
   `SerializeIn`, `SerializeOut`, `Copy` and `AggregateOut`.
6. **Write the runtime**: one handler per method returning
   `clio::run::TaskResume`, a `Client client_` member, `using CreateParams`,
   `GetWorkRemaining()`, the autogen override declarations, and
   `CLIO_TASK_CC(<ns>::<mod>::Runtime)` at the end of the .cc.
7. **Write the client**: `AsyncCreate` sent to `clio::run::kAdminPoolId`, one
   `Async<Name>` per method returning a `Future`. Define them in a .cc.
8. **CMake**: `find_package(clio-core CONFIG REQUIRED)`, then
   `add_clio_module_client` / `add_clio_module_runtime`, linking
   `clio::run::admin_client` / `clio::run::admin_runtime` yourself.
9. **Test** with a plain `main()` (TEMPLATE.md section 11): `CLIO_INIT(kClient,
   true)`, check Create's rc is 0 before anything else, then call your
   methods. Run it with `CLIO_REPO_PATH` = your library directory,
   `CLIO_TEST_MODE=1`, a private `CLIO_PORT`. Repeat with `CLIO_FORCE_NET=1`.
10. **Load it into a real runtime**: start `clio-run start` with
    `CLIO_REPO_PATH` set (or install the library next to the runtime library),
    then `clio-run compose start <file.yaml>`.
11. **After any task-struct change, rebuild everything** (client, runtime and
    every program that uses the client). The struct layout is shared by both
    sides; a half-rebuilt set reads fields at stale offsets and returns
    plausible garbage, not a crash.

## 3. Rules

- **Routing.** Never use `PoolQuery::Null()` (the task fails with rc -1001).
  Use `Local()` when unsure and `Dynamic()` for Create. Create goes to
  `kAdminPoolId`, takes an explicit non-null pool id and a caller-chosen pool
  name, and is get-or-create by that name.
- **Never hand-write what the generator writes** (`Init`, `Run`, `SaveTask`,
  `LoadTask`, `AllocLoadTask`, `Local*Task`, `NewTask`, `NewCopyTask`,
  `AggregateOut`/`AggregateIn` dispatch), except in an interposer, whose
  forwarding dispatch is hand-written and kept out of `clio_repo.yaml`.
- **`AggregateOut` touches OUT/INOUT fields only and never calls `Copy()`.**
  `Copy` overwrites the origin's identity and corrupts a live runtime.
- **Create reads params through `CreatePoolFields`**:
  `task.template Cast<clio::run::admin::CreatePoolFields>()->GetParamsAs<CreateParams>()`.
  Calling `task->GetParams()` is undefined for composed pools.
- **Handlers** wrap their body in `CLIO_TASK_BODY_BEGIN`/`END` and use
  `CLIO_CO_AWAIT`/`CLIO_CO_RETURN`, never raw `co_await`, so both coroutine
  back ends build.
- **Never block a worker**: no sleep, blocking syscall or spin. Await a
  future, or `CLIO_CO_AWAIT(clio::run::yield(us))` with `us > 0`. Put long
  blocking I/O on a thread pool you own.
- **Locks**: `CoMutex`/`CoRwLock` via their scoped guards; short sections; not
  held across an await; never log inside lock/unlock.
- **Other pools you call** are composed before yours and reached through
  their `Client`, stored as a member.
- **Test framework**: plain `main()` or your own; never Catch2 with runtime
  initialization (it segfaults). The runtime's `simple_test.h` is not installed.

## 4. Debugging a new module

| symptom | cause | fix |
|---|---|---|
| Create rc 22 (`EINVAL`), log `ChiMod '<name>' not found` | library not found, or `chimod_lib_name` != compose `mod_name` | put the dir on `CLIO_REPO_PATH` before the runtime starts; check the name; `CTP_LOG_LEVEL=debug` shows the scan |
| log `Skipping ChiMod ...: ABI fingerprint ... does not match` | built against headers of another version or build options | rebuild against the install whose `clio-run` you run |
| log `Skipping duplicate ChiMod` | an older copy earlier in the search path wins | remove it, or put your dir first |
| Windows: rc 22 although the DLL is on `CLIO_REPO_PATH` | a DLL it imports (for example your `*_client.dll`) is not on the DLL search path | add the dir to `PATH` too, or install next to `clio_run_cxx.dll` |
| CMake: `no declaration matches '...Runtime::SaveTask(...)'` (also `LoadTask`, `LocalLoadTask`, `LocalSaveTask`) | generator bug before clio-core PR #1418: `task_ptr` by value | the `sed` fix after every refresh (TEMPLATE.md section 13) |
| Windows CMake: `Could not find ... "yaml-cpp"` | the install has the dependency DLLs but not their CMake packages | add `<clio-core build>/vcpkg_installed/x64-windows` to `CMAKE_PREFIX_PATH` |
| Windows: test exits 127 at once | Debug build against a Release install, or `<prefix>/bin` not on `PATH` | build `--config Release`; add `<prefix>/bin` to `PATH` |
| CMake: `if given arguments: "AND" "NOT" "TARGET" "LibPressio::libpressio" ... Unknown arguments` in `iowarp-coreConfig.cmake` | the install was configured without compression and without `-DCLIO_CTP_ENABLE_LIBPRESSIO=OFF` | reinstall with that flag, or edit that line of the installed file to read `if(OFF AND NOT TARGET LibPressio::libpressio` |
| CMake: `Missing clio_mod.yaml` or wrong target names | file missing, or `namespace:`/`module_name:` appears in a comment first | remove those words from comments |
| CMake: `src/autogen/..._lib_exec.cc` not found | generator not run | `clio-run refresh <dir>` before configuring |
| handler never runs, task returns rc 0 at once | method missing from the generated dispatch | re-run `clio-run refresh` after editing `clio_mod.yaml` |
| link error: undefined handler or task method | struct/handler name does not match `k<Name>` | `kFoo` needs `FooTask` and `Runtime::Foo` |
| OUT field zero or garbage | not in `SerializeOut`, or half-rebuilt tree | add it; rebuild everything |
| right on one node, wrong across nodes | field missing from `SerializeIn` (the local path skips serialization) | add it; test with `CLIO_FORCE_NET=1` |
| broadcast result has one container's data | `AggregateOut` overwrites instead of merging | reduce (sum, max, concat) |
| test hangs or attaches to the wrong runtime | a runtime already listens on `CLIO_PORT` without your module | use a private `CLIO_PORT` and `CLIO_TEST_MODE=1` |
| hang inside a handler | worker blocked, a lock held across an await, or a pool awaited that was composed after yours | see Rules |

## 5. Contributing the module upstream to clio-core

Only for a module that will live inside the clio-core repository. There,
modules sit under a component directory (for example
`context-transfer-engine/<mod>/`) listed in that component's `clio_repo.yaml`
and added with `add_subdirectory`; the in-tree build links admin and bdev
automatically. Every C/C++ file gets the repository's BSD-3 license header and
doxygen docstrings, functions stay under 100 lines, and tests use the
in-tree `simple_test.h` registered with `CLIO_REPO_PATH` and a
`clio_add_force_net_test` variant. If the module has compose keys, add its
default entry (or the keys, commented out) to
`context-runtime/config/clio_default.yaml`, and document every key on its page
under `site/pages/modules/`. An out-of-tree module needs none of this.
