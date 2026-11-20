---
name: chimod-dev
description: 'Use when creating or changing a CLIO Core runtime module (chimod): clio_mod.yaml methods, task structs, the runtime container and client, codegen, CMake, compose registration, tests, or a new CTE interposer layer. Triggers on "new chimod", "add a method to the module", "clio_mod.yaml", "clio-run refresh", "CreateParams", "CoreInterposer", "TaskResume". Not for choosing or configuring the shipped modules; use storage-compose. Not for GPU paged-vector kernels; use gpu-vector.'
metadata:
  bundle: clio-core
  servers: none
  provenance: designed
  eval-status: scenarios-recorded
---

# Developing a CLIO Core module (chimod)

A chimod is a shared library loaded by the Context Runtime. Each instance is a
**pool**, with one **container** per node. Clients submit **tasks**, which are
serialized structs naming a method id. The runtime routes each task to a
container, and the container runs the matching handler as a coroutine.

Source paths are relative to the root of the
[iowarp/clio-core](https://github.com/iowarp/clio-core) repository. Facts here
were checked against `dev` at 86efd1a7e (2026-11-20).

## Files in this skill

| file | read when |
|---|---|
| **REFERENCE.md** | writing any file of the module: layout, YAML schema, codegen, task anatomy, runtime, client, CMake, loading, tests, GPU tasks |
| **INTERPOSER.md** | the module sits in the CTE data path (cache, encoding, replication-like layers that speak the core's Put/Get interface) |

## Copy a working module, not the docs

Start from `context-runtime/modules/MOD_NAME/`, the template and test module.
It compiles, it is regenerated with the current tools, and it exercises
every feature: coroutines, subtasks, CoMutex, collectives and large outputs.

Do not start from `context-runtime/test/unit/external-chimod/`. No CMake builds
it, and it does not compile against the current headers. For a data-path layer, copy `replication`, which is
the reference interposer.

## Workflow

1. **Name the methods and their routing first.** For each method, decide:
   - its inputs and outputs
   - which container must run it: local, by hash of a key, a specific
     container, or broadcast
   - whether it waits on other pools
   - whether it is periodic
2. **Write `clio_mod.yaml`** with ids. Ids 0-9 are inherited (`kCreate: 0`,
   `kDestroy: 1`, `kMonitor: 9`). Your methods start at 10. An interposer's own
   verbs start at **100**.
3. **Run codegen:** `clio-run refresh <repo_dir>`, where `<repo_dir>` holds
   `clio_repo.yaml` listing your module. Commit the generated
   `autogen/*_methods.h` and `src/autogen/*_lib_exec.cc`. CMake never runs the
   generator, so re-run it after **every** change to `clio_mod.yaml`.
4. **Write the tasks header.** Write `CreateParams` with `chimod_lib_name` and
   `LoadConfig` if the pool is composed from YAML. Then write one task struct
   per method, with `SerializeIn`, `SerializeOut`, `Copy` and `AggregateOut`.
5. **Write the runtime.** It needs a handler per method, a `Client client_`
   member, `GetWorkRemaining()`, and `CLIO_TASK_CC(<ns>::Runtime)` at the end
   of the .cc file.
6. **Write the client.** It needs `AsyncCreate` (sent to `kAdminPoolId`) and one
   `AsyncX` per method, each returning a `Future`.
7. **Write the CMake.** Call `add_clio_module_client` and
   `add_clio_module_runtime`, and add the directory to the repo's
   `CMakeLists.txt`.
8. **Write the tests.** Use `simple_test.h` and `clio::run::CLIO_INIT(kClient,
   true)`. Assert that Create's return code is 0 before anything else. Register
   the test with `CLIO_REPO_PATH` set so that the runtime finds your library.
9. **If the module has config, document it.** Add the default compose entry, or
   at least commented-out keys, to `context-runtime/config/clio_default.yaml`.
   Then update the module's compose-key table under `site/pages/modules/`.
10. **Build everything, with no `--target`, after any task-struct change.**
    Task layout is an ABI shared by client and runtime: a half-rebuilt tree
    reads fields at stale offsets and returns plausible garbage, not a crash.

## Non-negotiable rules

- **Pool queries.**
  - Never pass a null `PoolQuery`. Use `Local()` when unsure, and `Dynamic()`
    for Create.
  - Send `CreateTask` to `clio::run::kAdminPoolId`, not to your pool, because
    Create is the admin's GetOrCreatePool.
  - Pool names are always supplied by the caller. Never derive one from
    `pool_id_`, which is not yet set.
  - The admin pool is named `admin`.
- **Never hand-write what autogen defines:** `Run`, `SaveTask`/`LoadTask` and
  friends, `NewTask`, `NewCopyTask`, `AggregateOut`/`AggregateIn` dispatch.
- **`AggregateOut` touches OUT/INOUT fields only and never calls `Copy()`.**
  This is enforced by `test_aggregate_out_contract`.
- **Handler shape.** Handlers return `clio::run::TaskResume`, with bodies
  wrapped in `CLIO_TASK_BODY_BEGIN`/`END`. Use `CLIO_CO_AWAIT` and
  `CLIO_CO_RETURN`, never raw `co_await`, so that the Boost-fiber build also
  works.
- **Blocking.**
  - Never block a worker with sleep, a blocking syscall or a spin.
  - Wait with `CLIO_CO_AWAIT(future)`, or `task->Yield()`.
  - Long or blocking I/O goes on a dedicated thread pool, as the CAE reference
    reader does.
- **Locks.** Use `CoMutex` or `CoRwLock` through `ScopedCoMutex`. Never log
  inside lock or unlock paths.
- **Naming.** Name queue ids and priorities as constants, never raw integers.
  Timing prints are in ms.
- **Dependencies.**
  - OS headers and OS calls belong in `context-transport-primitives`. Use the
    `ctp::SystemInfo` wrappers.
  - Store `GetInstance()` in a local pointer before using it.
  - Never use `thread_local`.
- **Style.**
  - Every new C/C++ file gets the BSD-3 license header.
  - Every function gets a doxygen docstring and stays under 100 lines.
  - No stubs or mock code.
- **GPU compiler macros.** Use only `CTP_IS_GPU`, `CTP_IS_HOST` and
  `CTP_IS_GPU_COMPILER`, never `__CUDACC__` and friends.

## Debugging a new module

| symptom | cause |
|---|---|
| Create rc ≠ 0, "module not found" | library not on `CLIO_REPO_PATH` or next to `libclio_run_cxx`; or `chimod_lib_name` ≠ compose `mod_name` |
| handler never runs, task returns at once | method id missing from the regenerated `lib_exec.cc` (codegen not re-run) |
| output fields come back zero or garbage | field not in `SerializeOut`, or a stale half-rebuilt tree |
| broadcast result has only one container's data | `AggregateOut` overwrites instead of merging |
| works locally, wrong across nodes | a field missing from `SerializeIn`. The local path skips serialization; force the network path with `clio_add_force_net_test` |
| hang | a handler blocked the worker, or awaited a pool composed after it |
