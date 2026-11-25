# Evals - chimod-dev

Scenarios recorded 2026-11-21 against clio-core `dev` (bd1dc3183). Rubric is
pass/fail per bullet. Run each WITHOUT the skill to capture the gap, then WITH
it to confirm the gap closes.

## S1 - new module outside the repo

Setup: Prompt: "I installed CLIO Core with pip. Create a ChiMod `counter` in my
own project with an Increment method that adds to a per-container counter and
returns the new value, plus a test." No clio-core checkout is available.

Expected:

- Says the pip wheel has no headers or CMake package and names an install
  that does (conda iowarp channel, Spack, .deb/.rpm, or a source install);
  warns that the runtime that loads the module must come from that install.
- Produces an out-of-tree layout with `clio_repo.yaml`, `clio_mod.yaml`
  (top-level `kIncrement: 10` plus kCreate/kDestroy/kMonitor) and
  `find_package(clio-core CONFIG REQUIRED)`; never says "copy MOD_NAME from
  the repo".
- Runs `clio-run refresh <dir with clio_repo.yaml>` before configuring and
  does not hand-write `Run`/`SaveTask`/`NewTask`.
- Links `clio::run::admin_client` / `clio::run::admin_runtime` explicitly.
- Task has a default constructor, SerializeIn/SerializeOut/Copy/AggregateOut;
  AggregateOut touches OUT fields only.
- Client Create goes to `kAdminPoolId` with `PoolQuery::Dynamic()` and an
  explicit pool id; Create reads params through `CreatePoolFields::GetParamsAs`.
- Test is a plain `main()` (no `simple_test.h`, no Catch2), checks Create rc
  == 0, and sets `CLIO_REPO_PATH`, `CLIO_TEST_MODE=1` and a private `CLIO_PORT`.

## S2 - module will not load

Setup: Prompt: "My compose entry for `myorg_counter` fails with return code 22
and the log says ChiMod not found. The .so is in ./build/bin."

Expected:

- Explains the scan order (runtime library dir, `CLIO_REPO_PATH`,
  `LD_LIBRARY_PATH`, ./lib, ../lib, /usr/local/lib) and that the scan happens
  at runtime start.
- Checks that the file ends in `_runtime.so` and that `chimod_lib_name`
  equals `mod_name`.
- Mentions the ABI fingerprint mismatch and the first-found-wins duplicate
  rule as other causes, with their log lines.

## S3 - works on one node, wrong on two

Setup: Prompt: "My new task's output field is right in the single-node test but
zero across nodes."

Expected:

- Points at a field missing from `SerializeIn`/`SerializeOut` (local path
  skips serialization).
- Suggests re-running the test with `CLIO_FORCE_NET=1`.
- Mentions a half-rebuilt client/runtime pair as the other cause.

## S4 - data-path layer

Setup: Prompt: "Write a module that counts bytes written and read through the
CTE, transparently, without touching the core."

Expected:

- Derives from `clio::cte::core::CoreInterposer`; interposed ids equal the
  core's (`kPutBlob` 15, `kGetBlob` 16); own verbs >= 100.
- Hand-writes the dispatch with `Forward*` defaults and keeps the module out
  of `clio_repo.yaml`; does not override `ScheduleTask` for core verbs.
- Sets `interposer_next_pool_` from `next_pool_id` in Create; composes with
  `pool_query: local` after the next pool; clients use a
  `clio::cte::core::Client` on the interposer's pool id.
- Notes vectored puts / `kMultiPutBlob` / `kPod*` verbs bypass a Put/Get-only
  layer.

## Baseline failure modes to watch for (RED)

- Telling the user to copy files from `context-runtime/modules/MOD_NAME` or to
  edit `clio_default.yaml` in a repo they do not have.
- Compiling against the pip wheel.
- Calling an initializer, accessor or `PoolQuery` factory the headers do not
  define (for example `CLIO_RUNTIME_INIT`, `ToLocalCpu`, `CreateTaskNode`).
- Blocking a worker (sleep, blocking I/O) inside a handler.
