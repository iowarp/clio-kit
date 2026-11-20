# Evals - chimod-dev

Scenarios recorded 2026-11-20 against clio-core `dev` (86efd1a7e). Rubric is
pass/fail per bullet. Run each WITHOUT the skill to capture the gap, then WITH
it to confirm the gap closes.

## S1 - new module from scratch

Setup: Prompt: "Create a chimod `counter` with an Increment method that adds to
a per-container counter and returns the new value, plus a test."

Expected:

- Starts from `context-runtime/modules/MOD_NAME`, not external-chimod.
- `clio_mod.yaml` uses top-level `kIncrement: 10` keys with kCreate/kDestroy/kMonitor.
- Runs `clio-run refresh <repo_dir>` and does not hand-write `Run`/`SaveTask`.
- Task has SerializeIn/SerializeOut/Copy/AggregateOut; AggregateOut touches
  OUT fields only.
- Client Create sends to `kAdminPoolId` with `PoolQuery::Dynamic()`.
- Handler returns `TaskResume` with `CLIO_TASK_BODY_BEGIN/END`.
- Test uses `simple_test.h` and `clio::run::CLIO_INIT`, asserts Create rc == 0,
  sets `CLIO_REPO_PATH`.

## S2 - add a method to an existing module

Setup: Prompt: "Add a Reset method to the counter module."

Expected:

- Edits `clio_mod.yaml` and re-runs codegen.
- Rebuilds the whole tree (no `--target`) because a task struct changed.

## S3 - works on one node, wrong on two

Setup: Prompt: "My new task's field is right in the single-node test but zero
across nodes."

Expected:

- Points at a field missing from `SerializeIn`/`SerializeOut`.
- Suggests a `clio_add_force_net_test` variant.

## S4 - data-path layer

Setup: Prompt: "Write a module that encrypts blobs on Put and decrypts on Get,
transparently under the filesystem."

Expected:

- Derives from `CoreInterposer`; interposed ids equal the core's; own verbs >= 100.
- Does not override `ScheduleTask` for core verbs; forwards unknown methods
  with `Forward*`.
- Places it in the chain (below the indexer) and composes it after its next pool.
- States its guarantees and tests writer/reader on two runtimes.

## Baseline failure modes to watch for (RED)

- Calling an initializer, accessor or `PoolQuery` factory that the headers do not
  define (check `clio_runtime.h`, `task.h`, `pool_query.h`).
- Blocking a worker (sleep, blocking I/O) inside a handler.
- Using Catch2 with runtime initialization.
