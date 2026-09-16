# Where Clio's tests live

Routine contracts and three process smoke files protect core product boundaries.
The installed-package and real native timing checks belong to qualification.
Extended regressions are explicit development investigations.

## Layout

| Lane | Path | Command | Build needed |
|---|---|---|---|
| Required contracts | `tests/contracts/*.test.ts` | `npm run test:file -- <file>` | CLI tests require current dist |
| Required process smoke | ACP, binary boot, process lifecycle under `tests/smoke/` | `npm run test` includes them | Yes |
| Package qualification | Installed package and native call timing under `tests/smoke/` | `npm run ci:release` | Built once by qualification |
| Extended regressions | `tests/extended/`, `tests/extended-smoke/` | `npm run test:full` or a focused `test:file` | Yes for CLI scenarios |
| Static boundaries and pins | `scripts/check-hygiene.ts` | `npm run lint` | No |
| Required web boundaries | Selected files in `apps/clio-coder-web/tests/` | `npm run test:web` | Root build |
| Full web investigation | All web tests and browser matrix | `pnpm --filter @iowarp/clio-coder-web verify` | Yes |

`npm run ci` runs the routine lane. `npm run ci:release` qualifies a clean,
committed candidate and its exact tarball; `npm run release:preflight` checks
that qualification without repeating development tests. See CONTRIBUTING.md.

## Contract files

| Area | Required or extended focused files |
|---|---|
| Authentication | `auth-login-write-failure`, `auth-storage-durability` |
| Context, session, and state | `context-lifecycle`, `memory-scope`, `project-bootstrap`, `session-durability`, `state-file-lock`, `task-board-done`, `working-set-core` |
| Config, routing, and presentation | `footer-context-window`, `knob-aliases`, `pane-remedies`, `rendering-invariants`, `route-identity-keying`, `settings-migration` |
| Dispatch, fleet, and workers | `dispatch-admission`, `dispatch-lifecycle`, `dispatch-schema`, `fleet-lifecycle`, `host-verification-batch`, `intent-requirements`, `worker-attestation-surface`, `worker-boundary` |
| Prompts, engine loop, and middleware | `compact-prompt-contracts`, `engine-lifecycle`, `loop-detector`, `loop-guard-epoch`, `middleware-hooks`, `prompt-cache-correctness`, `prompt-prefix-layout`, `prompt-role-routing`, `prompt-session-snapshot`, `prompt-tool-hints` |
| Providers and model policy | `gemma-channel-filter`, `llamacpp-router-probe`, `local-model-family-resolution`, `provider-context-boundary`, `provider-transport`, `synthesis-lock`, `thinking-off-wire` |
| Safety and tools | `bash-exec-settlement`, `rejection-feedback`, `safe-resource-write`, `safety-gates`, `tool-boundaries` |
| Evidence, eval, and release | `eval-boundary`, `evidence-integrity`, `metering-integrity`, `release-boundary` |
| Extensions, interop, and skills | `extension-compatibility`, `extension-reload-coordinator`, `extension-reload-slash`, `extension-resources`, `extension-snapshot`, `interop-boundary`, `marketplace-offer`, `skill-install` |
| Documentation navigation | `docs-server` |

Append `.test.ts` to every stem in the table. Use `rg` over the files before
choosing a lane; related behavior can span more than one focused contract.

## Smoke files

| Boundary | File under `tests/smoke/` |
|---|---|
| ACP v1 over JSON-RPC stdio, permission requests, and text/image content | `acp-boundary.test.ts` |
| Core CLI health, local-provider run, receipts, events, and autonomy | `cli-core.test.ts` |
| `npm pack`, installed resources, and installed codewiki navigation | `installed-package.test.ts` |
| Signal propagation through a real tool child | `process-lifecycle.test.ts` |
| Real-binary setup, onboarding, migration, and launch behavior | `real-binary-boot.test.ts` |

The smoke files own their child-process helpers. There is no shared
`tests/harness/spawn.ts` and no PTY smoke lane in the current tree.

## Harness and fixture modules

| File | Purpose |
|---|---|
| `tests/harness/tmp-root.ts` | Preloaded guarded temp root and cleanup for every root test run |
| `tests/harness/tmp-git-guard.ts` | Prevents accidental `.git` creation in the system or test temp root |
| `tests/harness/scratch-env.ts` | Child-process and in-process Clio state isolation |
| `tests/harness/dispatch.ts` | Dispatch bundle, fast reproducibility, isolated state, and event-loop helpers |
| `tests/harness/dispatch-stub-context.ts` | Minimal domain context for dispatch contracts |
| `tests/harness/receipt.ts` | Typed run-envelope and receipt fixtures |
| `tests/harness/openai-compat-fixture.ts` | Loopback OpenAI-compatible server and target seeders |

Child fixtures in `tests/fixtures/` are
`capacity-lease-child.ts`, `codewiki-coordinator-child.ts`, and
`evidence-index-writer.ts`.

## Running a subset

```bash
# all contracts (the shell expands the file pattern)
npm run test:full

# one contract or smoke file
npm run test:file -- tests/contracts/skill-install.test.ts
npm run build
npm run test:file -- tests/extended-smoke/cli-core.test.ts

# only it.only or describe.only within one file
npm run test:file -- --test-only tests/contracts/skill-install.test.ts
```

## Writing tests

- Use `node:test` and `node:assert/strict`.
- End local TypeScript import specifiers in `.js` for NodeNext resolution.
- Keep `tsconfig.tests.json` strict, including `noUncheckedIndexedAccess` and
  `exactOptionalPropertyTypes`; narrow indexed values before use.
- Let the package script preload `tmp-root.ts`. Tests that mutate Clio state or
  `process.env` should use `scratch-env.ts` and restore in teardown.
- Use a loopback fixture for provider behavior. Do not contact a configured or
  public model from a deterministic test.
- Build before a focused smoke run. `npm run ci` already builds before testing.
- Keep a smoke process driver local to the boundary it exercises unless a
  genuinely shared contract appears.
