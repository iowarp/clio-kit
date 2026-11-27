---
name: clio-kit-clio-coder-test
description: 'Use when this workflow is requested: Verifies a Clio Coder source change
  against the current contract, smoke, boundary, and application test lanes, including
  temporary-state and local-provider harnesses. Not for deciding what may change;
  use clio-coder-dev. Triggers on "test Clio Coder", "verify a Clio source change".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/meta/clio-coder-test
  upstream-name: clio-coder-test
  upstream-eval-status: scenarios-recorded
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

Adapted skills use the `clio-kit-` prefix to distinguish them from upstream audited skills. For a companion skill named below, select its `clio-kit-` copy from this collection; native agents, fleets and external programs retain their original names.

# Clio Test

The root Node suite has a small required contract/process lane, an installed-package
qualification lane, and explicit extended development regressions. Contract tests
import source through tsx; CLI tests use the current built binary. Lint owns static
import boundaries and library pins. Publication performs only an exact-candidate
preflight after successful qualification.

For the question of whether a change may leave your machine (commit, push, or
PR), use `clio-coder-dev`. **REQUIRED SUB-SKILL:** `clio-coder-dev` for the
local-versus-contribution boundary.

## Commands

```bash
npm run typecheck                         # tsc -p tsconfig.tests.json
npm run lint                              # Biome plus scripts/check-hygiene.ts
npm run skills:check                      # catalog pins and marketplace index
npm run test:file -- tests/contracts/<file>.test.ts
npm run build                             # tsup plus the codewiki asset
npm run test:file -- tests/smoke/<file>.test.ts  # requires a current build
npm run test                              # required contracts and three process smoke files
npm run test:full                         # explicit extended root investigation
npm run test:web:full                     # explicit extended web investigation
pnpm --filter @iowarp/clio-coder-web verify # app types, lint, tests, build, browser
npm run ci                                # deterministic root gate
npm run ci:release                        # qualify clean committed source and exact installed tarball
npm run release:preflight                 # fast check of the unchanged qualified package
```

The required web tests cover authentication, permissions, cancellation, worker RPC,
egress and process ownership. The installed-package lane includes one real browser
boot/reconnect against the installed server. The full web `verify` command is an
explicit development investigation, including its viewport/accessibility matrix.
Do not run full development tests repeatedly during release or publication.

The separate trace viewer is retired. `apps/workbench/` is retained reference
source, excluded from builds, publication and product gates. Use `test:web` for
the required app tests alone; app `verify` requires headless Chrome.

No deterministic gate contacts a real model. When a task explicitly requires
live validation, build first and run `node dist/cli/index.js run` against a
configured target. Record the target, model, runtime, prompt, and serving
settings with the result.

## Which lane catches what

| Change site | Run first | Why |
|---|---|---|
| Pure logic in `src/domains/<x>/*.ts` | Closest file under `tests/contracts/` | Contracts import current source; no build is needed. |
| Dispatch, providers, prompts, safety, config, or persistence | Related contract file found with `rg` | Each behavior is divided among focused contract files rather than one domain-wide suite. |
| Skill catalog or loader | Related contract, then `npm run skills:pin` and `npm run skills:check` | Runtime behavior and generated catalog metadata are separate checks. |
| Any import edit under `src/` | `npm run lint` | Hygiene invokes all six boundary rules. |
| CLI, entry, process lifecycle, or ACP stdio flow | Build, then the closest file under `tests/smoke/` | Smoke executes the built binary. |
| Published package contents | Build, then `tests/smoke/installed-package.test.ts` | The test packs and installs the actual artifact. |
| Unified web application | `npm run test:web` | Required authentication/runtime boundaries; full `verify` is optional development work. |

Read `references/test-map.md` for the current file map and exact subset
commands.

## Boundary rules you must not break

`tests/boundaries/check-boundaries.ts` enforces six rules through
`scripts/check-hygiene.ts` and `npm run lint`. Fix a reported dependency edge;
never suppress the checker.

1. Imports of `@earendil-works/pi-*`, including type-only imports, stay under
   `src/engine/**`.
2. Worker value imports from domains are limited to the declared provider
   runtime rehydration seams. Other worker imports from domains must be
   type-only.
3. A domain never imports another domain's `extension.ts`.
4. `src/tools/**` never imports `src/interactive/**`.
5. `src/interactive/turn-*.ts` and `chat-loop.ts` never import `src/entry/**`.
6. External value imports enter the protected Stage 0 trees only through
   declared seams, and those seams do not create undeclared edges back into
   the closure.

The authoritative definitions and exceptions are in
`docs/architecture/architecture.md` under "Boundary invariants."

## Source and configuration reload

Contract tests and hygiene read current source, so they need no build. Smoke
tests run `dist/`, so rebuild first or keep `npm run dev` (`scripts/build.ts --watch`)
running. A running Clio process does not reload changed ESM modules; restart it
after a fresh build.

Configuration reload is separate. `src/domains/config/classify.ts` owns the
three buckets:

- `hotReload`: keybindings, autonomy, the model picker, smooth streaming, pane
  notifications and file opens, Git attribution, and safety review settings.
- `nextTurn`: targets; the remaining chat settings; most fleet, context, and
  safety limits; selected interface output settings; project resources,
  external agents, and library settings.
- `restartRequired`: fleet concurrency, interface mode and scrollbar, pane-host
  enablement, runtime plugins, and every unknown path.

## Iteration loop

1. Read the relevant source contract and nearby tests.
2. Run `npm run typecheck` and `npm run lint`.
3. Run the narrowest related contract or application test.
4. If the built boundary changed, build and run the closest smoke file.
5. Run `npm run ci` before handing back a broad root change. Use
   `npm run ci:release` on the clean committed candidate when packaging is in scope.
   It records the exact source/artifact; use only `release:preflight` afterward.
6. Report exactly what ran and what remains unverified.

For one test file or an `it.only` while debugging:

```bash
npm run test:file -- tests/contracts/<file>.test.ts
npm run test:file -- --test-only tests/contracts/<file>.test.ts
```

## What not to do

- Do not reintroduce `tests/unit`, `tests/integration`, or `tests/e2e`; the
  current taxonomy is contracts and smoke.
- Do not cite a shared spawn or PTY helper. The current tree has neither
  `tests/harness/spawn.ts` nor `tests/harness/pty.ts`; each smoke boundary owns
  its process driver.
- Do not add `scripts/diag-*.ts` or `scripts/verify-*.ts`. A durable check
  belongs in `tests/`; a disposable probe belongs under `/tmp`.
- Do not hide a boundary failure with an ignore or exclusion.
- Do not claim live-model evidence from a local fixture.
- Do not delete or skip a pre-existing failure. Report the evidence and
  separate it from the result of the requested change.

## Harness reference

For the current scratch-state helpers, dispatch fixtures, local
OpenAI-compatible fixture, ACP stdio driver, and disposable-probe pattern, read
`references/harness.md`.
