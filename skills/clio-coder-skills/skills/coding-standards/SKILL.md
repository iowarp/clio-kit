---
name: coding-standards
description: 'Use when this workflow is requested: Correct-by-construction TypeScript
  rules (errors as values, parse don''t validate, illegal states unrepresentable,
  deep modules) for writing or refactoring TypeScript when the project has not pinned
  its own standards. Triggers on "TypeScript coding standards", "errors as values".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/coding/coding-standards
  upstream-eval-status: scenarios-recorded
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

# Coding Standards (TypeScript, correct-by-construction)

Reference, not a workflow. Apply to new and refactored code. When rules
pull in different directions: correctness and debuggability first, then
these standards, then compatible project conventions; contain incompatible
existing patterns at the nearest boundary instead of copying them into new
code, and never rewrite unrelated old code without an explicit migration
request. **The host project's own documented standards always win over
this file.**

This skill's tool surface covers the whole write-code loop (read, grep,
find, ls, code_nav, bash, write, edit) so a solo run can apply the rules
it names. When it rides along with another skill, the surfaces merge as a
union, so it never narrows the tools the task in flight needs.

## Arguments

```text
/skill coding-standards [<task>]
```

- With a task: do the task, and hold every new or changed TypeScript line
  to the rules below. Keep using the ordinary `write`, `edit`, and `bash`
  tools; the standards change what you write, not how you write it.
- Without a task: answer as a reference. Quote the relevant rule and show
  a before/after snippet; do not touch the repository.

## How to apply

1. Read the host project's instruction file and one or two existing
   modules near the change. If the project pins its own conventions on
   errors, validation, or module layout, those win; note the conflict in
   one line and follow the host.
2. Before writing, state in your reply which rules the change touches (for
   a parser: errors, parse-don't-validate, illegal states; for a service:
   modules). This is the checklist for the code you are about to write.
3. Write the code. Then run the project's typecheck (`npx tsc --noEmit` or
   the repo script) with one plain `bash` call; never use `$(...)` or
   backticks in the command.
4. If you want runtime evidence beyond the typecheck, prefer a single
   `node -e` (or the project's runtime) call. If you must write a smoke
   script, put it under `.clio-coder/scratch/` and remove it with plain
   `rm <file>`; `rm -f`, `rm -r`, `find -delete`, and moving files to
   `/tmp` are all refused by the safety net in a headless run, and every
   refused attempt is a wasted turn.
5. Finish with a short audit of the diff against the checklist: each rule
   either satisfied, or deliberately not, with the reason. `git status`
   must show only the files the task asked for.

## Errors

- Expected failures are values in the return type, as custom tagged
  errors; callers handle or return them upward, and the outermost boundary
  translates them (HTTP response, exit code, retry decision). Use the
  repo's established result mechanism; absent one, a small local tagged
  union `{_tag: "ok"|"err", ...}` suffices.
- Promise rejection is throwing. Catch unclassified third-party rejections
  inside the owning adapter and translate before they cross its boundary.
- Throw only for defects: violated invariants, impossible branches,
  not-yet-implemented paths. A missing config value is an expected startup
  failure, not a defect.
- Custom errors carry: a stable `_tag` (`as const`), a useful message,
  structured context fields, optional `cause: unknown`. Keep error unions
  precise at module boundaries; broad `AppError` types belong only near
  entrypoints and rendering.

## Parse, don't validate

- Boundary code turns `unknown` input into domain types before it enters
  inner code; inner code never re-checks what a parser proved.
- `parseX(input): Result<X, ParseXError>` for untrusted input; `makeX` for
  smart construction from typed pieces; `isX` for predicates. A function
  returning a refined value is a parser — do not name it `validateX`.
- Schema libraries are boundary parsers, not validators sprinkled through
  core logic; use the repo's established one. Never pass a schema-inferred
  transport shape through the application.
- Brand types where a mix-up is realistic: IDs (`UserId` vs `OrgId`),
  parsed strings (`EmailAddress`), units (`Milliseconds`, `Bytes`).
  Construct branded values only through their parser.

## Illegal states unrepresentable

- Lifecycle states are tagged unions, not boolean fields:
  `{_tag:"Draft"|...}` beats `{isSent, isPaid, sentAt?, paidAt?}`.
- No behavior-controlling boolean parameters: `createUser(input,
  {emailVerification:"skip"})`, never `createUser(input, true)`.
- Push optionality outward: branch or parse before calling, not
  `undefined` checks inside every callee. `Partial<T>` is not an
  application input unless partiality is the domain concept.

## Modules

- Three responsibilities, not three mandatory layers: **domain** (meaning,
  invariants, pure transitions — no I/O, no ambient time or randomness),
  **application service** (policy and effect ordering through narrow,
  application-owned ports), **adapter** (one boundary's technology and
  translation; raw external types never leak out of it). Wire concrete
  adapters to ports at the composition root only.
- Deep modules: substantial behavior behind a low-burden interface.
  Deletion test: removing a module that makes complexity disappear was
  pass-through waste; removing one that spreads complexity across callers
  was earning its keep.
- Audit existing adapters/services before creating one: reuse as-is,
  extend if cohesive, create new only when reuse would couple wrongly.

## Testing

- Prefer, in order: end-to-end through real public entrypoints;
  integration through real seams; property tests for pure domain modules;
  unit tests of meaningful behavior.
- Never module-mock (`vi.mock`/`jest.mock`). Real seams: injected
  interfaces, local DB substitutes, in-memory or fake adapters.
- Assert observable outcomes (returned value, persisted state, emitted
  event), not call sequences. Tests never bypass parsers or invariants.

## Style and safety

- Strict compiler settings (`strict`, `noUncheckedIndexedAccess`,
  `exactOptionalPropertyTypes`). Immutable by default (`readonly`,
  `ReadonlyArray`); mutation stays localized behind precise interfaces.
- No `any`, no `!`, no `as` casts (`as const` is fine). The rare
  justified cast carries a `// SAFETY:` comment explaining why the
  invariant holds and why the type system cannot express it.
- Direct imports from owning files; no barrel layers by default. Precise
  file names (`email-address.ts`), never `utils.ts`/`helpers.ts`.
- Parse env/config at startup into typed config; no `process.env` reads
  scattered through the app; no import-time side effects outside true
  entrypoints. Secrets live in `Redacted`-style wrappers, unwrapped only
  at the adapter that needs the raw value, and never appear in errors,
  logs, or traces.
