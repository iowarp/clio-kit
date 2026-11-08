---
name: clio-kit-ast-grep
description: 'Use when this workflow is requested: Structural code search with ast-grep
  rules: finds code by shape when grep on text is too noisy, and never rewrites it.
  Not for plain string or filename lookups; use grep or find. Triggers on "structural
  code search", "ast-grep rule".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/coding/ast-grep
  upstream-name: ast-grep
  upstream-eval-status: scenarios-recorded
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

Adapted skills use the `clio-kit-` prefix to distinguish them from upstream audited skills. For a companion skill named below, select its `clio-kit-` copy from this collection; native agents, fleets and external programs retain their original names.

# ast-grep Structural Search

Translate a structural question into an ast-grep rule, prove the rule on a
tiny example, then scan the real codebase. Never scan first: an untested rule
that returns nothing tells you nothing.

Search only. Never use `--update-all` / `-U` / `--rewrite`; when matches need
changing, report them and edit through the normal edit tools.

## Arguments

```text
/skill clio-kit-ast-grep [--lang <language>] [--path <dir>] <structural question>
```

- `--lang`: tree-sitter language name (`javascript`, `typescript`, `python`,
  `rust`, `go`, ...). Omit to infer from the files under `--path`.
- `--path`: directory or file to scan. Omit to scan the project source
  directory (`src/` when it exists, otherwise `.`).
- Everything else is the question: what shape to find, what to exclude.

The five steps below are the plan; do not open a task list for them.

Examples:

- `/skill clio-kit-ast-grep find async functions that await but have no try/catch`
- `/skill clio-kit-ast-grep --lang python --path app/ find calls to open() outside a with block`

Shell rules for every `bash` call in this workflow: one command per call,
plain and direct. Never use `$(...)` or backticks; they trigger an approval
gate that ends a headless run. Always pass rules through a file, never
through `--inline-rules`, because `$VAR` metavariables inside shell quotes
produce quoting errors that look exactly like non-matches.

## Step 0 — Check the binary

```bash
ast-grep --version
```

If that fails, try the short alias, but verify it is really ast-grep:

```bash
sg --version
```

Only an output that starts with `ast-grep` counts. On many Linux systems
`sg` is the unrelated shadow-utils "switch group" command and prints a
usage error; that is NOT ast-grep. If neither binary is ast-grep, say so
(`npm i -g @ast-grep/cli` or `cargo install ast-grep`) and stop. Do not
fall back to text grep and present it as a structural result.

Use the resolved binary name in every later command.

## Step 1 — Pin down the question

Before writing any rule, write in your reply: the target language; the
structure to match; what must be included and excluded. If the request is
ambiguous on any of these, ask one clarifying question. When running
headlessly, state your reading of the ambiguity and proceed.

## Step 2 — Write a minimal example

Create a scratch directory inside the workspace and write the smallest
code snippet that MUST match and, when exclusions matter, one that must
NOT match:

```bash
mkdir -p .clio-coder/scratch/ast-grep
```

Then write `.clio-coder/scratch/ast-grep/positive.<ext>` and
`.clio-coder/scratch/ast-grep/negative.<ext>` with the `write` tool. These
are the rule's test fixture: one file that must match, one that must not,
nothing more. Never write under `/tmp`, not even by shell redirection; a
write outside the workspace needs an approval a headless run cannot give,
and the call is refused.

## Step 3 — Write the rule, simplest first

Write `.clio-coder/scratch/ast-grep/rule.yml` with the `write` tool.
Escalate only when the simpler form fails:

1. `pattern` alone for direct shapes: `console.log($ARG)`.
2. `kind` plus relational rules (`has`, `inside`) for context: "function
   containing await".
3. Composite rules (`all`, `any`, `not`) for logic: "has await but no
   try-catch".

Non-negotiable: every relational rule (`has`, `inside`) carries
`stopBy: end`, or traversal stops at the first non-matching child and the
rule silently under-matches.

```yaml
id: async-no-trycatch
language: javascript
rule:
  all:
    - kind: function_declaration
    - has: {pattern: await $EXPR, stopBy: end}
    - not:
        has: {pattern: 'try { $$$ } catch ($E) { $$$ }', stopBy: end}
```

This example is verified against ast-grep 0.45 on JavaScript. When the
question is this one, use it verbatim. Two facts it encodes that models
guess wrong: tree-sitter JavaScript and TypeScript have no
`async_function_declaration` kind (the `async` keyword is a token inside
`function_declaration`; assert it with `pattern: 'async function $F($$$) { $$$ }'`
when async matters), and an unknown `kind` fails the whole rule with
`Kind ... is invalid` (exit 8); answer that error with the tree dump in
Step 4, never with another guessed name.

Full syntax (atomic, relational, composite rules, metavariables, `constraints`,
`utils`): read `references/rule_reference.md` when the rule needs more than
the three forms above.

## Step 4 — Test against the example

```bash
ast-grep scan --rule .clio-coder/scratch/ast-grep/rule.yml .clio-coder/scratch/ast-grep
```

The rule must match the positive example and miss the negative one before
you scan anything real. When it passes, go to Step 5. When it does not,
debug in this order, changing the rule file with `edit`:

1. Strip the rule to its simplest positive key; re-add parts one at a time.
2. Confirm every `has`/`inside` has `stopBy: end`.
3. Dump the tree to check node kinds; never guess a `kind` name:

```bash
ast-grep run --pattern 'async function f() { await g(); }' --lang javascript --debug-query=cst
```

(`cst` = every node; `ast` = named nodes only; `pattern` = how ast-grep
parsed your pattern.) Pipe long dumps through `head -80`. Dump the tree
only after a test has failed; a dump before the first test is a wasted
turn.

## Step 5 — Scan the codebase

Simple one-node patterns skip the rule file:

```bash
ast-grep run --pattern 'console.log($ARG)' --lang javascript src
```

Rule-based scans:

```bash
ast-grep scan --rule .clio-coder/scratch/ast-grep/rule.yml src
```

Scan the requested path, not `.`, so your own fixture under
`.clio-coder/scratch` does not show up as a match; if it does, drop those
lines from the report. Add `--json` only when the output feeds further
processing.

Report matches as `path:line` with a one-line reading of each; do not paste
raw JSON at the user. Done when the reported matches answer the Step 1
question and false positives you noticed are either excluded by the rule or
called out. Delete the scratch files last, naming each file. `rm -r` and
`rm -f` are hard-blocked by the safety net, so:

```bash
rm .clio-coder/scratch/ast-grep/positive.js .clio-coder/scratch/ast-grep/negative.js .clio-coder/scratch/ast-grep/rule.yml
```
```bash
rmdir .clio-coder/scratch/ast-grep
```

## Red flags

- Scanning the codebase with a rule never proven on an example.
- A `has`/`inside` without `stopBy: end`.
- A guessed `kind` name instead of a `--debug-query=cst` dump.
- Any rewrite flag: this skill only reads.
- Treating shadow-utils `sg` as ast-grep because `command -v sg` succeeded.
- Cleaning up with `rm -rf`, or writing fixtures under `/tmp`; both are
  refused in a headless run.
- Falling back to grep halfway because rule debugging felt slow; fix the
  rule with Step 4, that is the job.
