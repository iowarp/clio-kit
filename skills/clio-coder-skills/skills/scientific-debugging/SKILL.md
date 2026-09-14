---
name: scientific-debugging
description: 'Use when this workflow is requested: Diagnoses stalled or cross-system
  failures and wrong, NaN, nondeterministic, or unexpectedly slow scientific results
  through falsifiable hypotheses across distinct fault classes, with evidence-cited
  verdicts before any fix. Not for designing benchmarks or pre-registered experiments;
  use experiment-protocol. Triggers on "wrong scientific results", "nondeterministic
  HPC code".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/research/scientific-debugging
  upstream-eval-status: smoke-checked
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

# Scientific Debugging

Debug by falsification, not by trying fixes. A fix attempted before a confirmed
diagnosis is an experiment run without a hypothesis; when it "works" you have
learned nothing, and when it does not you have contaminated the evidence.

Anti-trigger: if the failure is a typo, a missing import, or an error message
that names its own cause, fix it directly and skip this workflow. The loop
below is for failures that survived the first obvious fix.

## Arguments

```text
/skill scientific-debugging <failure description>
```

Everything after the skill name is the failure report: the observed wrong
behavior and whatever has already been tried. There is no operator in a
headless run: `ask_user` is not registered, so any call is refused as an
unregistered tool rather than answered. If the goal,
a fault-class split, or a ranking call is ambiguous, state your best reading
in Step 1 or Step 3 and proceed; never stall a step waiting for confirmation.

The Loop below is the plan; do not open a task list for it — `tasks` sits
outside this skill's tool surface and any call to it is refused.

Shell rules for every `bash` call: one command per call, plain and direct.
Never use `$(...)` or backticks; they trigger an approval gate that ends a
headless run. This skill has no `write`/`edit` tool — the structured-
investigation file in the Tiers section below is written with a `bash`
heredoc (`cat > file <<'EOF' ... EOF`), never through an edit tool that
isn't in this skill's surface.

## The Loop

1. **Goal.** One sentence stating the observable "fixed" state. "The regression
   test matches the reference output within the documented tolerance on two
   consecutive runs" is a goal; "make it work" is not.
2. **Hypothesize.** Write at least three hypotheses. Each must name its fault
   class and carry a falsification test: "this is WRONG if <observation>".
   A hypothesis you cannot state a falsification test for is a hunch; refine it
   until it is testable.
3. **Rank.** Order by test cost times prior likelihood. Run the cheapest
   decisive test first, not the most interesting one.
4. **Test.** One variable per test. Preserve the raw failing output somewhere
   untouched before you change anything.
5. **Verdict.** Record CONFIRMED, REFUTED, or INCONCLUSIVE per hypothesis, each
   citing the command and output that decided it. A verdict without a citable
   observation is a guess.
6. **Iterate.** Refuted everything? Generate new hypotheses from what the tests
   revealed. Confirmed one? Only now edit code.

## Fault Classes

Hypotheses must span at least two distinct classes. Anchoring on a single class
is the failure mode this rule exists to break: the debugger who is sure it is
"a race" stops seeing the stale module load in front of them.

| Class | Typical suspects |
|---|---|
| numerics | accumulation order, mixed precision, tolerance misuse, fastmath |
| data | format or layout drift, HDF5/NetCDF/Zarr metadata, units, corruption |
| concurrency | races, MPI collective mismatch, nondeterministic reduction order |
| environment | modules, compiler flags, library versions, scheduler context |
| resources | memory pressure, filesystem quirks, quota, node differences |
| regression | a recent change; bisect the history instead of staring at code |

## Tiers

**Quick diagnosis** (default): the loop above, state held in conversation,
time-boxed at fifteen minutes of investigation. If the box expires without a
CONFIRMED verdict, escalate. Say that you are escalating; do not silently keep
poking.

**Structured investigation**: write an investigation file (for example
`INVESTIGATION.md` or `.clio-coder/investigation-<slug>.md` via bash heredoc since
this skill does not edit code) containing the goal, baseline measurements of
the failing behavior, and one experiment per hypothesis with its verdict
condition committed *before* the experiment runs. Update verdicts as evidence
arrives. The file is the state; the conversation is commentary.

## Evidence Rule

The fix commit should cite the confirming observation, e.g. "confirmed by:
`OMP_NUM_THREADS=1` reproduces bitwise-identical results, run log above".
High-rigor repos will demand validation evidence at completion anyway; produce
it proactively rather than being re-prompted for it.

## Worked Example

Report: a weighted mean changed after a reduction refactor. Preserve the failing input and inspect the current implementation before choosing a fix.

- Goal: match an independently computed reference within a predeclared absolute tolerance on cancellation-heavy inputs.
- H1 (numerics): sequential floating-point addition loses small terms between large opposing terms. Refute if sequential addition and `math.fsum` agree on the failing input.
- H2 (data): values or weights changed. Refute by comparing input checksums and the exact arrays supplied to both implementations.
- H3 (environment): runtime-dependent reduction behavior explains the observation. Compare the same explicit loop and `math.fsum` under the recorded Python version; do not assume built-in `sum` uses naive accumulation.
- If the same input yields different sequential and compensated reductions, report that observation as evidence for H1. It does not by itself prove when the regression was introduced.
- Use a detached temporary checkout to compare revisions when needed; preserve the user's working tree. A clean prior revision supports a regression hypothesis rather than refuting it.
- Keep absolute tolerance near zero and relative tolerance away from zero explicit. Do not loosen tolerances to hide a numerical error.

## Red Flags

- Editing code before any hypothesis has a CONFIRMED verdict.
- All hypotheses drawn from one fault class.
- A test that changes two variables at once.
- "It seems better now" presented as a verdict.
- Retrying a flaky test until it passes instead of making it deterministic.
- The fifteen-minute box expiring without an explicit escalation.
- Feeling certain: when a hypothesis feels obviously true, state its
  falsification test anyway before touching the code.
