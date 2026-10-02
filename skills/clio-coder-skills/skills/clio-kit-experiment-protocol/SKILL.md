---
name: clio-kit-experiment-protocol
description: 'Use when this workflow is requested: Pre-registers a performance study,
  numerical comparison, parameter sweep, or benchmark: thresholds, tolerances, environment
  pins, and verdict conditions locked into the validation contract before any measurement.
  Not for diagnosing a stalled bug; use scientific-debugging. Triggers on "benchmark
  these implementations", "pre-register a performance experiment".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/research/experiment-protocol
  upstream-name: experiment-protocol
  upstream-eval-status: smoke-checked
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

Adapted skills use the `clio-kit-` prefix to distinguish them from upstream audited skills. For a companion skill named below, select its `clio-kit-` copy from this collection; native agents, fleets and external programs retain their original names.

# Experiment Protocol

Lock the success criteria before any result exists. Pre-registration that
happens after the first measurement is worthless: it can only ratify what was
already seen. Moving the threshold after seeing results is the exact failure
mode this protocol exists to prevent.

Anti-trigger: if the question is "why is this output wrong", that is a
diagnosis, not an experiment; use scientific-debugging.

## Arguments

```text
/skill clio-kit-experiment-protocol <what to benchmark, compare, or sweep>
```

There is no operator in a headless run: `ask_user` is not registered, so
any call is refused as an unregistered tool rather than answered. If a threshold, tolerance, or environment detail is unstated,
pick the most defensible default, record it as an explicit assumption in the
pre-registration, and proceed; never stall Phase 0 waiting for confirmation.

The phases below are the plan; do not open a task list for them — `tasks`
sits outside this skill's tool surface and any call to it is refused.

Shell rules for every `bash` call: one command per call, plain and direct.
Never use `$(...)` or backticks; they trigger an approval gate that ends a
headless run. Capture checksums and environment facts with direct calls
(`sha256sum data/mesh.h5`), never command substitution.

## Phase 0 - Pre-register

Before any measurement, write the protocol into the repository validation
contract: `.clio-coder/validation.yaml` (preferred) or `VALIDATION.md` at the repo
root. Creating this file raises Clio's repo-derived rigor to high immediately;
from the next turn onward, completion claims in this session must carry
validation evidence or state a limitation. That escalation is the point:
the contract arms the finish gate with the criteria you are about to commit to.

The pre-registration must contain:

- **Outcome**: one sentence, e.g. "the fused kernel reaches at least 1.8x the
  baseline throughput on the pinned input at equal accuracy".
- **Thresholds**: minimum (below this is REFUTED), target, stretch.
- **Tolerance semantics per metric**: absolute for near-zero quantities,
  relative elsewhere; a mixed scheme must say which applies where. "1e-6" with
  no stated semantics is not a tolerance.
- **Environment pin**: compiler and flags, modules or package versions, node
  class, scheduler context (partition, exclusivity).
- **Input identity**: paths plus checksums (`sha256sum`).
- **Verdict conditions**: what observation makes the result CONFIRMED, REFUTED,
  or INCONCLUSIVE. Committed now, immutable after the first measurement.

## Phase 1 - Baseline

Capture current behavior under the pinned environment before changing
anything. Store raw outputs and timings as artifacts; never edit them. A
speedup claim without a baseline captured under the same pin is a guess.

## Phase 2 - Experiment

- One independent variable per run. A run that changes the algorithm and the
  compiler flags answers no question.
- Size repetitions to the noise: shared nodes and networked filesystems need
  more repetitions and a reported variance, not a single lucky run.
- Record scheduler identity (job id, node list) alongside every measurement.

## Phase 3 - Analysis

Compare against the pre-registered thresholds only. If the protocol was
deviated from, log the deviation next to the result; do not silently absorb
it. Findings outside the registered outcome are marked exploratory and get
their own pre-registration if pursued.

## Phase 4 - Iterate

Keep a dead-ends ledger in the contract file or beside it: one line per
rejected approach with the reason it was rejected. Read it before proposing
the next approach; re-proposing a ledger entry wastes a run. Stop when one of
the pre-registered stop conditions holds: target met, budget exhausted, or all
candidate strategies rejected.

## Worked Example

Request: "make the halo exchange faster."

```yaml
# .clio-coder/validation.yaml
experiment: halo-exchange-overlap
outcome: overlap communication with interior compute; >= 1.5x step throughput
thresholds: { minimum: 1.2x, target: 1.5x, stretch: 2.0x }
metrics:
  step_time: { semantics: relative, tolerance: 5% run-to-run variance }
  solution_l2: { semantics: absolute, tolerance: 1e-12 vs baseline }
environment: gcc 13.2 -O3, openmpi 4.1.6, 4x cpu-bind=cores, exclusive nodes
inputs: { mesh: data/mesh-256.h5, sha256: "<checksum>" }
verdicts:
  confirmed: step_time speedup >= 1.2x AND solution_l2 within tolerance
  refuted: speedup < 1.2x with variance < 5%, or accuracy loss
  inconclusive: run-to-run variance > 5% (resize repetitions first)
dead_ends: []
```

Baseline: 20 repetitions on exclusive nodes, mean and variance recorded with
job ids. Experiment: nonblocking exchange only, flags untouched. Result: 1.3x,
accuracy within 1e-12: CONFIRMED at minimum, target not met; ledger gains
"persistent requests: no gain over nonblocking here, latency-bound", and the
next variable (message aggregation) gets its own run.

## Red Flags

- Any measurement taken before the contract file exists.
- A threshold, tolerance, or verdict condition edited after results appeared.
- Wall-clock numbers with no environment pin attached.
- A single-run victory claim on a shared machine.
- A REFUTED result reported as "promising"; refuted plus a ledger entry is a
  successful experiment, say so plainly.
- Deleting or rewriting baseline artifacts.
