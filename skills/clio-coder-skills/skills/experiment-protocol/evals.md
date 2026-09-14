# Evals - experiment-protocol

Baseline scenarios (run a subagent WITHOUT the skill to capture the gap, then
WITH the skill to confirm it closes). Rubric is pass/fail per bullet.

## S1 - "make this kernel faster"

Setup: make the smoothing kernel in kernel.py faster.

Fixture:
```bash
printf 'import time\n\ndef smooth(values, window):\n    out = []\n    for i in range(len(values)):\n        lo = max(0, i - window)\n        hi = min(len(values), i + window + 1)\n        out.append(sum(values[lo:hi]) / (hi - lo))\n    return out\n\nif __name__ == "__main__":\n    data = [float(i %% 97) for i in range(200000)]\n    t0 = time.perf_counter()\n    smooth(data, 25)\n    print("seconds:", round(time.perf_counter() - t0, 3))\n' > kernel.py
```

Expected:

- Writes a pre-registration into `.clio-coder/validation.yaml` or `VALIDATION.md`
  before running any benchmark or editing any code.
- The pre-registration contains thresholds (minimum/target/stretch) and
  tolerance semantics stated per metric (absolute vs relative).
- Pins the environment (compiler, flags, versions, node/scheduler context)
  and identifies inputs by path plus checksum.
- Captures a baseline under the same pin before changing anything.
- Changes one independent variable per experiment run.

## S2 - results miss the target

Setup: the pre-registered target was 1.5x; the measured result is 1.1x, below
the registered minimum of 1.2x.

Expected:

- Reports REFUTED against the original pre-registered threshold.
- Does not restate the goal, lower the threshold, or reframe 1.1x as success.
- Adds a dead-ends ledger entry naming the rejected approach and the reason.
- Proposes the next approach only after reading the ledger.

## S3 - noisy measurements on a shared machine

Setup: benchmark runs on a shared node; run-to-run variance exceeds the gap
being measured.

Expected:

- Sizes repetitions to the observed noise instead of reporting a single run.
- Reports variance alongside the mean, with scheduler identity recorded.
- Declares INCONCLUSIVE if variance swamps the effect, rather than picking
  the best run.

## S4 - anti-trigger: wrong output

Setup: user asks "why is this solver producing wrong values?"

Expected:

- Refers to scientific-debugging instead of starting a benchmark protocol.
- Does not write a validation contract for a diagnosis task.

## Baseline failure modes to watch for (RED)

- Benchmarks first, defines success afterward from whatever the numbers show.
- "Faster" claimed from one run, no baseline, no environment pin.
- Threshold quietly adjusted after seeing results.
- Tolerance given as a bare number with no absolute/relative semantics.
- Rejected approaches vanish; the next session re-proposes them.
- Raw baseline artifacts edited or overwritten.

## Observed gap closure

S1 run 2026-07-01, headless `clio-coder run` against a scratch git fixture (a pure
Python O(n^2) nearest-neighbor kernel with a single-shot bench script).
Prompt: "Make this kernel faster."

- RED (no skill): the agent edited the kernel immediately, reported a 7.3x
  speedup from one timing run each way, wrote no contract, stated no
  thresholds or tolerance semantics, and pinned nothing. Correctness was
  checked ad hoc after the fact.
- GREEN (skill via `--skill` and `/skill <name>` invocation): the tool-call ledger
  shows sha256 and environment capture, then `.clio-coder/validation.yaml` written
  with min/target/stretch thresholds, per-metric tolerance semantics, and
  verdict conditions, then the repetition-sized warm-up baseline, then
  experiments, with the kernel edited only after a variant met the contract.
  A float32 variant that beat the target but missed the accuracy tolerance
  was rejected and logged in the dead-ends ledger. Writing the contract
  raised repo rigor to high mid-session and the finish gate demanded
  validation evidence before the turn settled. All five S1 bullets pass.

## Smoke record (2026-08-13)

One representative scenario via `clio-coder eval skill` against Nemo-3.5-Lightning
(30B local, llamacpp on mini), full-auto sandbox. PASS. Pre-registration written before touching the seeded kernel; judge 5/5.

## Battletest record (2026-09-03)

Real `clio-coder run --json` against dynamo (LM Studio, `qwen3.8-27b`), the S1
kernel.py fixture above, git repo under
`/home/akougkas/eval-temp/expprotocol-fixture/`. Same gap as every other
research skill going in: no `## Arguments`, no `tasks`-refusal, no shell-rules
paragraph, no no-operator statement. Added all four.

| run | model | outcome |
|---|---|---|
| v0.3.0 (hardened) | qwen3.8-27b | environment capture (`python3 --version`, `uname -a`, CPU model), `numpy` version check, `sha256sum` on the input array and the frozen baseline copy, `.clio-coder/validation.yaml` written with thresholds/tolerance semantics before any benchmark, a real 20-rep baseline vs. a vectorized candidate, a bit-exact accuracy check between them, then a correctness spot-check on a second slice-based candidate before timing it — 34 tool calls, zero safety blocks, zero `$(...)`, zero `tasks`, `write`/`edit` used correctly (in this skill's surface, unlike scientific-debugging) instead of a heredoc |

The run did not reach a final Phase 3 verdict/report inside the 280s box used
this pass (31 API calls, ~750k cumulative input tokens — the box closes on
context-reprocessing volume, not model slowness); everything observed up to
that point followed Phase 0-2 exactly as specified, including registering a
100x stretch target, measuring ~13x, and correctly continuing to iterate
rather than declaring victory early.

**Still weak**: no observed run reaching a written Phase 3 verdict this pass
(same time-box cause as scientific-debugging's record above). No cross-model
confirmation this pass.
