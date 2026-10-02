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
