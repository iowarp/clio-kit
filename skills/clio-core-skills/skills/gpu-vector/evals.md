# Evals - gpu-vector

Scenarios recorded 2026-11-20 against clio-core `fs-large-write-perf`
(6eba3abe5). Rubric is pass/fail per bullet. Run each WITHOUT the skill to
capture the gap, then WITH it to confirm the gap closes.

## S1 - port a dense kernel out of core

Setup: Prompt: "Port this stencil kernel to gpu_vector so the grid can be 4x
bigger than GPU memory."

Expected:

- Writes or keeps a plain device-array baseline and compares at matched
  geometry (same blocks x threads).
- Lays the data out so a page is a work unit with exactly one writer.
- Uses `Init(yv.Block())`, never `blockIdx.x`.
- Fetches each step's working set in batched ranges (<= 8 ranges, <= 64
  pages), holds only fetched bytes, flushes written ranges before unpinning.
- Validates resident (zero evictions, equal answer) and then out of core
  (evictions > 0, same answer), and checks `HitRoundCap()`.

## S2 - "it passes" with zero evictions

Setup: Prompt: "My gpu_vector kernel matches the baseline, so it works out of
core, right?" with a log that shows `evictions=0`.

Expected:

- States that a zero-eviction run proves nothing about out-of-core behavior
  and asks for a run with cache well below data.

## S3 - lost writes

Setup: Prompt: "Values I wrote on the GPU come back as zeros after the cache
fills up."

Expected:

- Explains that eviction does no I/O: unflushed GPU-generated data is lost.
- Points at the missing `Flush` before `UnpinRange`, not at the store.

## Baseline failure modes to watch for (RED)

- Calling a verb under a thread-dependent branch.
- Relying on plain `__shared__` or registers across a suspend.
- Declaring success without an out-of-core run or without checking the round cap.
