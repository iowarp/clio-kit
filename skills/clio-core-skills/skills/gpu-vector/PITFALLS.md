# PITFALLS.md -- traps, validation, debugging

## 1. Correctness traps, most expensive first

**Silent wrong answers**
- **Lost writes.** Eviction does no I/O and there is no dirty bit. A written page that is unpinned before its flush lands can be evicted and the write is gone. Order: fetch -> write hold -> flush (`EndFlush` landed) -> `UnpinRange`. This includes data *generated* on the GPU: until it is flushed it exists only in the cache. `ClearCache` also discards unflushed data -- a code that called it mid-run silently reloaded its initial state with every check green.
- **Flush of a non-resident page is silently skipped** (`flush_skipped`). One multi-node code lost every boundary publish this way.
- **Non-disjoint writers.** Two blocks writing one page through whole-page flushes clobber each other; counters stay clean. One owner per destination (gather, not scatter).
- **Write holds widen the valid range.** Two disjoint write holds mark the gap valid; a later read of the gap passes the check and returns garbage. Fetch before a partial write.
- **Generation 0 means "any version".** A consumer of a peer's page that passes 0 can read a stale copy (three separate defects were this).
- **`CoFetch`/`MFetch` lack the co_await `Fetch`'s wait-until-generation loop.** In the transpiled and macro forms a stale frame can be read as current.
- **Rescoring pages you are about to rewrite** moved them between tiers and served a stale replica; do not.
- **Round cap**: `RunToCompletion` stops at `max_rounds`. It once returned normally after 17 of 22 iterations. Always check `HitRoundCap()`.
- **Reused driver**: a `Yieldable`/`YieldStack` reused without `Reset()` still reads "done" and returns an instant, empty success.

**Silent drops at the limits**
- More than 8 ranges per fetch or flush: extras are dropped silently.
- More than 64 pages per fetch: extras are never fetched; a later hold traps (code 2).
- More than 64 records per flush: traps (code 6).

**Block-collective divergence**
- A branch to a barrier whose condition reads state other blocks mutate must be voted with `__syncthreads_or`. Unvoted, it produced "illegal instruction" one run in ten, vanishing under `CUDA_LAUNCH_BLOCKING` and compute-sanitizer.
- Never put a verb under `if (threadIdx.x == 0)` or after an early `return` of some lanes. Every lane runs every loop iteration that contains a verb.

**State across a park**
- Plain `__shared__` is garbage after any suspend: shared-memory staging tables caused an MMU fault in `$_resume`, invisible for weeks because resident runs never park. Use `CLIO_SHARED_PERSIST` or re-stage.
- Transpiled coroutines: values across `CO_AWAIT` must be trivially copyable; no named `Held`; no awaiter capturing `this` (Intel GPUs move private memory: "CoroHandle: null RunContext"); no default arguments.
- Bind with `Init(yv.Block())`, never `blockIdx.x`.
- C++20 frames need ~8 KB lanes (256 B overflows; code 103).

**Generations**
- A generational get waits at most `CLIO_GEN_WAIT_MS` (default 120 s; it was a fixed 10 s, which fired on healthy neighbours at 16+ nodes), then fails -> code 7. A writer that never publishes shows up far away, much later.
- A page's generation is stamped by the fetch that delivers it. Demanding a generation on your own page hangs; demand only on peers' pages; initial data publishes as 1.
- Off-by-one on the last generation (demanding one more than was published) only worked while generation-0 data satisfied any demand.
- Several writers to the same bytes in one generation are unsupported; a shared page under a generational demand can stall forever ("gen stall ... fetching=1 pins=4").
- Publish and demand a generation in separate launches.

**Capacity**
- Set full of pinned frames -> code 5 after long retries. Causes: cache below the floor, too few ways per set, leaked pins (every fetch needs one `UnpinRange`), hold-while-waiting.
- The vector's per-block frame floor (8) means the real cache can be far larger than you asked for: a run you think is out of core may be fully resident. Check faults and evictions.
- The store itself can fill: a put is refused -> code 8. Snapshots count against tier capacity.

**Layout**
- Pad partial tail pages and empty slots with sentinels, not zeros (zero-filled slots were read as thousands of phantom records).
- After reorganizing data between pages, reset destination slots before filling them, or stale entries reappear. Test long enough for the reorganization to actually run.
- A held span can cross several pages; harmless resident, wrong out of core.

## 2. Known open defects in the runtime (as of 2026-09)

- A file tier filled to capacity under gather-like refaulting can serve a page under the wrong identity. Failed gets/puts now trap instead of serving bad data; the root cause is open.
- Narrow sets combined with eviction of written pages can produce a wrong answer. Use per-block sets with headroom.
- A node presumed dead by the membership protocol can make remote reads return zeros. SWIM is off by default (timeouts now 300/150/3600 s); leave it off for benchmarks. For running and debugging procedures see the `clio-bench` skill.
- An undersized tier can produce wrong results instead of an out-of-space error. Size tiers to hold the data plus snapshots.

## 3. Validation that works

1. **Baseline at matched geometry** (same blocks x threads, same input). Mismatched geometry produced a false 1.29x instead of 2.46x.
2. **Resident gate**: cache >= data -> assert zero faults and zero evictions; answer equals the baseline.
3. **Out-of-core gate**: cache well below data -> **require evictions > 0** (a run that never evicted proves nothing) and the **same answer** as resident. Paging must be invisible to the result. Same across compilers and GPU vendors.
4. **Make results deterministic** so step 3 is meaningful: fixed-order sums, fixed-point accumulation, block-private accumulators reduced in order. Never gate on an atomically accumulated float checksum.
5. **Anti-vacuity**: negative controls that must change the answer (disable the exchange, disable remote reads); a generation refetch counter > 0 proving data crossed nodes; integrity counters (items processed vs expected, pages done per block).
6. **Layered gates** for iterative codes: the first step exactly against a reference, a short run within tolerance, then long-run invariants. Check finiteness: a NaN passes every tolerance comparison.
7. **Gate on the terminal line** and veto on fatal signatures -- "marker present" gates have passed crashed runs.
8. **Repeat and report spread** on file tiers (a parallel filesystem varied 3.2x across identical runs). Non-monotone behavior in cache size is a bug or noise, not a result.
9. Bit-exact float comparisons across compilers: `-fp-model=precise` (icpx), `-fmad=false` (nvcc) or explicit `fma` mirrored on the host.

## 4. Debugging

- `FatalReport()` / the SIGABRT print: code plus args a1..a7; codes 2/3/5 include a set census. Only the first fatal is latched: a code-7 root cause can hide the code-2 site.
- `EnableStats()` then `ReadStats`: faults, evicts, puts, `get_errors`, `put_errors`, `flush_skipped`, generation ok/stale/busy.
- "gen stall" printf every 100k rounds names the page and the generation wanted vs held.
- Bisect by iteration count, by cache slots, and by region; write a value that encodes page and offset to see whose bytes a frame holds.
- A read-only probe kernel separates write bugs from read bugs.
- A GPU coredump locates faults in `$_resume`. `cuobjdump` / `-Xptxas -v` for registers (one rung per TU). nsys on NVIDIA; VTune on Intel (GTPin crashes coroutine kernels).
- Promote every reproducer to a test.

## 5. Build and environment

- After changing a task struct, rebuild everything (no `--target`); install before rerunning (RPATHs ignore `LD_LIBRARY_PATH`).
- A build tree rebuilt across many commits can fail storage registration (`PutBlob` rc=11): configure a fresh build directory.
- Root-owned leftovers in `/tmp` give "Permission denied" on tier files: `chown` them.
- `CLIO_INIT` once per process; re-initializing hangs. Destroy vectors before finalize.
- `gpu: queue_depth` below the block count hangs (default 16).
- clang-CUDA: CUDA 12.x headers (13 fails to parse).
- Intel PVC: no device atomics on host USM; `clock64()` is 0; `__nanosleep` is a counted spin; icpx defaults to fast-math.
