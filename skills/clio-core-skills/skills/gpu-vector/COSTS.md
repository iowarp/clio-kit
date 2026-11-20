# COSTS.md -- where the time goes

Measured on an RTX 4070 Laptop GPU, NVIDIA A100 nodes, and Intel PVC nodes.
Workloads are described only by access pattern.

## The ten overheads, in order, and what to do about each

| # | overhead | size | what the kernel author does |
|---|---|---|---|
| 1 | **Fault round trip** | ~110 us per request vs ~6 us copy (256 KB page); 300-500 us at 4-8 nodes; ~2.7 ms at 32 | fetch ranges not pages; fill the 64-page / 8-range batch; fit the working set; 1-4 MB pages |
| 2 | **Park/relaunch rounds** | ~1.3-1.6 ms per round | wait on completion words (`ResumeWhenComplete`), never poll by relaunch; keep waits out of heavy kernels; overlap independent waits on separate blocks |
| 3 | **Paging code inlined into kernels** | hold path +120 regs; unbounded coroutine kernels 130-192 regs (1-7 blocks/SM) | `__launch_bounds__(256,4)`; compute in `noinline` helpers; two-phase |
| 4 | **Holds on the hit path** | ~40% of one resident kernel; a resident paged kernel 4x a native one with zero faults | hold per chunk/band/span, not per item; power-of-two pages |
| 5 | **Unneeded publish / write-back** | 2.5x and 7x in two kernels | flush only what peers read; nothing when resident; one flush for several ranges |
| 6 | **Frame and address-space codegen** | frame loop 1 GB/s vs 32 GB/s; generic address space 2x on Intel | nothing hot in the coroutine frame; raw pointer per page; global casts |
| 7 | **Cache and set geometry** | one huge set 9x; narrow sets 2-3x; hold-while-waiting wedge | 8-16 ways per set, one set per block, ~4x headroom; respect floors |
| 8 | **Cross-node traffic** | remote 317 us vs local 22 us; host-staged device bulk ~1.8 ms/send | node-local pages; decompose so nodes read their own pages; batch peer gets |
| 9 | **Slow tiers and migration** | parallel filesystem 2-11x at small pages; a migration ~271 us vs ~46 us saved | >= 1 MB pages on file tiers; no tier policy unless reuse >= 4 |
| 10 | **Persistence** | checkpoints can dominate a run; sync `Copy` 148 ms/GB; lazy `Copy` 0.4 ms | lazy `Copy` unless the source is overwritten before read; checkpoint rarely |

## Faults and batching

- One fault ~110 us round trip, ~6 us of it the 256 KB copy. Batching is the only thing that shrinks the round-trip term; that is why a fetch takes 64 pages.
- Fault path breakdown (512 KiB pages): staging p50 1 us; route + handler + storage read p50 349 us (95%); completion 16 us.
- A data-dependent gather went from 1 page per fetch to 32 per fetch: 23.3 s -> 4.0 s. Batching only pays if run-ahead pages survive until used: size the in-flight window `>= blocks * min(64, pages_per_block)` (wasted fetches 87% -> 0%).
- Host-side service latency is overlapped with the GPU: cutting it from 381 to 22 us did not move wall time. Optimize round trips and waits, not the handler.
- CPU side: a batch of <= 8 records runs inline (~80 us per record); larger batches dispatch (~560 us per child). Local puts run serially inline (dispatch was 7x slower).

## Rounds (park and relaunch)

- ~1.3 ms/round (one step: 41 rounds = 53.8 ms, 99% launch+sync, vs 0.5 ms of arithmetic); ~1.6 ms per resumed round in a flush microbenchmark.
- Poll-by-relaunch vs completion-word wait: 76,745 -> 15,217 rounds, 3.3 -> 0.66 relaunches per fault. A tag-0 await spun ~40 rounds per flush (98% of flush time).
- Moving waits out of the heavy kernel: 47.0 -> 27.8 ms/step, heavy-kernel GPU time 2940 -> 719 ms. Splitting independent flush and fetch waits (pay max not sum), then overlapping interior with boundary work: 25.5 -> 24.9 -> 22.97 ms/step.
- Mechanism overhead alone: a 256 KB paged stream as a coroutine is 1.31x plain; at 4x oversubscription it is invisible (PCIe dominates).

## Registers and occupancy

Register ladder (sm_89; measure one rung per translation unit or they merge):
raw loop 14 -> + init 22 -> + empty coroutine 28 -> lookup without coroutine 40 -> lookup inside coroutine 72 -> + full HoldPage 106.
Per verb (regs / instrs): HoldPage 43/1480; BeginFetch 66/3328; AwaitFetch 32/576; BeginFlush 66/1312; EndFlush 32/568.

- Unbounded C++20 coroutine kernels: 130 regs regardless of body (7 blocks/SM); some reached 192 (1 block/SM). `__launch_bounds__(256,4)` -> 64 regs, 50% occupancy, no spill. A register-cap sweep: 40 -> 6.24, 64 -> 5.58, 72 -> 5.97 ms/step.
- Occupancy did not fix a faulting kernel (4x occupancy, no gain): faults, not waves, were the limit.
- Transpiled coroutines: machinery +16 regs, scaffolding +16, inlined fetch+hold +122. Under a 64-register cap a heavy kernel spilled heavily.
- Frame hygiene: a loop in the frame 246 -> 31 ms once made `noinline`; lane 4096 -> 2048 B and smaller frames 401 -> 241 ms; lane stride 256 -> 64 B: 21%.

## Holds, sets, eviction

- A resident paged kernel with zero faults ran 4.0x a native one-sided-communication version; holds were ~40% of its main pass, and the arithmetic 21%.
- Chunked holds: 1 item per hold 9.06, 2 -> 6.43, 4 -> 8.87, 8 -> 13.13 ms/step.
- One 512-way set: 9x slower than per-block sets. Narrow hashed sets: 2-3x slower. Batched set scans (8 loads in flight): ~1.6x faster.
- Out of core vs cache slots in one kernel: 66 slots 6.5 ms/step; 48 -> 42.1; 40 -> 58.8; 36 -> 67.7; 32 wedged (hold-while-waiting; reserving the whole hold set first made 32 pass at 190 ms/step).

## Cross-node

- Task round trip: local 22 us, remote 317 us (per hop: send queue 45-65, receiver 30-60, transport 5, wire 5-10 us).
- Device frames sent as bulk were host-staged at ~1.8 ms/send; `neighborhood: 1` and one shared device queue took one kernel 6.5 -> 2.9 s.
- Host-staged exchange 18.8 ms vs GPU-direct 1.6 ms for the same bytes. A one-sided per-block pull moved 3x the necessary bytes.
- Done right, paged exchange beat the message-passing baselines: slowest-rank communication 33.8 vs 336.8 ms and 3.4 vs 47.6 ms in two codes.

## Tiers, migration, placement

- Migration = read + write staged through the host, ~271 us (256 KB); fault saved by a faster tier ~46 us. Every prefetch policy tried was slower than none; a page needs >= 4 reuses in the fast tier to repay its migration.
- Fill cost of a fast tier: `(tier_bytes / page_bytes) x 271 us`. Compare with the run time before building a policy (104 ms to fill vs a 45 ms run cannot win).
- Tier composition at fixed capacity: a streaming read ran 2.0-2.2 s from DRAM, 4.4-5.2 s DAOS-heavy, 23 s Lustre-heavy; codes that write back every iteration paid 2.9-5.4x below DRAM; very large pages (tens of MB) made Lustre beat DAOS. Parallel-filesystem repeats varied 3.2x.
- Put path: 1 MB into pageable shared memory 429 us vs pinned 83 us.

## Page and cache size

| page (streaming read, 4 nodes, cache held at 512 MB) | DRAM-only | Lustre-heavy |
|---|---|---|
| 64 KB | 11.42 s | 22.64 s |
| 256 KB | 3.19 s | 4.94 s |
| 1 MB | 2.62 s | 3.33 s |
| 4 MB | 2.53 s | 3.33 s |

Cache sweep 1/2/4/8 GB: a streaming read was flat until the data became resident (then a cliff down); a sliding-window kernel improved 23%; a sweep-heavy training kernel was fastest at the smallest cache.

## Checkpoints and compression

- `Copy` lazy 0.36-0.39 ms; sync 9.5 / 37 / 148 ms at 64 MB / 256 MB / 1 GB. Per-step checkpoint 10.4 ms lazy vs 36.7 ms sync.
- A resident checkpoint through the vector was ~11 ms per 18.7 MB vs 0.4 ms device-to-host (it went through one block).
- Batched flush reaches 10.9 GB/s at 16 blocks (0.96x raw memcpy); scalar puts cap at ~9 GB/s.
- Compression on the fault path: 8x worse when the data fits (27.6 -> 3.0 GB/s), 2.4x better at 4x oversubscription (2.55 -> 6.10 GB/s). Compress only when it makes the data fit.

## What closed the resident gap to native baselines (in order of payoff)

1. Skipping publishes and write-backs nobody reads (2.5x-7x).
2. Batching fetches of everything a step needs into one call.
3. Global address-space casts on Intel (2.2x -> 1.12x).
4. Bounding registers to 64 and moving compute out of the coroutine body.
5. Holding per chunk/band instead of per item; a sliding window instead of refetching.
6. Matching the baseline's geometry before comparing.
With these, paged codes reached 0.93-1.19x of message-passing baselines on four very different access patterns.
