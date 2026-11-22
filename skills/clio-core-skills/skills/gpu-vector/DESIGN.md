# DESIGN.md -- designing kernels and data layouts for the paged vector

General rules, each with the measured effect behind it. The paged kernel is
usually **latency-bound, not work-bound**: removing the arithmetic entirely
from one paged kernel moved its time by 3%. Design for fewer faults, fewer
waits, and fewer holds.

## 1. Ownership and layout

1. **Keep state on the device for the whole run.** The host configures, loads initial data, and reads back a few scalars; no bulk state crosses PCIe per iteration. Plugging a paged kernel into a host-owned code that shipped state in and out every step cost 12.5x; keeping the state device-canonical brought the same computation to 1.5x of an optimized native code.
2. **A page is a work unit, and every page has exactly one writer.** Choose the layout so a block's range covers whole pages and no logical record straddles a page boundary (fixed-size records, rows, slices, bins). Enforce it on the host and refuse sizes that break it.
3. **Byte-disjoint writers; never flush bytes you did not write.** Several writers each flushing a whole shared page clobber each other with every counter reading clean (over half of one accumulated array vanished this way). Within a node, disjoint slices of one page are safe *if each writer flushes exactly its slice*. **Across nodes, make ownership whole pages**, because a fetch transfers whole pages: a node's range must end on a page boundary.
4. **Pull, don't push.** Invert any scatter into a gather in which each destination's owner reads its sources; a scatter in which two blocks wrote one page produced fatal errors and silent corruption out of core. Mirror case: when the randomly accessed side can stay resident, stream the paged side sequentially and scatter into the resident structure.
5. **Keep index and metadata resident.** Counts, offsets, lookup tables, destination maps, page-pointer tables: plain device arrays. Paging the index of the paging is circular.
6. **Do not page small, hot, shared structures.** Accumulators, parameters and anything every block touches: keep them resident, replicated per node or per block. One shared hot page deadlocked under generation demands and served stale replicas.
7. **Coalesce like any GPU code.** Transposed, padded layouts (entry k of item s at `k*n + s`) give warp-contiguous reads; per-thread contiguous runs scatter a warp over 32 cache lines and lost to a brute-force alternative.
8. **Block-private accumulation beats global atomics.** Per-item atomics into a small global array serialized the device once most updates hit the same few entries (iteration time grew 20 s -> 116 s); accumulate per block (or per page in shared memory) and merge once.

## 2. Working sets and cache sizing

9. **Make working sets static and computable from geometry.** The pages a block needs next must be arithmetic, not discovered by scanning. Then the fetch is one call and the cache floor is known.
10. **Measure the cache floor, enforce it, refuse below it.** The floor is the pages a block holds simultaneously x blocks sharing the cache, plus headroom. Also: the pins that land in one *set* must fit in `set_size`, and the vector enforces a per-block frame floor (8), so **real cache = max(requested, blocks x 8 x page)** -- size in bytes and check.
11. **Reserve before you hold.** If a block holds some pages while waiting for others, blocks deadlock each other's frames (hold-while-waiting livelock: at one cache size fine, 25% smaller wedged). Fetch the whole hold set in one call before the first hold, in a fixed order, and park instead of spinning.
12. **Streaming does not need cache; reuse does.** A pure streaming pass was flat or faster at 1/8 the cache; a kernel with a sliding reuse window gained 23% from 8x more cache. Size the cache for reuse distance, not data size. Sweep cache in **bytes**, and hold it fixed when sweeping page size, or you sweep both at once.
13. **Page size: 1-4 MB for streaming.** 64 KB pages were 4.4x slower than 1 MB; 4 MB only 2% better than 1 MB. Large pages also shrink slow-tier penalties (2.0x -> 1.3x). With a very small cache, smaller pages with more blocks can win (more pages in flight): measure. Prefer powers of two (a hold once cost 316 div/mods).

## 3. Placing fetch, hold, and flush

14. **Fetch ranges, not pages, and batch them.** One `Fetch` takes up to 8 ranges and 64 pages for one round trip: fetch everything a step needs together. A fault is ~110 us of round trip for ~6 us of copy.
15. **Amortize holds.** Hold count, not hold size, is what you pay (holds were ~40% of one resident kernel's time). Hold contiguous spans instead of pieces, hold once per chunk of work (a chunk of 2 rows beat 1 by 1.4x and 8 by 2x: amortization vs lost parallelism), hold a whole band and keep sums in registers, and size pages so one hold is one work unit.
16. **Sliding windows fetch each input once.** For a kernel that reuses a window of pages, keep a ring indexed modulo the window, fetch only the page entering and unpin the page leaving (1.5x over re-fetching the window per output).
17. **Resolve one raw pointer per page** and compute through it. Reading through the handle (a coroutine local living in global memory) reloaded its fields on every access: ~2x slower.
18. **Flush at the write site, ranged, before unpinning, whenever eviction is possible.** Eviction does no I/O; without write-site flushes one simulation silently undid its updates while its sanity check still passed.
19. **When everything fits, do not publish at all.** The frame is the truth; use generation 0 and skip flushes nobody reads (removing them gave 2.5x in one kernel and 7x in another; publishing initial data that nothing read was tens of GB of puts that hung setup). Keep one code path, with the regime chosen by the host.

## 4. Overlap and waits

20. **There is no in-block double buffering** (one fetch slot per block; an in-block prefetch attempt ran at 0.86x of serial). Get overlap from:
    - independent waits on separate blocks (pay the max, not the sum);
    - doing interior work first and boundary work last while an exchange is in flight;
    - a separate runner with its own reserved tables for publish/exchange;
    - async flush on release, awaited later.
21. **Never park the heavy kernel.** Moving a remote fetch out of a compute kernel into a small side kernel took one code from 47 to 25.5 ms/step.
22. **Two-phase when compute is heavy.** Phase 1 (coroutine): fetch and pin every page of the step, write their pointers to a device table. Phase 2: a *plain* kernel -- often the baseline's own -- computes through the table at full occupancy. Phase 3 (coroutine): flush and unpin. Two-phase reached 1.19x of the baseline where the in-kernel version was 2.0-2.6x. Out of core, process the domain in bands sized to the cache.

## 5. Multi-node

23. **Decompose so peers need nothing** when possible; otherwise **exchange partials or slices instead of pulling peer pages** (pulling a whole peer structure moved 3 GB per step; per-node partials combined in order fixed it).
24. **The generation is the barrier.** Producers flush peer-visible data with generation `g`; the consumer's fetch of a *peer's* page demands `g`. Delete explicit barriers and invalidations (an explicit barrier cost more than the compute; per-step invalidation stopped eviction from ever engaging).
    - Demand only on the **consumer**, only on **peer** pages. Own pages and outputs use 0: a page's generation is set by the fetch that delivers it, so demanding one on a page you wrote locally hangs.
    - Initial data publishes as generation 1; iteration s reads s+1 and publishes s+2.
    - Never `ClearCache` a shared cache mid-run: it holds the only copy.
25. **Publish only what peers read** (boundaries, not whole partitions: a whole-partition flush was 150x the baseline).
26. **Pin boundary data like ghost buffers** and refresh it in place; evicted boundary pages refault into newer generations.
27. **Keep pages node-local** (`neighborhood: 1`) and decompose so each node reads its own pages: remote round trip ~317 us vs ~22 us local.

## 6. Codegen, registers, occupancy

28. **Keep the coroutine body thin.** The fetch+hold path inlines ~+120 registers. Put compute in `noinline` plain functions over raw pointers; a loop left in the coroutine frame wrote at ~1 GB/s vs 32 GB/s.
29. **Bound registers**: `__launch_bounds__(256, 4)` / `-maxrregcount=64` (64 was the swept optimum). Occupancy is rarely the bottleneck, though: 4x the occupancy did not speed up a faulting kernel.
30. **Block-uniform tables go in `CLIO_SHARED_PERSIST` or shared memory, not the frame** (each frame read cost ~6 dependent global loads). Recompute geometry rather than storing it in the frame.
31. **On Intel GPUs (SYCL)**: `sycl::address_space_cast<global>` on every pointer at each use (generic address space cost 2x), larger register files for register-heavy kernels, split large suspending loops into `noinline` stage coroutines (compiler crashes), avoid SIMD32.
32. **Grid sizing**: 64 blocks x 64-256 threads was the sweet spot for paged kernels on NVIDIA; Intel PVC wanted 1024-2048 blocks. Always compare against the baseline at the *same* geometry (geometry alone was worth 1.7x in one comparison).

## 7. Determinism, so paging bugs are visible

33. **Make the answer independent of paging and block count**: fixed-order sums, fixed-point accumulation, block-private accumulators reduced in block order. Then "resident and out-of-core runs agree exactly" is a real gate (PITFALLS.md sec 3). Do not gate on a float checksum accumulated with atomics: it is not reproducible run to run.
