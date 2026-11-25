# API.md -- the current gpu_vector contract

Source of truth: `context-transfer-engine/adapter/gpu_vector/include/clio_cte/gpu_vector/{device_vector.h,gpu_vector.h,page.h,prefetch.h}`
and `context-runtime/include/clio_runtime/gpu/{yieldable.h,yield_stack.h,yield_coro.h}`.
When a comment in the tree disagrees with this file, check the code.

Namespaces used below: `gv = clio::cte::gpu_vector`, `gy = clio::run::gpu`
(yield machinery), `u32/u64 = clio::run::u32/u64`.

## 1. Host side

### Runtime

```cpp
// Minimal server config (write it to a file, point CLIO_SERVER_CONF at it).
networking:
  port: 9431
runtime:
  num_threads: 4
  queue_depth: 4096
gpu:
  queue_depth: 4096          # default 16 hangs at 128 blocks
compose:
  - mod_name: clio_bdev
    pool_name: "ram::chi_default_bdev"
    pool_query: local
    pool_id: "301.0"
    bdev_type: ram
    capacity: "512MB"
  - mod_name: clio_cte_core
    pool_name: cte_core
    pool_query: local
    pool_id: "512.0"
    storage:                  # at least one target, or the first writeback hangs
      - path: "ram::gv_tier"
        bdev_type: "ram"
        capacity_limit: "256MB"
        score: 1.0
    dpe:
      dpe_type: "max_bw"
```

```cpp
ctp::SystemInfo::Setenv("CLIO_SERVER_CONF", "gpu_vector_test.yaml", 1);
REQUIRE(clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true));  // once per process
std::this_thread::sleep_for(std::chrono::milliseconds(500));
REQUIRE(clio::cte::core::CLIO_CTE_CLIENT_INIT());
clio::run::IpcManagerGpuInfo gpu_info =
    CLIO_CPU_IPC->GetGpuIpcManager()->GetGpuInfo(0);
// ... vectors live and die here ...
// destroy every Vector BEFORE clio::run::CLIO_RUNTIME_FINALIZE()
```

To make data spill, add slower tiers (a second `ram::` with a small
`capacity_limit`, or a `file` target) and make the fastest tier smaller than
the data. Pages are written at blob score `kVectorBlobScore = 0.5`: a tier
scored above 0.5 is never chosen for them on its own (see ORGANIZER.md).

### `gv::Vector<T>`

```cpp
Vector(const std::string &tag_name,        // CTE tag; same name = same data (across nodes too)
       const std::vector<int> &gpu_ids,    // e.g. {0}; one device view per GPU
       u64 page_bytes,                     // elems_per_page = page_bytes / sizeof(T)
       u32 nblocks,                        // >= logical blocks in any grid using it
       u32 set_size,                       // frames per associative set
       u64 num_elems,                      // logical length
       clio::run::PoolId storage_pool_id = PoolId::GetNull(),  // null = CTE pool 512.0
       int compress_lib = 0, int compress_preset = 1,          // 0 = none
       u32 nsets = 0,                      // 0 -> nblocks
       u32 capacity_pages = 0);            // 0 -> nsets*set_size; = the cache size in pages
```

- The cache is **one per vector per GPU, shared by the whole grid**. Page `p` lives in set `Hash(p) % nsets`, in any of its `set_size` ways.
- Cache bytes = `capacity_pages * page_bytes`. Everything a block pins at the same moment must fit in its pages' sets, or you eventually trap with code 5.
- Geometry that works: one set per block (`nsets = nblocks`), `set_size` 8-16 (floor 8) with ~4x headroom over the pins per set. Not one huge set (a lookup scans its whole set: one 512-way set was 9x slower than per-block sets), not narrow sets (collisions, 2-3x slower).
- `DeviceVector<T> GetDevice(int gpu_id)` -- pass by value to kernels.

Other host calls:

| call | does |
|---|---|
| `Preload(const T*, n)`, `PreloadPages(lo, hi, fill)` | host data into the store |
| `Download(T*, n)`, `DownloadPages(lo, hi, sink)` | store back to host |
| `Prefetch(pg_lo, pg_hi, gpu_id=0)` | host copy of pages into GPU frames; only on a fresh or just-cleared vector |
| `FlushResidentToCte()` | host writeback of every resident frame's valid range |
| `ClearCache(gpu_id=0)` | drop every frame. **Unflushed data is discarded.** Never on a cache that holds the only copy of live data. |
| `Copy(new_name, sync=false)` | checkpoint: flushes resident frames, then a copy-on-write snapshot (lazy: ~0.4 ms). A lazy copy returns *later* bytes if the source is overwritten before the copy is read -- use `sync=true` (148 ms/GB) then. Needs the checkpoint module linked. |
| `MaterializeAll(inflight=32)` | force a lazy `Copy` to real bytes, server side |
| `EnableStats()`, `ReadStats(gpu)`, `ResetStats()` | counters (off by default). `ResetStats` only zeroes the first 5 of 17. |
| `PrefetchNow(hints)`, `RegisterPrefetcher`, `SetPrefetchPolicy`, `YieldObserver()` | tier rescoring only -- never makes pages GPU-resident. See ORGANIZER.md. |
| `FatalReport()` | the latched device-fatal note as a string |

## 2. Device side: `gv::DeviceVector<T>`

**Every verb is block-collective.** Thread 0 does the submission internally.

Setup and geometry:

```cpp
void Init(u32 block);                 // first thing in the kernel: v.Init(yv.Block())
u64 size(); u64 ElemsPerPage(); u64 NumPages(); u64 PageBytes();
u64 PageOf(u64 off);                  // off / epp
u64 PageLo(u64 off);                  // first element of off's page
u64 PageSpan(u64 off, u64 count);     // elements from PageLo(off) covering whole pages through off+count
static constexpr u32 kMaxFetchRanges = 8;
```

### The verbs (C++20 `co_await` form shown; other forms in sec 3)

```cpp
YCoroTask BeginFetch(u64 gen, u64 off0, u64 n0, u64 off1, u64 n1, ...);  // submit
YCoroTask AwaitFetch();                                                  // park until landed
YCoroTask Fetch(u64 gen, u64 off0, u64 n0, ...);                         // both
YCoroTaskT<Held<T>> HoldPage(u64 off, u64 count, bool write = false);
void UnpinRange(u64 off, u64 count);
void UnpinRange2(u64 off0, u64 n0, u64 off1, u64 n1);
YCoroTask BeginFlush(u64 gen, u64 off, u64 count, ...);                  // submit
YCoroTask EndFlush();                                                    // park until landed
YCoroTask Flush(u64 gen, u64 off, u64 count, ...);                       // both
```

**Fetch**
- Takes `(offset, count)` element pairs. **Pairs past the 8th are silently dropped**; zero-count pairs are skipped.
- Claims and pins every page the ranges touch. A newly claimed frame is always fetched **whole** (clipped to the vector's end), even if you named a slice.
- **At most 64 pages per call.** Pages beyond 64 are silently not claimed; a later hold on them traps (code 2).
- Missing pages go out as one batched request; if all are resident, nothing is submitted and the block does not park.
- One fetch in flight per block: `BeginFetch` first retires the previous one. There is no in-block double buffering.
- Full set: spins 4096 times, then yields the round and retries; traps (code 5) after 65536 stalled rounds.
- `gen = 0`: any version. `gen = g > 0`: the runtime serves it only once the page's bytes carry generation >= g (waits up to 10 s, then refuses -> code 7). A resident frame older than g is refetched. The `co_await Fetch` form additionally waits until every named page shows >= g; **`CoFetch` and `MFetch` do not have that wait loop.**

**HoldPage**
- Does not fetch. If the page is being fetched (by you or a peer block) it waits; if it is absent it traps (code 2).
- Read hold: traps (code 3) unless `[off, off+run)` is inside the frame's valid range. Write hold: skips the check and marks the range valid -- two disjoint write holds mark the gap between them valid too, so fetch before a partial write.
- Returns `Held<T>` (coroc: `PageRef<T>`): `ptr()` = pointer at element `off`; `run()` = `min(epp - off % epp, count)` (a hold never crosses a page: loop with `i += h.run()`); `operator[](u64 abs_off)` takes the **absolute** element offset; `begin_off()`; `Rescore(float)` sets the frame's eviction rank (higher = keep).
- The guard does **not** own the pin; destroying it does nothing. `UnpinRange` releases.

**Eviction**: only frames with `pins == 0` and no fetch or flush in flight; lowest score first, then LRU (LRU is inert on SYCL: `clock64()` is 0). **No I/O.**

**Flush**
- `(off, count)` pairs, at most 8, **byte-exact** per range: disjoint slices of one page from different blocks are safe.
- Up to 64 records per call; more traps (code 6, "split the range").
- **Pages that are not resident are silently skipped** (counted in `flush_skipped`) -- that is lost data. Flush before unpinning.
- One flush in flight per block; `BeginFlush` retires the previous one. Do not modify a range between `BeginFlush` and `EndFlush` (the put reads the frame asynchronously).
- `EndFlush` returns once the put has landed: other blocks and nodes can read it. `gen > 0` stamps the written bytes with that generation.
- A refused put (store full) traps (code 8).

`UpdateRange(off, count, from, write, gen)` is just `Fetch + HoldPage`; `from` is ignored.

## 3. Kernel forms and the relaunch driver

Three spellings of the same verbs. Pick by target:

| form | when | compiler |
|---|---|---|
| **coroc** (`CO_AWAIT`, `Co*` verbs) | **benchmarks; anything that must also run on Intel (Aurora)** | source transpiler, then nvcc (`-DCLIO_COROC -rdc=true -maxrregcount=64`) or icpx |
| C++20 (`co_await`, `YCoroTask`) | unit tests, CUDA-only code | clang-CUDA (`-DCLIO_CORE_ENABLE_CUDA=ON -DCLIO_GPU_CLANG=ON`); cannot compile for SPIR-V |
| macro (`CLIO_YCALL`, `M*` verbs) | SPIR-V without the transpiler | any |

### coroc form (preferred for benchmarks)

```cpp
// A coroutine is a plain function; the transpiler appends a context param.
CTP_GPU_FUN CLIO_COROC_INLINE void ScaleCoro(gv::DeviceVector<float> v,
                                             u64 per, u64 epp, float a, u32 block) {
  const u64 base = static_cast<u64>(block) * per;          // this block's slice
  for (u64 off = 0; off < per; off += epp) {
    const u64 n = (off + epp <= per) ? epp : (per - off);   // slice is page-aligned
    CO_AWAIT(v.CoFetch(0, base + off, n));
    auto h = CO_AWAIT(v.CoHoldPage(base + off, n, /*write=*/true));
    float *pg = h.ptr();                     // resolve a raw pointer ONCE per page
    ScalePage(pg, n, a);                     // noinline compute, no suspends inside
    __syncthreads();
    CO_AWAIT(v.CoFlush(0, base + off, n));   // write back exactly what we wrote
    v.UnpinRange(base + off, n);
  }
}

__global__ GV_LAUNCH_BOUNDS void ScaleKernel(clio::run::IpcManagerGpuInfo info,
                                             gv::DeviceVector<float> v, u64 per, u64 epp,
                                             float a, gy::YieldableView<> yv,
                                             gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.Init(yv.Block());
  __syncthreads();
  CLIO_COROC_RUN(yv, ys, ScaleCoro(_cy, v, per, epp, a, yv.Block()));  // write _cy here
}
```

coroc rules (violations either fail to transpile or silently do not wait):
- The source **must** be transpiled; untranspiled, `CO_AWAIT` is a no-op and nothing waits.
- Everything live across a `CO_AWAIT` must be trivially copyable (`static_assert`): use `PageRef`, never a named `Held`.
- One declarator per statement; no default arguments (spell `write` out); no `CO_AWAIT` in expression position, range-for, or directly in a lambda; awaiters never capture `this` or a local's address.
- `CoHoldPage(off, n, write)` has no default for `write`.

### C++20 form (tests; clang-CUDA only)

```cpp
__device__ gy::YCoroMain FillCoro(gv::DeviceVector<u32> v, u64 n) {
  for (u64 i = 0; i < n;) {
    u64 run = 0;
    {
      co_await v.Fetch(0, v.PageLo(i), v.PageSpan(i, 1));
      auto h = co_await v.HoldPage(i, n - i, /*write=*/true);
      run = h.run();
      for (u64 k = threadIdx.x; k < run; k += blockDim.x) h[i + k] = u32((i + k) * 7 + 1);
    }
    co_await v.Flush(0, i, run);
    v.UnpinRange(v.PageLo(i), v.PageSpan(i, 1));
    i += run;
  }
}
__global__ void FillKernel(clio::run::IpcManagerGpuInfo info, gv::DeviceVector<u32> v, u64 n,
                           gy::YieldableView<> yv, gy::YieldStackView ys) {
  CLIO_GPU_INIT(info, nullptr);
  v.Init(yv.Block());
  gy::YieldTlsPublish(ys, yv.Y(), yv.Block());   // CLIO_COROC_RUN does this itself
  __syncthreads();
  CLIO_YCORO_RUN(FillCoro(v, n));
}
// launch with dynamic smem: Kernel<<<g, b, CLIO_YIELD_SMEM_BYTES>>>(...)
```
Nested suspending helpers return `gy::YCoroTask` / `gy::YCoroTaskT<V>`. Lane bytes 8192 (frame overflow traps 103). Never build with `-DCLIO_YIELD_SYNC` (deadlocks by construction).

### Macro form (SPIR-V fallback)

```cpp
CLIO_YFRAME();
CLIO_YLOCAL_INIT(u64, off, 0);
CLIO_YLOCAL(gv::Held<float>, h);
CLIO_YBEGIN();
for (; off < per; off += epp) {
  CLIO_YCALL(v.MFetch(0, base + off, n));
  CLIO_YCALL(v.MHoldPage(&h, base + off, n, /*write=*/true));
  /* compute */
  CLIO_YCALL(v.MBeginFlush(0, base + off, n));
  v.UnpinRange(base + off, n);
}
CLIO_YCALL(v.MEndFlush());
CLIO_YEND();
```
Only `CLIO_YLOCAL*` values survive a suspend; one yield per source line; nesting <= 8 (code 101). A `CLIO_YLOCAL` read in a hot loop costs ~29x -- copy it to a plain local first. `CLIO_YKERNEL_ENTER` does **not** save/restore `CLIO_SHARED_PERSIST`.

### State across a park

A park is a kernel exit. Plain `__shared__` contents are garbage afterwards
(C++20 coroutine locals survive in the frame; coroc locals survive if trivially
copyable). For block-uniform tables use `CLIO_SHARED_PERSIST(Type, name)`: one
per coroutine, <= 1024 B, declared before the first suspend (C++20 and coroc
runners only).

### The host driver

```cpp
gy::Yieldable<> drv(nblocks, nthreads);                 // nthreads == blockDim.x
gy::YieldStack stack(nblocks, nthreads, /*bytes_per_lane=*/8192);
drv.Reset(); stack.Reset();                             // REQUIRED when reusing: otherwise the
                                                        // driver still reads "done" and returns
                                                        // an instant, empty success
const u32 rounds = drv.RunToCompletion(
    [&](dim3 g, dim3 b, gy::YieldableView<> view) {
      ScaleKernel<<<g, b>>>(gpu_info, vec.GetDevice(0), per, epp, a, view, stack.View());
    },
    [] {}, /*max_rounds=*/2000000, gv::ResumeWhenComplete);
if (drv.HitRoundCap()) { /* livelock: fail loudly */ }
```
- `gv::ResumeWhenComplete` relaunches a block only when the word it waits on flips. Never poll by relaunching everything (3-5x more rounds).
- In the 4-argument form the service callback's return value is ignored.
- `gy::CoroRunner<> r(nblocks, nthreads, lane_bytes); r.Run(launch, max_rounds); r.HitRoundCap();` is an equivalent wrapper.
- A generation published and demanded in the **same launch** is not guaranteed to complete: publish and consume in separate runs.

### Registers

`GV_LAUNCH_BOUNDS` (`benchmark/gv_launch_bounds.h`) = `__launch_bounds__(256, 4)` -> 64 registers, 50% occupancy; launching with > 256 threads then fails at runtime. Unbounded coroutine kernels land at 130-192 registers (1-7 blocks/SM). With nvcc match it with `-maxrregcount=64`; CMake has `clio_coro_regcap(tgt)`.

## 4. Build

CUDA (C++20 form): configure with `-DCLIO_CORE_ENABLE_CUDA=ON -DCLIO_GPU_CLANG=ON -DCLIO_CORE_ENABLE_TESTS=ON` (clang cannot parse CUDA 13 headers; use 12.x). Link:
`clio_run_cxx clio_run_cxx_gpu clio_admin_client clio_bdev_client clio_cte_core_client clio_cte_core_runtime clio::cte::gpu_vector ctp::cuda_cxx Threads` (+ `clio_cte_checkpoint_*` for `Copy`). Include dirs: `context-runtime/include`, `context-runtime/test`, `context-runtime/modules/{admin,bdev}/include`, `context-transfer-engine/core/include`, `context-transfer-engine/adapter/gpu_vector/include`.

coroc form: `context-runtime/coroc/build.sh` builds the transpiler; `benchmark/build_newcoro.sh <dir> [name]` transpiles and builds with nvcc (`-std=c++20 -O2 -rdc=true -maxrregcount=64 -DCLIO_COROC`). SYCL: `build_newcoro_sycl.sh`; Aurora: `AOT=1 build_newcoro_aurora_all.sh` (`spir64_gen -device pvc`).

SYCL on Intel: exactly one `-fsycl` TU defines `CLIO_SYCL_KERNEL_TU 1` before every clio include and is built as a SHARED library; call `gy::SyclInitBlockIpcManagers(max_blocks, gpu_info)` once; kernels are `q.parallel_for(nd_range{...}, [=](sycl::nd_item<1>){ auto dev = v; dev.Init(vw.Block()); ... })`. Use `-fp-model=precise` (icpx defaults to fast-math) and `sycl::address_space_cast<global>` on pointers at each use.

After changing any task struct, rebuild everything (no `--target`); install before rerunning (RPATHs ignore `LD_LIBRARY_PATH`).

## 5. Fatal codes

Latched into a pinned-host channel (first writer wins); `FatalReport()` returns it, and a SIGABRT handler prints it. On SYCL a "segfault at 0x0" is the trap working.

| code | name | cause |
|---|---|---|
| 1 | InitBlock | `Init(block >= nblocks)` |
| 2 | NotResident | hold of a page never fetched: forgot to fetch, fetch > 64 pages, or unpinned before holding. A page fetched and then evicted by a peer before the hold is re-fetched (up to 64 times, since abf541997); trapping anyway means the set cannot keep it -- see code 5 |
| 3 | NotCovered | read hold outside the frame's valid bytes |
| 4 | Unbound | verb before `Init` |
| 5 | SetFull | a set full of pinned frames through every retry: cache/sets too small, or pins leaked |
| 6 | FlushSplit | flush needing > 64 records |
| 7 | GetFailed | runtime refused a get -- usually a generation nobody published (10 s timeout) |
| 8 | PutFailed | runtime refused a put -- store full (prints as "unknown") |
| 101/102/103 | yield depth / frame / coroutine frame | raise `bytes_per_lane` for 103 |

Hitting the round cap is not a trap: a stderr line plus `HitRoundCap()`.
