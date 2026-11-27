# Evals - clio-cte-caches

Scenarios recorded 2026-11-21 against clio-core `dev` (bd1dc3183). Rubric is
pass/fail per bullet. Run each WITHOUT the skill to capture the gap, then WITH
it to confirm the gap closes.

## S1 - lost writes after a crash

Setup: Prompt: "I pip installed iowarp-core, wrote a few MB through the CTE
Python client, then the node rebooted. The data is gone. The config is the
default one. Why, and how do I make writes survive?"

Expected:

- Says the default write-back cache (567.0) acks after the write to the
  volatile DRAM core, before persistent storage.
- Names the flush triggers: `flush_period_ms` (10000 ms), `flush_fill_pct`
  (80), `flush_dirty_mb` (256 MiB), `SyncTag`/fsync, graceful `clio-run stop`.
- Mentions the WAL fsync period (`flush_metadata_period_ms`, 5000 ms) for
  power loss.
- Offers fsync/`SyncTag` (noting Python has no SyncTag binding) or switching to
  write-thru, and does not claim lowering `flush_period_ms` to 0 helps (0
  disables the periodic flush).

## S2 - composing write-thru

Setup: Prompt: "Replace the write-back cache in my compose with write-through.
I also use the checkpoint module."

Expected:

- Uses `mod_name: clio_cte_core_persist_write_thru` with `next_volatile` and
  `next_persistent`.
- Flags that the default id 566.0 collides with checkpoint and sets an
  explicit free `pool_id` (e.g. `"569.0"`), re-pointing the shm cache's
  `next_pool_id`.
- Sets the shm cache's `invalidate_on_flush: false` and drops `flush_pool`.
- States that `flush_*` keys and `fallback_to_persistent` are ignored, and
  that every put now pays persistent-chain latency.

## S3 - stale size on another node

Setup: Prompt: "Two nodes, default compose. Node A appends to a blob; node B
calls GetBlobSize right after and gets the old size for several seconds. Bug?"

Expected:

- Explains `md_only: true` + `invalidate_on_flush: true`: B's cached metadata is
  invalidated only when A's write-back flush lands at the blob's home.
- Also notes that write-back buffers A's write on A, and reads go to the owner.
- Gives fixes: `SyncTag` on A, lower `flush_period_ms`, or
  `invalidate_on_flush: false`.
- Checks that write-back has `flush_pool: "568.0"`; without it the staleness
  has no bound.

## S4 - should I use clio_cte_cache?

Setup: Prompt: "Should I add clio_cte_cache (563.0) on top of my default
chain to speed up reads on 8 nodes?"

Expected:

- Says no for the default two-core layout: its copies go to core 512.0 (the
  disk core there), and write-back already keeps DRAM copies.
- Names when it fits: one core at 512.0 with a memory tier, several nodes,
  remote reads dominate, often with the compressor.
- Mentions `CLIO_CTE_POOL=563.0` is needed when no shm cache fronts it.
