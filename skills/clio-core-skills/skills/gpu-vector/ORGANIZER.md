# ORGANIZER.md -- tier-placement policies for gpu_vector pages

Read COSTS.md first: most placement policies lose, and its fill-cost rule
tells you which before you write one.

CLIO places each page (a blob in the CTE) on a storage tier by **score**. A
data organizer is a policy that rescores pages over time so that what the
application is about to touch sits in a fast tier and what it is done with
sinks. The right policy follows from the access pattern; a policy that is
right for one pattern is wrong for another.

## Score direction -- read this twice

The `MaxBwDpe` placement engine keeps tiers whose `score <= page score` and
prefers the highest-scored of those. So:
- **higher page score = faster tier**;
- a tier scored above a page's score is *not eligible* for it;
- every gpu_vector page is written at `kVectorBlobScore = 0.5`, so with a
  typical config (fast tier 1.0, host RAM 0.2, file 0.0) pages **never reach
  the 1.0 tier on their own**. Something (a hint or an organizer) must promote
  them.

Rescoring clamps to [0, 1] and is a no-op below `score_difference_threshold`
(default 0.05). Each rescore still costs a task: do not issue ones you know
are below it.

## The three levers

1. **A `DataOrganizer`** in the CTE core factory: a periodic coroutine that
   sees every page's stats (`OrganizerBlobStat`: current score, last read,
   last modified, access count, size) and rescores them
   (`ReorganizeBlobInternal`).
2. **`ReorganizeHint(int)`** from host code: the application tells the
   organizer which *phase* it is in. The integer is opaque to the core; the
   organizer reads it with `server->OrganizerHint()`.
3. **Device-side prefetch**: a kernel raises the score of pages it is about to
   need (`PodMultiScore`, `PrefetchPolicy` in `gpu_vector/prefetch.h`). For
   lookahead within a pass; use the organizer for decisions across passes.

Where things are: `context-transfer-engine/core/include/clio_cte/core/data_organizer/data_organizer.h`
(interface, factory, `OrganizerBlobStat`), `core/src/data_organizer/data_organizer.cc`
(factory registration), `frecency_organizer.{h,cc}` (reference implementation:
recency blended with saturating frequency), `Runtime::CollectOrganizerBlobStats`,
`Runtime::ReorganizeBlobInternal`, `Runtime::OrganizerHint` (`core_runtime.{h,cc}`).
Config: `organizer`, `organizer_tasks`, `organizer_period_ms` under the
`clio_cte_core` compose entry.

## Workflow

**1. Extract the access pattern (before any code).** For each kernel that
touches paged data, from the source, not a guess: order (sequential, strided,
pseudo-random, data-dependent); reuse distance (once per pass, every pass,
within a sliding window); read/write mix; phase boundaries. Write a
*simplified kernel* -- the few lines stating which parameters drive the
address sequence. If you cannot write it, you do not understand the pattern.

**2. Research the data (only when the data decides the order).** For
graphs, sparse structures and other data-dependent orders, the order is a
property of the dataset: read about its locality and structure, and say so
when a policy relies on an assumed distribution. Layout is often a
precondition: without a locality-preserving ordering, no page-level policy
can find a signal.

**3. Design and implement.** Custom organizer when the policy can be computed
from page stats plus the phase hint:

```cpp
clio::run::TaskResume MyOrganizer::Reorganize(Runtime *server, clio::run::u32 replica) {
  const clio::run::i32 phase = server->OrganizerHint();   // 0 = no hint yet
  std::vector<OrganizerBlobStat> stats;
  server->CollectOrganizerBlobStats(replica, stats);      // already partitioned by replica
  for (const auto &s : stats) {
    const float target = ScoreFor(s, phase);              // your policy
    if (std::fabs(target - s.score_) < kEpsilon) continue;
    clio::run::u32 rc = 0;
    CLIO_CO_AWAIT(server->ReorganizeBlobInternal(s.tag_id_, s.blob_name_, target, rc));
  }
  CLIO_CO_RETURN;
}
```
Register the name in `DataOrganizerFactory::Get`, add the source to the CTE
core CMakeLists, document new config keys. Phase hints when the application
knows something stats cannot show: `CLIO_CTE_CLIENT->ReorganizeHint(k)` at the
boundary, with a small enum of meanings written next to the organizer.

**4. Prove it, single node, with tiers.** Two or three tiers with a fast tier
smaller than the working set; A/B against `organizer: none`, same input;
report faults, evictions, wall time. Cover a regime the policy should help
and one it should not. A run with zero evictions proves nothing about
placement.

## The economics that decide whether a policy can pay

- A migration = one page read + one write ~ **271 us** (256 KB); the most it
  can save = slow-tier fault minus fast-tier fault ~ **46 us**. A promoted
  page must be hit several times (>= 4) in the fast tier to break even.
- A fault is ~110 us of round trip against ~6 us of copy: placement touches
  only a few percent of a fault's cost unless the tier underneath is genuinely
  slow (a file system). With host RAM as the slow tier, every policy measured
  was within +-1% of none.
- **Fill cost rule, which predicts the sign before you write code:** filling
  the fast tier costs `(tier_bytes / page_bytes) x 271 us`. If that is a large
  fraction of the run, the policy cannot win at any hit rate (one run cost
  104 ms to fill against a 45 ms run, and lost 14-23%).
- Gate promotions on **stalls** (a block parked on a transfer proves the page
  was on the wrong tier), not on position alone.
- Measured gains, where positive, were 2-4% and only with a file-backed slow
  tier.

## Gotchas

- **Stats can be blind.** The paged fault path does not stamp `last_read_`,
  so a policy on "written but never read" rescored everything every round
  (+9.9%). Check what the hot path records before building on a timestamp.
- **A demotion is a migration.** Dead data wants to be dropped, and the
  interface has no drop verb.
- **Freeing a tier only helps if live data can use it**: with pages at 0.5
  and the fast tier at 1.0, freeing that tier optimizes a tier nothing uses.
- **Recency is an anti-signal for cyclic sweeps**: the most recently touched
  page is the one furthest from its next use. A frecency policy's scores top
  out near 0.85 and never clear a 1.0 tier.
- `organizer_tasks` replicas partition pages by name hash: a policy needing a
  global view ("top-k hottest") must not assume it sees every page.
- Task structs are ABI: rebuild everything after touching one. Never log in
  lock paths; the organizer runs on a worker thread.
