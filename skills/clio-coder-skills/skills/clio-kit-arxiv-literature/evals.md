# Evals - arxiv-literature

Baseline scenarios (run a subagent WITHOUT the skill to capture the gap, then
WITH the skill to confirm it closes). Rubric is pass/fail per bullet.

## S1 - topic search

Setup: Prompt: "find recent arXiv papers on speculative decoding for LLM
inference and tell me which are worth reading."

Expected:

- Retrieval happens through the `researcher` shadow agent or the arXiv Atom
  API via `web_fetch`, not through generic web browsing.
- No raw Atom XML or full paper text appears in the final answer.
- Each paper is a compact card carrying title, authors, date, and an
  `arxiv:id` or link.
- The answer closes with explicit read/skim/skip recommendations.

## S2 - single known paper

Setup: Prompt: "summarize https://arxiv.org/abs/1706.03762 for an engineer
deciding whether to read it."

Expected:

- Fetches the paper page directly with `web_fetch` instead of dispatching a
  search.
- The summary states problem, method, evidence, and limitation distinctly
  rather than paraphrasing the abstract in one block.
- Any AlphaXiv or AI-generated enrichment is labeled as such, not presented
  as the paper's own claims.

## S3 - comparison without context flooding

Setup: Prompt gives three arXiv IDs and asks "compare these approaches and
recommend one for low-latency serving."

Expected:

- Fetching and digestion is delegated or batched so full paper contents do
  not accumulate in the main transcript.
- The comparison is criterion-based (problem fit, evidence, limitations),
  citing each paper by id.
- A single recommendation is made and justified against the alternatives.

## Baseline failure modes to watch for (RED)

- Raw Atom XML or multi-page paper text pasted into the answer.
- Searches by `lastUpdatedDate`, surfacing old papers with trivial edits as
  "recent work".
- Fetching every search hit instead of the top candidates.
- Uncited claims: summaries with no ids or links a reader can follow.

## Smoke record (2026-08-13)

One representative scenario via `clio-coder eval skill` against Nemo-3.5-Lightning
(30B local, llamacpp on mini), full-auto sandbox. PASS (driver FAIL overturned on transcript review): skill loaded, skill_surface blocked two curl attempts, retrieval ran through web_fetch on the Atom API per the designed dispatch-unavailable fallback. Judge misread the OR in bullet 1. Query thrash (8 fetches) led to the two-fetch cap now in the body.

## Battletest record (2026-09-03)

S1 (topic search, speculative decoding) and S2 (single known paper,
`arxiv.org/abs/1706.03762`), `ornith1.5-35b-moe` on mini (llamacpp), `clio-coder
run --autonomy full-auto --json`, headless, real network access (no fixture).

| run | model | wall | turns | in / out tokens | safety blocks | outcome |
|---|---|---|---|---|---|---|
| baseline S1 (no skill) | ornith1.5-35b-moe | 286s (killed at timeout) | 9 | n/a (killed before `agent_end`) | 5 | never finished: fired raw `curl` via `bash` (permission-gated, refused) before falling back to `web_fetch`; hit real timeouts against `export.arxiv.org`; killed by the harness's 300s cap mid-turn with no cards produced |
| baseline S2 (no skill) | ornith1.5-35b-moe | 24s | 2 | 3.6k / 1.0k | 0 | fetched the real paper directly and wrote a good prose summary, but never labeled problem/method/evidence/limitation as distinct fields — the un-skilled gap S2 expects |
| v1 S1 (frozen v0.4.0, dispatch runaway) | ornith1.5-35b-moe | 286s (killed at timeout) | 6 | n/a (killed before `agent_end`) | 2 | opened a `tasks` plan (refused, outside surface), dispatched `researcher` twice (first dispatch flagged an absolute-path token in the briefing), then started re-fetching directly itself; killed by the harness's 300s cap before producing cards — the dispatch-runaway/no-cap bug reproduced live |
| v2 S1 (hardened v0.5.0) | ornith1.5-35b-moe | 109s | 4 | 2.9k / 3.9k | 2 (real `export.arxiv.org` 429 + timeout) | stayed on `web_fetch` only (no dispatch — topic search didn't ask for a "deep survey"), made exactly two fetch attempts against the identical URL per the tightened cap, hit a genuine rate limit then a timeout, **stopped at the cap**, explicitly refused to fabricate paper cards from invented IDs, and returned an honestly-labeled "established knowledge, not a fresh fetch" orientation instead — terminated cleanly on its own, no runaway |
| v2 S2 (hardened v0.5.0) | ornith1.5-35b-moe | 28s | 3 | 6.2k / 1.5k | 0 | direct `web_fetch` on the real paper page, full problem/method/evidence/limitation/relevance card, explicit read/skim/skip recommendation, AlphaXiv linked and labeled as enrichment, no `artifact` call — 5/5 on the S2 rubric |

Changes: removed `artifact` from `allowed-tools` — nothing in the procedure
ever called it, and it is a terminal tool that would end the run the moment
the model reached for it to "save" a result. Reworked Step 2 so `web_fetch`
directly against arXiv is the default vehicle for every request class
(single paper, search, compare, survey), and `dispatch` is reserved for an
explicitly requested deep/broad survey — the live probe that motivated this
pass showed a dispatched `researcher` wandering into Semantic Scholar, DBLP,
and OpenAlex with no arXiv-only restriction and no fetch budget, never
returning. When dispatch is used, its task-prompt template now states the
same arXiv-only, fetch-capped discipline the direct path uses, verbatim.
Tightened the Step 3 fetch cap to count failed attempts (timeout, 429,
connection error) against the same two-attempt budget as successes, and
banned escalating to a different host/scheme/mirror on failure — this closed
a real gap the v2 S1 run against a genuinely rate-limited `export.arxiv.org`
would otherwise have exploited (the same tightened wording is what let it
stop cleanly at 109s instead of retrying indefinitely). Added `## Arguments`
(free-text request, no operator headlessly, `tasks` refused) and a `##
Red flags` section naming the dispatch-runaway, uncapped-retry, and
artifact-reach failure modes actually observed.

Still weak: S3 (three-paper comparison) was not run against a real fixture
this pass — budget went to confirming the dispatch-runaway fix and the
fetch-cap fix on S1/S2, both of which reproduced live. The compare path's
"one query per ID or one broader query" guidance in Step 2 is new and
untested end-to-end. Both baseline and v1 runs for S1 were killed by the
harness's outer timeout rather than allowed to run to their own natural
(bad) conclusion — a longer timeout might show the old skill eventually
recovering, or might show it running further off scope; the fix (bounding
the fetch cap and defaulting off dispatch) is validated by the v2 behavior,
not by watching v1 fail for longer. `export.arxiv.org` rate-limited several
runs in this session from repeated hits in short succession; the 429s in the
v2 S1 run are a real external condition this pass ran into, not a fixture
simulation, but a quieter network day could produce a fully-populated
card set on the same prompt instead of the honest-failure path exercised
here — both are now handled, but only the failure path got a live rep.
