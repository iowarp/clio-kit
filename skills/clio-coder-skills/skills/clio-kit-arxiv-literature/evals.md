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
