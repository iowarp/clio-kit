---
name: clio-kit-arxiv-literature
description: 'Use when this workflow is requested: Searches arXiv, summarizes or compares
  papers, and builds a compact literature survey as citation-ready, source-linked
  paper cards. Prefer the Researcher shadow agent for noisy multi-paper retrieval.
  Triggers on "search arXiv", "summarize an arXiv paper".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/research/arxiv-literature
  upstream-name: arxiv-literature
  upstream-eval-status: smoke-checked
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

Adapted skills use the `clio-kit-` prefix to distinguish them from upstream audited skills. For a companion skill named below, select its `clio-kit-` copy from this collection; native agents, fleets and external programs retain their original names.

# ArXiv Literature

Find, summarize, or compare academic papers without flooding the main context
window. Raw search results and paper text stay in a worker or get compressed
immediately; only paper cards reach the user.

## Arguments

```text
/skill clio-kit-arxiv-literature <request>
```

The request is free text: a paper URL/ID, a topic, or two or more IDs to
compare. There is no operator in a headless run: `ask_user` is not registered, so
any call is refused as an unregistered tool rather than answered. If the request is ambiguous (no
clear topic, an ID that doesn't resolve), state your best reading and
proceed; never stall waiting for clarification. `tasks` sits outside this
skill's tool surface and is refused; the steps below are the whole plan, do
not open a task list for them.

## Step 1 — Classify the request

Pick exactly one:

- **Single paper**: one arXiv URL/ID, "summarize this paper", "explain this paper".
- **Search**: "find papers about X", "search arxiv for X", "latest papers on X".
- **Compare**: two or more paper URLs/IDs, "compare these papers".
- **Survey**: "what is the literature on X", "best papers for X".

## Step 2 — Pick the vehicle

Default every request — single paper, search, compare, and survey alike — to
`web_fetch` directly against arXiv:

- Paper URL/ID: fetch the arXiv page. Clio normalizes it into structured
  metadata plus AlphaXiv enrichment when available.
- Topic, compare, or survey: fetch the arXiv Atom API; Clio compacts the XML
  into paper cards:

  ```text
  https://export.arxiv.org/api/query?search_query=all:QUERY&sortBy=submittedDate&sortOrder=descending&start=0&max_results=10
  ```

  For a compare request, run one query per paper ID (`id_list=ID` instead of
  `search_query`) or one broader query covering all of them — whichever stays
  inside the fetch cap in Step 3.

Useful categories for `search_query`: `cs.AI` (AI), `cs.LG` (ML), `cs.CL`
(NLP/LLMs), `cs.CR` (security), `cs.SE` (software engineering), `cs.MA`
(multi-agent), `cs.IR` (retrieval/RAG), `cs.CV` (vision), `cs.RO` (robotics).

**Only dispatch the `researcher` shadow agent when the user explicitly asks
for a deep or broad survey** ("survey the field", "don't just skim arXiv, go
wide") — never as the default for an ordinary search or compare. Left
unbounded, a dispatched worker has no arXiv-only restriction and no fetch
budget of its own, and will wander into Semantic Scholar, DBLP, OpenAlex, and
general web search, taking several minutes to return nothing useful. When you
do dispatch, state the same bound this skill uses directly, in the task
prompt itself:

```text
Research arXiv literature for: <user goal>.
Search arXiv only (export.arxiv.org Atom API or arxiv.org paper pages) — do
not query Semantic Scholar, DBLP, OpenAlex, or general web search. One
round, at most two fetch attempts total (successes and failures both count).
On a timeout or HTTP error, do not retry with a different host, scheme, or
protocol — retry the identical URL at most once, then stop and report the
failure. Build cards from whatever you have; do not keep escalating.
Return only compact source-linked paper cards, comparison/synthesis,
caveats, and read/skim/skip recommendations.
```

If dispatch is unavailable, fall back to the direct `web_fetch` path above.

## Step 3 — Return paper cards only

Never paste raw Atom XML or full paper text into the response. Output format:

```markdown
## Literature Result

### Query
...

### Best Matches / Papers
1. **Title** — authors, date, `arxiv:id`
   - Problem:
   - Method:
   - Evidence:
   - Limitation:
   - Relevance:
   - Links:

### Recommendation
- Read:
- Skim:
- Skip:
```

Done when every returned paper has a card with a working link and the
recommendation section is filled in. Stop after one search round unless the
user asks to go deeper; do not keep fetching to "be thorough". One round
means at most two Atom API fetch attempts total: the initial query plus one
refinement, or one retry of a failed fetch. Failed attempts (timeout, 429,
connection error) count against this cap the same as successful ones — a
timeout is not a free retry. Rewording the same query a third time, or
retrying through a different host/scheme (`http` vs `https`,
`export.arxiv.org` vs `arxiv.org/search`, an unofficial JSON mirror) after a
failure, is thrash: build cards from whatever the attempts returned, or
report the network failure plainly and stop.

## Gotchas

- arXiv Atom is XML, not JSON; fetch it through `web_fetch` so Clio compacts it.
- `lastUpdatedDate` surfaces old papers with trivial edits; use `submittedDate`
  for newly submitted work.
- AlphaXiv is AI-generated enrichment: useful for scanning, never citable as
  authoritative.
- Fetch or enrich only the top candidates, not every result.
- `export.arxiv.org` rate-limits (HTTP 429) under repeated hits; that is a
  reason to stop at the fetch cap, not a reason to retry against a mirror.
- This skill's tool surface has no file-writing tool and no `artifact`. The
  paper cards are the chat reply, never a document; do not go looking for a
  way to save one.

## Red flags

- Dispatching `researcher` for an ordinary search or compare "just in case"
  it does a better job than a direct fetch — it is slower and unbounded by
  default; reserve it for an explicit deep-survey ask.
- A dispatch task prompt without the arXiv-only, fetch-capped instruction —
  that omission is what let a prior run wander into Semantic Scholar, DBLP,
  and OpenAlex for minutes with nothing to show for it.
- Treating a timeout or 429 as free of the fetch cap and retrying against a
  different host, scheme, or unofficial mirror instead of stopping.
- Raw Atom XML, a full abstract dump, or more than the top few candidates
  reaching the final reply.
- Reaching for `artifact` or any write tool to "save" the result — it is not
  in this skill's tool surface; the reply is the deliverable.
