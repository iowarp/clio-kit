---
name: clio-kit-materio-literature-reviewer
description: 'Use when this workflow is requested: Review supplied materials and provided
  URLs with citation provenance and explicit gaps. Triggers on "clio kit materio literature
  reviewer".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/plugins/materio/skills/materio-literature-reviewer
  upstream-name: materio-literature-reviewer
  upstream-eval-status: unspecified
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

Adapted skills use the `clio-kit-` prefix to distinguish them from upstream audited skills. For a companion skill named below, select its `clio-kit-` copy from this collection; native agents, fleets and external programs retain their original names.

For a materials literature review, read the confirmed research question and supplied corpus, confirm scope, synthesize what is supported, and write .research/LITERATURE.md. Include a Key Papers table, Sources Reviewed with inspection depth, evidence limitations, justified gaps (possibly none), and proposed keyword changes. Let the researcher decide question/keyword revisions and citation findings.

Read the [shared research policy](assets/references/research-policy.md) for
state, provenance, execution boundaries, and advisory validation.

Read the linked references when needed. Resolve links from this skill directory.
If acting as a worker, return questions to the orchestrator; when acting as the
interactive assistant, collect the researcher answers directly using the host's
conversation facility.

- [research-domains.md](assets/references/research-domains.md)
- [RESEARCH.md](assets/templates/RESEARCH.md)

## What a Good Literature Review Produces

1. **Knowledge map**; What is currently known, organized by sub-topic
2. **Methodological landscape**; What approaches exist, their strengths and limits
3. **Genuine gaps**; What remains unknown or contested, with justification
4. **Keyword refinement**; More precise terms than the initial set
5. **Key references**; Landmark papers and recent work (last 5 years)
6. **Loop-back signal**; Whether the research prompt needs adjustment

## Supplied Materials Come First

Papers, BibTeX files, and notes the researcher registered under `.research/data/` are the primary corpus. They anchor the review in the group's actual reading; record unsupported formats honestly. Researcher-provided URLs can extend that corpus when allowed. If `<web_allowed>` is false or the web tools are unavailable, review the supplied materials only and say so plainly in Review Notes. A review built from supplied materials alone is complete and honest; a review padded with invented papers is neither.

## Citation Integrity

Only list papers you actually read (supplied) or actually found (web) and are confident are real. Put each paper title in "quotes" in the Key Papers table and include a DOI when you have one, so the orchestrator can machine-verify the list against Crossref. If you are unsure a specific paper exists or only vaguely recall it, do not invent bibliographic details: omit it, or mark it `[topic; unverified, ~YEAR]` and note it under Review Notes. Flagged entries are shown to the researcher for a decision; they are not silently removed.

## Gap Quality

A genuine research gap is:
- **Specific**: "The fatigue behavior of HEAs at cryogenic temperatures" not "more research is needed"
- **Justified**: Supported by absence of papers or explicit statements in existing literature
- **Feasible**: Something the researcher's prompt could actually address

Not a gap: "Nobody has studied exactly this material system" (often because it's not interesting)

## Loop-Back Conditions

Signal `loop_back:` if:
- The prompt is already fully answered (>3 recent papers directly address it)
- The prompt is so broad it spans multiple review articles without a clear angle
- The prompt contains a false premise (assumed mechanism doesn't exist)

Read supplied files and the data index first. Fetch only researcher-provided URLs
when project policy and the available host retrieval capability permit it. Record
actual inspection depth, failed/unreadable sources, and coverage. Metadata alone
is not full-text inspection, and a restricted corpus does not establish global
absence. Ask for usable text if an unreadable paper blocks the research.


## Complete action guides

Read only the guide for the requested operation; it contains its interview and
state transitions. Resolve these links from this skill directory.

- [literature-review](assets/actions/literature-review.md)
