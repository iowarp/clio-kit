# Evals - building-a-bibliography

Current revision review (2026-09-10): tool names and workflow claims were checked
against the shipped server schemas and implementation. Historical records below
apply to earlier text; they are not fresh model evaluations of this revision.

Current acceptance criterion: Deduplicate by stable identifier and version. Check each BibTeX key, title, authors, date, identifier and publication status against retrieved records; retain source links and flag unresolved fields instead of inventing them.

Baseline scenarios: run each WITHOUT the skill to capture the gap, then WITH it
to confirm the gap closes. Rubric is pass/fail per bullet.

## S1 - no invented citations

Setup: Prompt: "give me BibTeX for the five most important papers on
checkpointing in HPC."

Expected:

- Every entry originates from `export_to_bibtex` on real search results.
- No entry is composed from memory, and none is "corrected" by hand.
- If fewer than five are found, that is stated rather than padded.

## S2 - verify before citing

Setup: Prompt gives an arXiv ID and asks for a citation.

Expected:

- `get_paper_details` is called; authors, year and version are checked.
- The answer notes if the record is a preprint and does not present it as
  published.

## S3 - concurrent download

Setup: Prompt: "get me the PDFs for these eight papers."

Expected:

- `download_multiple_pdfs` is used rather than a loop of single downloads.

## Baseline failure modes to watch for (RED)

- A BibTeX entry that did not come from a tool.
- Editing a returned entry from memory instead of re-verifying.
- Citing a paper whose details were never fetched.
- Citing a preprint as published.
- Looping single downloads where the concurrent tool exists.
- Leaving datasets uncited.

## Trigger record (2026-08-21)

Ran through `evals/trigger_eval.py`, which loads the skill plugins into the
Agent SDK with an empty `setting_sources` and only the Skill tool allowed, so
selection is measured without the operator's own configuration influencing it.

Prompt: "Give me BibTeX for these three arXiv IDs."

This skill fired, and no sibling fired alongside it. Across the suite: 20 of 20
skills selected correctly on their own prompt, and 3 control prompts outside the
kit fired nothing.

Selection is checked. Whether the skill improves the final answer, versus an
agent working without it, is still not measured.

