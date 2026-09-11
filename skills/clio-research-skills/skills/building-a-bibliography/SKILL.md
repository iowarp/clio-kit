---
name: building-a-bibliography
description: Use when assembling verified citations, BibTeX records or a research reading list. Triggers on "BibTeX", "cite these papers", "reference list". Not for broad topic discovery; use surveying-literature-and-datasets.
metadata:
  bundle: clio-research
  servers: clio-arxiv, clio-ndp, clio-web
  provenance: designed
  eval-status: scenarios-recorded
---

# Build a Verified Research Bibliography

A fabricated citation is the worst failure available here. It is fluent,
correctly formatted, and points at a paper that does not exist. Every entry has
to come from a tool that returned it.

Prefer `clio-arxiv:export_to_bibtex` for arXiv records. For other sources,
use publisher/repository citation metadata. Manual formatting or correction is
acceptable only when every field is checked against a retrieved record; never
fill gaps from memory.

## Steps

**1. Collect candidates.**

Search by the axis that matches the question — see
`surveying-literature-and-datasets` for which of the seven search tools to use.
For a bibliography specifically, two are unusually useful:

- `clio-arxiv:search_papers_by_author` — for a group's body of work, and for
  finding the rest of a line of research once you have one paper from it.
- `clio-arxiv:find_similar_papers` — takes a paper you already have and finds
  neighbours by its categories and keywords. This is how the survey stops being
  a keyword list and starts covering a field.

**2. Verify each one you intend to cite.**

`clio-arxiv:get_paper_details` for the full record. Check the authors, the year
and the version. A preprint's title and author list change between versions, and
citing v1's title for v3's content is a real error.

**3. Export.**

`clio-arxiv:export_to_bibtex` on the results. Read what comes back: ArXiv
metadata is author-supplied and imperfect. Names, capitalisation in titles, and
journal fields for papers since published all need checking.

**4. Get the PDFs if they will be read.**

`clio-arxiv:get_pdf_url` for a link, `download_paper_pdf` for one file, or
`download_multiple_pdfs` for several with bounded concurrency. Respect service limits and report download failures.

## Preprints and published versions

An ArXiv entry is a preprint. Some are later published, often with a different
title and always with a different citation. The ArXiv record does not always know
this.

When it matters — a citation in a submission, or a claim resting on peer review —
check with `clio-web:search` for the published version and cite that. Say which
you are citing; the two are not interchangeable, and a reviewer will notice.

## Citing datasets

Data deserves a citation as much as a paper does.
`clio-ndp:get_dataset_details` returns the metadata a citation needs, and many
datasets carry a DOI. A results section resting on data with no citation cannot
be checked by anyone.

## What not to do

- Do not write a BibTeX entry that did not come from a tool.
- Do not "fix" a returned entry from memory — verify it instead.
- Do not cite a paper whose details you have not fetched.
- Do not cite a preprint as published without checking.
- Do not loop single downloads where the concurrent tool exists.
- Do not leave the datasets uncited.

## Tool discovery across agents

Names such as `clio-hdf5:open_file` identify a server and its tool in this
guide. Your agent may expose a different prefix. Match the server and tool
against its live MCP inventory, then use the advertised name and input schema.
If a required server is unavailable, report it before attempting the workflow.

## Completion check

Deduplicate by stable identifier and version. Check each BibTeX key, title, authors, date, identifier and publication status against retrieved records; retain source links and flag unresolved fields instead of inventing them.
