# Evals - clio-indexer

Scenarios recorded 2026-11-21 against clio-core `dev` (bd1dc3183). Rubric is
pass/fail per bullet. Run each WITHOUT the skill to capture the gap, then WITH
it to confirm the gap closes.

## S1 - search returns nothing

Setup: Prompt: "I pip installed iowarp-core, started `clio-run start`, wrote
text blobs from Python with clio_cte_core_ext over TCP, and
`client.SemanticSearch('.*', '.*', 'sea ice', 5)` always returns an empty
list. Why?" Config: default `clio_default.yaml`.

Expected:

- Says the CTE core (512.0) does not serve SemanticSearch; only the indexer
  (564.0) does.
- Says a client that cannot map the shm cache (TCP/remote) stays bound to the
  core, and the binding returns `[]` without raising.
- Gives the fix: `CLIO_CTE_POOL=564.0` set before importing the module.
- Gives a runnable check with a positive score for a known document.

## S2 - scope the index for bulk data

Setup: Prompt: "We are about to write 20 TB of HDF5 and binary checkpoints
through the FUSE mount on nodes with 256 GB RAM. We still want to search our
markdown notes. Is the default config OK?"

Expected:

- Warns that the default indexes everything in RAM and cites the measured OOM
  (16 GB of 1 MiB writes killed the runtime at about 10 GB on a 4 GB host) and
  the unbounded token length.
- Proposes `tag_re` (full-match) restricted to text extensions, with a valid
  YAML-escaped regex, on the 564.0 entry.
- Mentions `CLIO_INDEXER_PASSIVE` (any value enables it) or removing the
  indexer, and which `564.0` references must then be re-pointed.

## S3 - persistence and restarts

Setup: Prompt: "After `clio-run start --fresh` my searches return blobs that do
not exist anymore. And what happens to the index if the node is killed -9?"

Expected:

- Explains the indexer restores `index_log_path` regardless of `--fresh`, and
  says to delete `<path>` and `<path>.wal` before a fresh start.
- States that after kill -9 the snapshot + WAL are restored, but keys acked
  and not yet drained are lost from the dirty set and need a rewrite or
  `ReindexScan`.
- States the WAL is flushed but not fsynced (power loss can lose its tail).
- Notes `ReindexScan` is C++ only and that writing to a tag triggers a
  per-tag backfill only the first time that tag is seen.

## S4 - rank files from Python

Setup: Prompt: "Write a Python script that prints the 5 files under
`notes/` in my CLIO mount that best match a question, with their paths."

Expected:

- Uses `blob_regex` `[0-9]+` (page blobs) and a full-match tag regex such as
  `.*notes/.*`.
- Filters `score > 0` and explains that zero-score blobs are returned.
- Maps `tag_id` to paths via `TagQuery` + `Tag(name).GetTagId()` because the
  Python result has no tag name.
- Groups or explains multiple page hits per file.
