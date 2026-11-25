# Evals - clio-cae

Scenarios recorded 2026-11-21 against clio-core `dev` (bd1dc3183). Rubric is
pass/fail per bullet. Run each WITHOUT the skill to capture the gap, then WITH
it to confirm the gap closes.

## S1 - import from a pip install

Setup: Prompt: "I installed iowarp-core with pip. How do I import all the
files in ~/runs/2026-11 into CLIO so I can search them? `clio-cae` is not
found."

Expected:

- Says the `clio-cae` CLI is not in the pip wheel and uses
  `clio_cee.ContextInterface().context_bundle([...AssimilationCtx(src="file::...",
  dst="iowarp::<tag>", format="binary")])` instead, one context per file.
- Uses absolute paths and notes the runtime (not the client) reads them and
  `allowed_roots` must permit the directory.
- States what lands: `description` + `chunk_N` blobs of 1 MiB per tag, and
  that empty files fail (rc -5).
- Says imports are searchable because the CAE writes through 564.0 in the
  default config, and shows a `context_query(..., prompt=...)` check.

## S2 - write an OMNI file

Setup: Prompt: "Write an OMNI file that imports /data/sim.h5 but only the
datasets under /fields, skipping anything ending in _tmp, and then a second
transfer that must run after it." Build has HDF5.

Expected:

- Uses `src: hdf5::/data/sim.h5`, `dst: iowarp::<tag>`, `format: hdf5`,
  `dataset_filter.include_patterns`/`exclude_patterns`.
- Notes `*` crosses `/` and that a filter matching nothing fails with -9.
- Uses `depends_on` with the exact `src` or `dst` of the first transfer, not
  a transfer name.
- Names the resulting tags `<dst>/fields/...` and the 1.5 MiB chunk blobs.

## S3 - method-id collision

Setup: Prompt: "Can I just compose clio_cae_core at pool 512.0 so my
existing CTE clients go through it, like the test configs do?"

Expected:

- Says no: CAE ids 10-13 (ParseOmni, ProcessHdf5Dataset, ExportData,
  ImportData) collide with the core's RegisterTarget, UnregisterTarget,
  ListTargets, StatTargets.
- Recommends CAE at 400.0 with `next_pool_id: "564.0"`, composed after the
  chain.
- Warns that an uncomposed CAE is auto-created by `clio_cee` without
  `next_pool_id`, bypassing the indexer.

## S4 - summarizer cost

Setup: Prompt: "Turn on LLM summaries for everything we import; we import
about 2 TB a day of mixed binary and text."

Expected:

- Explains inference is synchronous on the worker and the put waits for it,
  with 120 s per-request timeout and per-1 MiB-chunk labels.
- Proposes narrow `label_matches` (anchored regexes, text tags only, e.g.
  `^chunk_[0-9]+$` with a `.txt` tag rule), and notes regexes are substring
  matches.
- Gives the compose order (indexer -> summarizer 401 -> CAE 400 with
  `next_pool_id: "401.0"`) and that the summarizer is not in the pip wheel and
  needs libcurl + nlohmann_json.
- Keeps tokens out of YAML via `label_api_key_file` / `label_api_key_env`.
