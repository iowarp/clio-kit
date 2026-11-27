---
name: clio-indexer
description: 'Use when configuring, scoping, debugging or calling the CLIO Core keyword search index (clio_cte_indexer, pool 564.0): BM25 SemanticSearch over blob contents from Python (clio_cte_core_ext, or clio_cee context_query with a prompt) or C++, index persistence, memory growth, ReindexScan. Triggers on "SemanticSearch", "search my files", "context_query prompt", "564.0", "index_log_path", "indexer OOM". Not for ingesting data (use clio-cae) or ordering the chain (use storage-compose).'
metadata:
  bundle: clio-core
  servers: none
  provenance: designed
  eval-status: scenarios-recorded
---

# Indexer: keyword search over blob contents (`clio_cte_indexer`)

Facts checked against clio-core `dev` (bd1dc3183, 2026-11-21). Source paths in
parentheses are provenance for maintainers only.

## Files in this skill

| file | read when |
|---|---|
| **SEARCH.md** | calling search: Python (`clio_cte_core_ext`, `clio_cee`), C++, result format, BM25 details, `ReindexScan`, worked examples. |
| **evals.md** | checking that this skill closes the gaps it targets. |

## What it is

The indexer is an **interposer** in the CTE chain. It speaks the CTE core's
task interface, forwards every call down to `next_pool_id`, and on the way
notes which blobs changed. In the background it reads those blobs back,
tokenizes their bytes and keeps a per-node **forward index** (term counts per
blob) in RAM. It is the only pool that answers `SemanticSearch` (core method
id 35): BM25 keyword ranking over blob **contents**, filtered by tag-name and
blob-name regexes. The CTE core no longer implements search.

Cost: a put pays one O(1) dirty-key enqueue on the ack path. The real work
(read-back + tokenize) runs in a periodic sweep or at the start of the next
search. The index lives in process memory and grows with every indexed blob.

"Semantic" is a historical name. It is exact-token BM25: no embeddings, no
stemming, no fuzzy matching.

## Availability

| item | value |
|---|---|
| `mod_name` | `clio_cte_indexer` |
| default `pool_id` | `564.0` (`pool_name: clio_cte_indexer`) |
| pip wheel | **yes** (the default config composes it; without the library `clio-run start` fails to create 564.0) |
| build option | built whenever the CTE is built (`CLIO_CORE_ENABLE_CTE=ON`, the default) |
| module-only method ids | `kIndexSweep` 100 (internal periodic task), `kReindexScan` 101 (C++ client only) |
| interposed core ids | PutBlob 15, DelBlob 18, DelTag 19, SemanticSearch 35, TruncateBlob 37, RenameTag 38, MultiPutBlob 48, plus UpdateTagNames |

## When to compose it / when not

Compose it when people or agents need to **find blobs or files by the words in
them**: notes, logs, JSON, CSV, source code, metadata, text imported by the
CAE. It is in the default chain.

Leave it out, or scope it tightly, when the data is large binary (simulation
output, model weights, images, checkpoints) or when you write GB-scale data
through the chain. See "Memory growth" below. Name-only lookups
(`TagQuery`, `BlobQuery`, `context_query` without a prompt) never need the
indexer; the core answers them from metadata.

## Compose entry

Default entry (from the shipped `clio_default.yaml`):

```yaml
compose:
  # ... bdev, cores, replication, write-back cache, shm cache 568.0 above ...
  - mod_name: clio_cte_indexer
    pool_name: clio_cte_indexer
    pool_query: local
    pool_id: "564.0"
    next_pool_id: "568.0"            # shm cache below the indexer
    index_log_path: "${CLIO_STORAGE_ROOT}/cte_indexer_index"
    index_sweep_period_ms: 100        # async drain cadence (0 = lazy)
    index_wal_compact_bytes: "8MB"    # WAL size that triggers compaction
    tag_re: ".*"                      # index scope: tag names...
    blob_re: ".*"                     # ...and blob names that get tokenized
```

Minimal stand-alone entry over a single core:

```yaml
  - mod_name: clio_cte_indexer
    pool_name: clio_cte_indexer
    pool_query: local
    pool_id: "564.0"
    next_pool_id: "512.0"
    index_log_path: "${CLIO_STORAGE_ROOT}/cte_indexer_index"
```

Keys `IndexerConfig::LoadConfig` reads (indexer/include/clio_cte/indexer/indexer_tasks.h):

| key | type | default | meaning and constraints |
|---|---|---|---|
| `next_pool_id` | `"major.minor"` string | unset: forwards to `512.0` | Pool every call is forwarded to. Must already be composed (earlier in the list). |
| `index_sweep_period_ms` | unsigned int, ms | `100` | Period of the background drain. `0` disables it: indexing then happens only when a search runs, and the WAL is compacted only at clean shutdown. |
| `tag_re` | ECMAScript regex, full-string match | `".*"` | Index scope. Only blobs whose full tag name matches are tokenized. Out-of-scope blobs cost no read, no RAM, no WAL. A regex that fails to compile logs an error and **opens the scope to everything**. |
| `blob_re` | ECMAScript regex, full-string match | `".*"` | Same, for blob names. Both must match. |
| `index_log_path` | path; `${VAR}` and `~` expanded | empty: in-memory only | Snapshot file at this path plus an append log at `<path>.wal`. With it, a restart restores the index; without it, the index starts empty. |
| `index_wal_compact_bytes` | size (`"8MB"`, `"1GB"`, bytes) | 8 MiB | WAL size past which the sweep rewrites the snapshot and truncates the WAL. |

Standard keys `mod_name`, `pool_name`, `pool_query` (`local`), `pool_id` as for
every module. Unknown keys are ignored. **Parsing is best-effort and stops at
the first bad value**: a non-numeric `index_sweep_period_ms` silently leaves it
and every key parsed after it (`tag_re`, `blob_re`, `index_log_path`,
`index_wal_compact_bytes`) at their defaults. Check the runtime log line
`Indexer: async index sweep every N ms` after start.

Environment variables:

| variable | read by | effect |
|---|---|---|
| `CLIO_CTE_POOL=564.0` | client process | Binds the process's CTE client singleton (Python `clio_cte_core_ext`, C++ `CLIO_CTE_CLIENT`) to the indexer. Set it before import/init. Needed for search from any process that cannot map the shm cache (TCP clients, remote nodes). |
| `CLIO_STORAGE_ROOT` | runtime | Root for `index_log_path` in the default config. Default `~/.clio`. |
| `CLIO_INDEXER_PASSIVE` | runtime | **Any value, even `0`**, makes the indexer forward-only: nothing is enqueued or indexed. Unset it to re-enable. |

## Where it goes in the chain

Default (dev) chain, top to bottom:

```
CAE 400 / filesystem 560 / stream 565 / Python and C++ clients
        -> indexer 564 -> shm cache 568 -> [compressor 562] -> write-back 567
        -> volatile core 540 | replication 561 -> persistent core 512
```

- **Above the compressor**, so read-backs see logical (uncompressed) bytes.
- **Above everything that writes data you want searchable.** Only writes that
  pass through 564.0 are noticed. Writes sent straight to a lower pool are
  indexed only by a backfill (see SEARCH.md, "Getting old data in").
- **Compose order:** after the pool it names (`next_pool_id`), before every
  pool that names it (`entry_pool_id` of the shm cache, `next_pool_id` of
  CAE, stream, filesystem, summarizer).
- To **remove** it, re-point every `564.0` reference: the shm cache's
  `entry_pool_id` (to its own id or the pool above it), and the
  `next_pool_id` of CAE, stream and filesystem (to `568.0`).

## How indexing works

1. **Enqueue.** A successful PutBlob, MultiPutBlob or TruncateBlob marks its
   (tag, blob) key dirty. Keys coalesce: ten overwrites cost one re-index.
   Replica-addressed writes are skipped. DelBlob and DelTag remove documents;
   RenameTag and filesystem renames update tag names.
2. **First-insertion backfill.** The first write to a tag the indexer has never
   seen also enqueues that tag's pre-existing blobs (once per tag).
3. **Drain.** The sweep (or a search) pops each dirty key, asks the chain below
   for the blob's current size, reads the **whole blob** into one shared-memory
   buffer, and tokenizes it. Size 0 or a failed read removes the document.
4. **Tokenizer.** ASCII letters and digits only, lowercased; any other byte
   (including every byte >= 0x80) separates tokens; runs shorter than 2 are
   dropped. **There is no maximum token length.** Queries are tokenized the
   same way.

## Guarantees

The index is **derived state**. It never changes what a write means: the
indexer forwards each write first and acks only after the pool below acks.
Indexing failures never fail a write.

| event | effect on the data | effect on search |
|---|---|---|
| acknowledged write | as defined by the chain below | **Read-your-writes:** a search issued after the ack drains every dirty key first, so it sees the write (on the node that owns the blob; a `Local` search sees only the submitting node). |
| client crash | none | none; acked keys are already enqueued |
| runtime crash (`kill -9`) with `index_log_path` | as the chain below | Snapshot + WAL are restored. Keys acknowledged but not yet drained at the crash are **lost from the dirty set**: those blobs stay stale or unsearchable until rewritten or `ReindexScan`. Every drained document was WAL-flushed. |
| runtime crash without `index_log_path` | as below | Index comes up empty. Each tag backfills on its next write; untouched tags need `ReindexScan`. |
| clean stop (`clio-run stop`) | as below | Snapshot includes still-dirty keys and pending backfills; the next start resumes exactly. |
| power loss | as below | The WAL is flushed to the OS, **not fsynced**, and the snapshot is rewritten in place (not atomically). The WAL tail, or part of a snapshot being compacted, can be lost. Repair with `ReindexScan`. |
| disk loss (index files) | as below | Index empty; `ReindexScan`. |
| temporary node loss | as below | That node's slice of the index (its blobs) cannot answer; a Broadcast search follows the runtime's dead-node handling (see the `clio-runtime-config` skill). |
| permanent node loss | as below | That node's index slice is gone. Replica writes are never indexed, so blobs served from a successor are not searchable there until `ReindexScan` runs on the survivors. |
| `clio-run start --fresh` | cores start empty | **The indexer still restores `index_log_path`** (it does not honor `--fresh`). Searches return ghost hits for blobs that no longer exist, and new tags can reuse old tag ids. Delete `<index_log_path>` and `<index_log_path>.wal` before a `--fresh` start. |

## Client API (summary)

Clients call the **CTE core's API on pool 564.0**. Full signatures and
examples: SEARCH.md.

| language | call |
|---|---|
| Python | `cte.get_cte_client().SemanticSearch(tag_regex, blob_regex, query_text, k=10, pool_query=cte.PoolQuery.Broadcast())` -> `list[SemanticSearchResult]` (`.tag_id`, `.blob_name`, `.score`) |
| Python (CEE) | `clio_cee.ContextInterface().context_query(tag_re, blob_re, max_results=0, prompt="words")` -> `list[str]` of blob names, best first |
| C++ | `clio::cte::core::Client(PoolId(564,0)).AsyncSemanticSearch(tag_regex, blob_regex, query_text, k=10, PoolQuery::Broadcast())` |
| C++ only | `clio::cte::indexer::Client::AsyncReindexScan(tag_regex=".*", blob_regex=".*", PoolQuery::Broadcast())` -> `blobs_enqueued_` |

## Memory growth and the GB-scale OOM

- RAM per document: tag name, blob name, and one hash-map entry per distinct
  term (term string + count). Distinct-term-heavy data (hex, UUIDs, base64,
  numbers) costs the most.
- A run of letters/digits with no separator is **one token of any length**.
  A 1 MiB page of letters-only data becomes a 1 MiB token stored in RAM and
  written to the WAL on every re-index.
- Measured 2026-10-09 (release 2.3.0, container with 4 GB DRAM): writing 16 GB
  of 1 MiB blobs through the default chain OOM-killed the runtime at about
  10 GB written (anonymous RSS 7-8 GB, a 13.6 GB WAL). After the runtime died,
  the FUSE mount did not return EIO: each write blocked about 120 s.
- WAL compaction runs only after a sweep empties the dirty set. Under a
  sustained write stream the WAL can grow far past `index_wal_compact_bytes`.

Mitigations, best first: scope with `tag_re`/`blob_re` to text-like names
(e.g. `tag_re: ".*\\.(md|txt|csv|json|log)$"`); keep bulk data under tags the
scope excludes; set `CLIO_INDEXER_PASSIVE=1` for a bulk load, then
`ReindexScan` the text you need; or remove the indexer from the chain. Cap
container memory (`docker update --memory`) for benchmarks.

## Pitfalls and failure signatures

| symptom | cause | fix |
|---|---|---|
| `SemanticSearch` returns `[]` for everything, no exception | Client is bound to the core (512.0), which does not serve method 35. Happens to TCP/remote clients that cannot map the shm cache, or if `CLIO_CTE_POOL` names a pool below 564. | `export CLIO_CTE_POOL=564.0` before importing `clio_cte_core_ext` / `clio_cee`. |
| Results exist but every score is 0 | No query token occurs in the matched blobs. Every matched blob is ranked, including zero scores. | Filter `score > 0`. Check tokenization (ASCII, min 2 chars, no stemming: `ice` does not match `icy`). |
| Blob is listed by `BlobQuery` but never found by search | Written below 564.0 (bypassed the indexer), out of scope (`tag_re`/`blob_re`), size 0, or non-ASCII text. | Write through 564.0; widen scope; `ReindexScan` (C++). |
| Search returns `[]` and the log says `bad regex` | Invalid `tag_regex`/`blob_regex` (task rc 1; Python hides it). | Fix the regex. Patterns are full-match: use `.*notes/.*`, not `notes`. |
| Ghost results after `start --fresh` | Indexer restored its old snapshot. | Delete the index files, restart. |
| Runtime OOM / huge `cte_indexer_index.wal` during bulk writes | Index of bulk/binary data; unbounded tokens. | Scope, passive mode, or remove the indexer (above). |
| Two nodes corrupt each other's index | Same `index_log_path` on a shared filesystem (e.g. NFS home, default `~/.clio`). The path is used as given; nothing per-node is appended. | Point `CLIO_STORAGE_ROOT` or `index_log_path` at node-local disk on each node. |
| Search is slow | Every query scans every document in the node's index (forward index, no postings) and the first search after a write burst pays for indexing it. | Narrow scope; keep `index_sweep_period_ms` > 0. |
| Partial writes to huge blobs are slow to index | Each drain re-reads the whole blob into one buffer. A failed allocation leaves the old entry. | Keep indexed blobs small (FUSE pages are 1 MiB). |

## Verify it works

```bash
clio-run start            # default config composes 564.0
clio-run status           # running
```

```python
import os
os.environ.setdefault("CLIO_WITH_RUNTIME", "0")   # attach to the running runtime
os.environ.setdefault("CLIO_CTE_POOL", "564.0")   # bind to the indexer
import clio_cte_core_ext as cte
assert cte.clio_init(cte.RuntimeMode.kClient, False)
assert cte.initialize_cte("", cte.PoolQuery.Dynamic())
tag = cte.Tag("idx_smoke")
tag.PutBlob("a", b"Arctic sea ice extent fell to a record low.", 0)
tag.PutBlob("b", b"Atmospheric CO2 reached 421 ppm.", 0)
hits = cte.get_cte_client().SemanticSearch("idx_smoke", ".*", "sea ice", 5)
print([(h.blob_name, round(h.score, 3)) for h in hits])
assert hits and hits[0].blob_name == "a" and hits[0].score > 0
```

Expected: `[('a', <positive>), ('b', 0.0)]`. Also check the runtime log for
`Indexer: async index sweep every 100 ms` and, after a restart,
`Indexer: restored N docs (M pending) from ...`. The dashboard (`clio-run viz
start`) Pools page lists `clio_cte_indexer` at 564.0.
