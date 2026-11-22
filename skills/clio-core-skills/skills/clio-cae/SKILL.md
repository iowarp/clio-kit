---
name: clio-cae
description: 'Use when importing external data into CLIO Core with the Context Assimilation Engine: writing OMNI files, running clio-cae, calling ParseOmni or clio_cee context_bundle from Python/C++, choosing sources (file, HDF5, HTTP, S3, GCS, Globus), allowed_roots, or composing the LLM summarizer (401.0). Triggers on "OMNI file", "clio-cae", "assimilate", "ParseOmni", "context_bundle", "label_matches", "summarizer". Not for searching imported data (use clio-indexer) or chain ordering (use storage-compose).'
metadata:
  bundle: clio-core
  servers: none
  provenance: designed
  eval-status: scenarios-recorded
---

# Context Assimilation Engine (`clio_cae_core`, `clio_cae_summarizer`)

Facts checked against clio-core `dev` (bd1dc3183, 2026-11-21). Source paths in
parentheses are provenance for maintainers only.

## Files in this skill

| file | read when |
|---|---|
| **OMNI.md** | writing an OMNI file, running `clio-cae`, choosing a source scheme, reading error codes. Every key, every source, full examples. |
| **SUMMARIZER.md** | composing the LLM summarizer: keys, rules, cost, endpoints (Ollama, OpenAI-compatible). |
| **evals.md** | checking that this skill closes the gaps it targets. |

## What it is

**Assimilation** copies data that lives outside CLIO (a local file, an HDF5
file, an HTTP/S3/GCS/Globus object, or an in-memory string) into CTE **tags**
and **blobs**, so it is served from the CTE tiers like anything written
through the CTE client. The CAE core is a pool (400.0) that receives a list of
transfers (an **OMNI** list) in one `ParseOmni` task and runs them in order
inside the runtime. It writes through `next_pool_id` (default config: the
indexer, 564.0), so imports are cached, replicated and keyword-searchable like
any other write. Cost: the import runs on a runtime worker and reads the
source from the **runtime's** host and credentials, not the client's.

The **summarizer** (401.0) is an optional interposer: for each PutBlob that
matches a rule, it sends the payload to an LLM and stores the answer as
`<blob>_label` in the same tag. Inference runs synchronously on the worker
and the put is not acknowledged until it finishes.

## Availability

| item | CAE core | summarizer |
|---|---|---|
| `mod_name` | `clio_cae_core` | `clio_cae_summarizer` |
| default `pool_id` / name | `400.0` / `cae_main` | `401.0` / `clio_cae_summarizer` |
| method ids | Create 0, Destroy 1, Monitor 9, **ParseOmni 10, ProcessHdf5Dataset 11, ExportData 12, ImportData 13**; forwards GetOrCreateTag 14, PutBlob 15, GetBlob 16, SemanticSearch 35 | interposes PutBlob 15; forwards every other core verb |
| pip wheel | **library yes**; `clio-cae` CLI **no** (use Python `clio_cee`) | **no** |
| default config | composed | commented-out example |
| build | `CLIO_CORE_ENABLE_CAE=ON` (default) | built with the CAE; real inference needs **libcurl and nlohmann_json** found at configure time, else every summary is empty |

The `clio-cae` binary (legacy aliases `clio_cae`, `clio_cae_omni`) is
installed by source builds and the deb/rpm packages. Source options per
scheme are in OMNI.md; the wheel supports `file::`, `string::` and, on
Linux/macOS, `http(s)://`.

## When to use it / when not

Use it to bring existing datasets into CLIO without writing a loader:
"import these files into tag X", "stage this HDF5 file into the fast tier",
"pull this S3 object in". Use the summarizer to attach short LLM summaries to
text as it arrives.

Do not use it for data your application writes itself (write through the
CTE client or the FUSE mount), for formats with no assimilator (ADIOS2 BP,
Parquet, CSV parsing: convert with an OMNI `run:` step first, or import them
as opaque bytes with `file::`), or for transfers that must be atomic.

## Compose entries

Default CAE entry (must come **after** the pool it names):

```yaml
  - mod_name: clio_cae_core
    pool_name: cae_main
    pool_query: local
    pool_id: "400.0"
    next_pool_id: "564.0"                     # chain top (indexer)
    allowed_roots: ["~/data", "/scratch/$USER"]
```

| key | type | default | meaning and constraints |
|---|---|---|---|
| `next_pool_id` | `"major.minor"` | unset: `512.0` (bare persistent core; imports skip the indexer and caches) | Pool the CAE writes through. Must be composed earlier. A malformed value is ignored silently. |
| `allowed_roots` | path or list of paths; `~`, `$VAR` expanded; symlinks resolved | unset: any path except the deny list | Local paths `file::`/`hdf5::` sources may read and `ExportData` may write. Read once, at the first import after start: changes need a restart. Every `clio_cae_core` entry's list is merged. |

Always refused, even inside a root: paths through `.ssh`, `.aws`, `.gnupg`,
`.config`, `.clio`, `.docker`, `.kube`, `.azure`, `.pki`, `.password-store`,
`.gcloud`; files `.netrc`, `.git-credentials`, `.pgpass`, `.vault-token`,
`.npmrc`, `.pypirc`, `.boto`, `.s3cfg`, `.my.cnf`, `id_rsa`; `/proc`, `/sys`,
`/dev`, `/etc/shadow`, `/etc/gshadow`, `/etc/sudoers`, `/etc/ssl/private`.
Writes are also refused under `/etc`, `/usr`, `/bin`, `/sbin`, `/lib*`,
`/boot`, `/root` and to any hidden file or directory. **The runtime does not
authenticate clients**: any local client can ask the CAE to read any allowed
path as the runtime user. Set `allowed_roots`.

Summarizer entry and its keys: SUMMARIZER.md.

Runtime environment variables (set where `clio-run start` runs):

| variable | effect |
|---|---|
| `CLIO_CAE_TELEMETRY=<file>` | Append one JSON record per HDF5 assimilation with phase timings (ms). |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_PROFILE`, `AWS_DEFAULT_REGION`, `S3_ENDPOINT` / `AWS_ENDPOINT_URL`, `AWS_NO_SIGN_REQUEST` | `s3://` credentials, region, S3-compatible endpoint, unsigned public reads. |
| `GOOGLE_APPLICATION_CREDENTIALS`, `GCS_ENDPOINT` | `gs://` credentials; endpoint override (anonymous). |
| `GLOBUS_ACCESS_TOKEN` | `globus://` token when `src_token` is empty. |
| `CAE_HTTP_AUTH_FILE`, `CAE_HTTP_STREAMS` (8), `CAE_HTTP_PART_MB` (64), `CAE_HTTP_PART_RETRIES` (5), `CAE_HTTP_BACKOFF_MS` (1000), `CAE_HTTP_OVERWRITE` (0), `CAE_HTTP_PROGRESS_S` (30), `CAE_HTTP_CURL` (`curl`) | HTTP downloads (see OMNI.md). |
| `CLIO_VOL_CHUNK_SIZE` | Chunk size for `format: hdf5_vol`; must equal the application's. |
| `https_proxy` / `no_proxy` | Proxy for HTTP, S3, GCS and the summarizer; loopback is always direct for the summarizer. |

## Where it goes in the chain

```
clio-cae / clio_cee ──ParseOmni──> CAE 400 ──> [summarizer 401] ──> indexer 564 ──> ... ──> cores
```

- The CAE is an entry point, not part of the data path of other clients.
- Point `next_pool_id` at the chain top (564.0) so imports are indexed. With a
  summarizer, CAE -> 401 and summarizer -> 564.
- Order in the file: chain pools, then the summarizer, then the CAE.
- **Never compose `clio_cae_core` at `512.0` (or any CTE pool id), and never
  send CAE calls to a CTE pool.** CAE ids 10-13 collide with the core's
  RegisterTarget 10, UnregisterTarget 11, ListTargets 12 and StatTargets 13:
  a ParseOmni sent to 512.0 runs RegisterTarget, and a RegisterTarget sent to
  a CAE sitting at 512.0 runs ParseOmni.
- If no CAE pool is composed, `clio_cee` / `CLIO_CAE_CLIENT_INIT` creates one
  at 400.0 named `cae_client_pool` with **no `next_pool_id`**, so imports go
  straight to 512.0 and are not indexed. Compose it explicitly.

## Client API

Python (pip wheel), through the CEE:

```python
import os
os.environ.setdefault("CLIO_WITH_RUNTIME", "0")   # attach to a running runtime
os.environ.setdefault("CLIO_CTE_POOL", "564.0")
import clio_cee as cee
ctx = cee.ContextInterface()
rc = ctx.context_bundle([
    cee.AssimilationCtx(src="file::/home/me/data/run42.bin",
                        dst="iowarp::run42", format="binary"),
])
print(rc)                                        # 0 = every transfer stored
print(ctx.context_query("run42", ".*"))          # ['description', 'chunk_0', ...]
```

`AssimilationCtx(src, dst, format, depends_on="", range_off=0, range_size=0,
src_token="", dst_token="")`, or with `src_data=` (bytes as `str`) for
`string::<blob>` sources. Use keyword arguments. Python cannot set
`dataset_filter`, `run`, `on_fault`, `chunk_size` or `blob_prefix`; use an
OMNI file with `clio-cae`, or C++. `context_bundle` returns `0` or the
negative CAE code (OMNI.md, "Error codes"); `1` if the client could not
initialize.

C++ (`clio::cae::core::Client`, header `clio_cae/core/core_client.h`; source,
conda or Spack installs only):

| call | meaning |
|---|---|
| `Client(clio::cae::core::kCaePoolId)` | bind to 400.0 (never a CTE pool) |
| `AsyncParseOmni(const std::vector<AssimilationCtx>&)` | run transfers; read `result_code_` (int, 0 ok), `error_message_`, `num_tasks_scheduled_`. Check `result_code_`, not only `GetReturnCode()`. |
| `AsyncExportData(tag_name, output_path, format /*"hdf5"|"binary"*/, pool_query=Local())` | write a tag's blobs to a file; `bytes_exported_` |
| `AsyncImportData(tag_name, input_path, format /*"hdf5"*/, pool_query=Local())` | import the HDF5 dataset whose in-file path equals `tag_name` into the key-value HDF5 layout; `bytes_imported_` |
| `AsyncProcessHdf5Dataset(pool_query, file_path, dataset_path, tag_prefix)` | one dataset; used internally to spread HDF5 work across nodes |
| `CLIO_CAE_CLIENT_INIT()` / `CLIO_CAE_CLIENT` | process singleton (also inits the CTE client) |

`AssimilationCtx` C++-only fields: `include_patterns`, `exclude_patterns`,
`on_fault` (ingest nothing now; fetch each blob from the source the first time
it is read, "ingest on fault"), `chunk_size` (bytes per blob, 0 = 1 MiB),
`blob_prefix` (default `"chunk_"`).

## Guarantees

What one transfer produces (`file::`): tag `<dst>` with blob `description`
(text `binary<size=N, offset=O>`) and blobs `chunk_0`, `chunk_1`, ... of
1 MiB each (last one shorter). `hdf5::`: one tag per dataset,
`<dst>/<dataset path without leading />`, each with a `description` blob
(type and dimensions) and `chunk_N` blobs of 1.5 MiB. `string::name`: one blob
`name`. Details per source: OMNI.md.

| question | answer |
|---|---|
| When does `ParseOmni` return 0? | After every transfer finished and every PutBlob of `file::`, `string::` and `hdf5::` transfers was acknowledged by `next_pool_id`. The CLI's "assimilation is ASYNCHRONOUS" note is conservative for these sources. |
| What does an acknowledged import mean? | Whatever an acknowledged put means in the chain below (default chain: in the owner's volatile core, flushed to the persistent core later; see `storage-compose`). |
| Order and failure | Transfers run one at a time, in list order. The first failure stops the list; later transfers never start; `num_tasks_scheduled_` tells how far it got. **No rollback:** blobs already stored stay. |
| Idempotent? | Re-running writes the same blob names and overwrites them. Not cleaned: if the source shrank, higher-numbered chunks from the old run remain. For an exact copy, delete the tag first (`context_destroy`). HTTP downloads skip a destination that already holds the exact bytes. |
| Client crash mid-import | The import keeps running in the runtime to completion; the result is lost with the client. Verify by listing the tag. |
| Runtime crash (`kill -9`) mid-import | No import journal. Acked chunks survive as the chain below guarantees; the tag is partial. Re-run the OMNI file. |
| Power loss, disk loss, node loss | As for any CTE write through `next_pool_id`. The CAE keeps no state of its own except the on-fault registry tag (C++ `on_fault`). |
| `clio-run start --fresh` | Imported data is discarded with the rest of the node's CTE state; re-import. |
| Summarizer | A summary failure never fails the put. The `_label` blob is written below the summarizer after the original put succeeded; a crash between the two leaves the data without a label. |

## Pitfalls and failure signatures

| symptom | cause | fix |
|---|---|---|
| `No assimilator found for source: file::...` (rc -2) for a file that exists | Path refused by the path policy (`refusing source ...: outside allowed_roots` / `credential location` in the runtime log), or scheme not compiled in. | Add the directory to `allowed_roots` and restart; or rebuild with the scheme's option. |
| `clio-cae: command not found` after `pip install iowarp-core` | The CLI is not in the wheel. | Use `clio_cee` `context_bundle`, or a source/deb/rpm install. |
| rc -3 `depends_on '...' is not the src or dst of any earlier transfer` | `depends_on` names a transfer by `name`, or a later transfer. | Use the exact `src` or `dst` string of an earlier transfer. |
| rc -9 on HDF5 | `include_patterns` matched no dataset. `*` crosses `/`. | Check dataset paths (`CTP_LOG_LEVEL=debug` lists them). |
| rc -10 / `PutBlob failed` in the runtime log | Tier full below. | Add capacity or free space; re-run. |
| rc -5 on `file::` | Source missing, unreadable by the runtime user, or **0 bytes** (empty files are rejected). | Fix the path; skip empty files. |
| Imported data not found by keyword search | CAE `next_pool_id` unset or points below 564.0, or CAE pool auto-created by `clio_cee`. | Compose the CAE with `next_pool_id: "564.0"`. |
| S3 import uses the wrong region or no credentials | Credentials are read from the **runtime's** environment. | Export `AWS_*` where `clio-run start` runs. |
| Cluster misconfigured, targets vanish, or ParseOmni does nothing | CAE composed at, or client pointed at, a CTE pool id (method-id collision). | Keep the CAE at 400.0. |
| Puts through the summarizer take seconds | Synchronous inference on the worker. | Narrow `label_matches`; see SUMMARIZER.md. |

## Verify it works

```bash
clio-run start
printf 'hello assimilation\n' > ~/data/hello.txt
```

```python
import os
os.environ.setdefault("CLIO_WITH_RUNTIME", "0")
os.environ.setdefault("CLIO_CTE_POOL", "564.0")
import clio_cee as cee
ctx = cee.ContextInterface()
src = "file::" + os.path.expanduser("~/data/hello.txt")
assert ctx.context_bundle([cee.AssimilationCtx(src=src, dst="iowarp::cae_smoke",
                                               format="binary")]) == 0
print(sorted(ctx.context_query("cae_smoke", ".*")))   # ['chunk_0', 'description']
print(ctx.context_retrieve("cae_smoke", "chunk_0"))  # ['hello assimilation\n']
print(ctx.context_query("cae_smoke", ".*", 1, "assimilation"))  # ['chunk_0']
ctx.context_destroy(["cae_smoke"])
```

With a source install, the same check is `clio-cae smoke.yaml` (OMNI.md) and
exit status 0. The runtime log shows `ParseOmni: Successfully scheduled 1
assimilations`; the dashboard Pools page lists `cae_main` at 400.0.
