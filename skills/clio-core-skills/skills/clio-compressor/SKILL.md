---
name: clio-compressor
description: 'Use when enabling, configuring or debugging clio_cte_compressor (pool 562.0): building with CLIO_CTE_ENABLE_COMPRESS or Spack +compress, default_compress and presets, static vs dynamic (model-chosen) codecs, GPU codecs, chain placement, and stored vs logical size. Triggers on "compress my blobs", "default_compress", "zstd", "lz4", "kCompressDynamic", "562.0", "compression ratio". Not for choosing or ordering the whole chain; use storage-compose. Not for the blob API itself; use clio-cte-core.'
metadata:
  bundle: clio-core
  servers: none
  provenance: designed
  eval-status: scenarios-recorded
---

# clio-compressor: transparent blob compression

## What it is

The compressor is the encoding layer of the CTE chain. It speaks the CTE core's
task interface. On a **whole-blob** put it compresses the payload, prefixes a
32-byte header, and forwards that stored image to the next pool. On a get it
fetches the whole stored image, decompresses it once, and copies out the
requested range(s), so every layer above sees the original bytes. A size probe
returns the logical (uncompressed) size. It costs CPU (or GPU) time on every
put and every read, and a full-blob decode for any partial read.

Facts here were checked against clio-core `dev` (bd1dc3183), 2026-11-21.

## Files in this skill

| file | read when |
|---|---|
| **CODECS.md** | picking a codec, setting one per put from C++, dynamic selection and its models, GPU codecs and their environment variables |
| **evals.md** | checking the skill still closes its gaps |

## Availability

| item | value |
|---|---|
| `mod_name` | `clio_cte_compressor` |
| default `pool_id` | `562.0` (by convention; commented out in the default config) |
| pip wheel `iowarp-core` | **not shipped** (the wheel is built with `CLIO_CTE_ENABLE_COMPRESS=OFF`) |
| build option | `-DCLIO_CTE_ENABLE_COMPRESS=ON` (default OFF; the `debug` preset turns it on, `release` does not) |
| Spack | `spack install iowarp+compress` (adds lzo, bzip2, zstd, lz4, zlib, xz, brotli, snappy, c-blosc2; `+compress +cuda` adds cuSZ) |
| conda | the recipe builds the `release` preset; add `IOWARP_CMAKE_ARGS="-DCLIO_CTE_ENABLE_COMPRESS=ON"` and the codec libraries to the build env |

Every codec library is optional. A codec whose library was not found at
configure time is compiled out; the configure log says so
(`snappy not found - snappy codec disabled`). There is **no separate lz4
module** on dev: LZ4 is the codec `lz4` inside this module.

Install the whole of CLIO Core from one build with the option on. Do not drop a
compressor library built elsewhere into a pip install: the wheel ships no
headers or CMake package, and all modules must come from the same build.

## When to compose it, and when not to

- Compose it for compressible data that is written in whole blobs:
  checkpoints, simulation fields, model pages, clio-fs (which writes fixed-size
  page blobs).
- Do not compose it for data that is already compressed (images, archives):
  results that do not save at least one eighth are stored raw anyway, after
  paying the attempt.
- Do not compose it for workloads that patch byte ranges inside blobs or
  append (see Pitfalls).
- Skip it for latency-bound small reads of large blobs: every read decodes the
  whole blob.

## Compose entry

Default chain with the compressor enabled: compose it after write-back (567.0)
and re-point the shm cache (568.0) at it.

```yaml
compose:
  # ... bdev 301.0, persistent core 512.0, volatile core 540.0,
  #     replication 561.0, write-back 567.0 come first ...
  - mod_name: clio_cte_compressor
    pool_name: clio_cte_compressor
    pool_query: local                  # one container per node
    pool_id: "562.0"
    next_pool_id: "567.0"              # write-back cache
    default_compress: "zstd"           # codec for writes that name none
    default_preset: "balanced"         # fast | balanced | best
    tracking_enabled: true
    # qtable_model_path: ""            # trained models for "dynamic"
    # dnn_model_weights_path: ""
    # linreg_model_path: ""
    # distribution_model_path: ""
    # trace_folder_path: ""
  - mod_name: clio_cte_core_shm_cache_writethru
    pool_id: "568.0"
    next_pool_id: "562.0"              # was 567.0
    # ... rest of the shm cache entry unchanged ...
```

Every key `LoadConfig` reads:

| key | type | default | meaning and constraints |
|---|---|---|---|
| `next_pool_id` | string `"major.minor"` | none (falls back to the core at 512.0) | Pool the stored image is forwarded to. A value without a dot is ignored. |
| `default_compress` | string | unset = no compression | Codec for writes whose context names none: a codec name (`zstd`, `lz4`, `zlib`, `lzma`, `bzip2`, `snappy`, `blosc2`, ...; see CODECS.md), `dynamic` (models choose per blob), or `none`. **FUSE and Python writes name no codec, so without this key they are stored uncompressed.** A name not built into this binary logs `default_compress '<x>' is not a codec built into this binary` at pool creation and leaves no default. |
| `default_preset` | string | `balanced` | `fast`, `balanced` or `best`. Anything else is `balanced`. Ignored by single-mode codecs. |
| `tracking_enabled` | bool | `true` | Each decompress records the reading node per tag (at most 32 per tag); later explicit `Compress` tasks for that tag run on the latest reader. `false`: placed by hash of the tag id. Does not affect interposed puts. |
| `qtable_model_path` | string dir | unset | Q-table predictor for dynamic selection (used first). |
| `dnn_model_weights_path` | string (JSON weights) | unset | Dense-NN predictor; used only if no Q-table loaded and the build has Eigen3. |
| `linreg_model_path` | string dir | unset | Loaded at start, but dev's selection does not consult it. |
| `distribution_model_path` | string | unset | Accepted; no model is loaded (the classifier is built in). |
| `trace_folder_path` | string dir | unset | Where per-put prediction and scheduling traces go, for puts whose context sets `trace_`. |

No trained models ship with any package. Without them, dynamic selection always
picks zstd balanced (CODECS.md).

Parsing is best-effort: a malformed value silently stops parsing there. Check
the startup log for `writes without a codec are compressed with zstd`.

Environment variables (runtime process): GPU codecs only, listed in CODECS.md.
`CLIO_CODEC_TRACE=1` prints codec decisions to stderr.

## Where it goes in the chain

- **Above the write-back cache** (the default placement), below the shm cache
  and indexer. Then:
  - the indexer reads logical bytes;
  - the volatile core, the persistent core and every replication copy hold the
    compressed image;
  - compression happens on the put path before the write-back ack.
- **Never below write-back.** The write-back flush writes only the byte ranges
  that changed. Those are partial writes, which the compressor stores raw, and
  a raw range inside a compressed blob makes it unreadable.
- Always **above replication**, so copies are the compressed form.
- Compose it after the pool it names, and before any pool that names it.
- `pool_query: local`, same entry on every node.
- Compression runs on the node that receives the put.

## Guarantees

- **Ack.** The compressor compresses synchronously, then forwards. Its ack is
  the next pool's ack for the compressed image (in the default chain: in the
  owner's volatile core). Durability and loss on crash, power loss, disk or
  node loss, and `--fresh` are exactly those of the chain below. Compression
  adds no state of its own: the codec, preset and sizes are in each blob's
  stored header.
- **Failed or unprofitable compression never fails a put.** If the codec is
  missing, fails, or saves less than 1/8 (stored image > 87.5% of raw), the raw
  bytes are stored and the blob is not marked compressed.
- **Reads.** A get of a compressed blob returns the original bytes, for any
  range or vectored segment list. Ranges past the end are short reads.
- **Per-blob metadata.** In the core's blob record (persisted in the metadata
  WAL): `transform_flags_` (transformed + compressed bits, the authoritative
  "is this compressed" answer), `compress_lib_` and `compress_preset_` (codec
  requested; provenance only). In the stored bytes: a 32-byte header (magic
  `CTEC`, codec wire id, preset, original size, compressed size).
- **The compressed bit is sticky.** Once set it is cleared only when the blob
  is deleted.

## Client API

Clients use the **CTE core API on the chain top** (or on 562.0 directly). With
`default_compress` set, nothing else is needed; Python and clio-fs can only use
the default.

C++ (source/conda/Spack builds) can request a codec per put through `Context`:

```cpp
#include <clio_cte/core/core_client.h>
#include <clio_ctp/compress/compress_factory.h>
clio::cte::core::Context ctx;
ctx.dynamic_compress_ = clio::cte::core::kCompressStatic;      // 1
ctx.compress_lib_ = ctp::CompressionFactory::WireIdForName("zstd");  // 10; 0 if not built
ctx.compress_preset_ = 2;                                      // 1 fast, 2 balanced, 3 best
auto put = client.AsyncPutBlob(tag_id, "field.bin", 0, size, data, 1.0f, ctx,
                               clio::cte::core::kCtePutReplace);
put.Wait();
// put->context_.actual_compressed_size_, actual_compression_ratio_,
// actual_compress_time_ms_ report what happened (0 if stored raw).
```

Module-only verbs (C++ `clio::cte::compressor::Client`, ids >= 100):
`DynamicSchedule` (100), `Compress` (101), `Decompress` (102), and the periodic
`PollNodeLoad` (103) and `PollConsumers` (104). Interposed core verbs:
`PutBlob` (15), `GetBlob` (16), `GetBlobSize` (23), `MultiPutBlob` (48), and the
POD put/get variants used by GPU producers.

Read return codes from the compressor: `2` buffer allocation failed, `3` codec
not available in this build, `5` decode failed, `6` bad header or size,
`10 + x` the size probe or read below failed with `x`.

## Pitfalls and failure signatures

| symptom | cause | fix |
|---|---|---|
| Stored size equals logical size; nothing compresses | writer names no codec and `default_compress` unset (all FUSE and Python writes) | Set `default_compress` |
| Pool creation fails with an unknown module | build without `CLIO_CTE_ENABLE_COMPRESS` (always true for the pip wheel) | Rebuild with the option, or `spack install iowarp+compress` |
| `is not a codec built into this binary` in the log | codec library missing at configure time | Install its dev package and reconfigure; check the configure log |
| `default_compress: "brotli"` never compresses | brotli's wire id is 0, which means "no codec" | Use `zstd`, `lzma` or `bzip2` for high ratio |
| Read of a blob returns rc 5 or 6 after an update | a range write, append, or raw rewrite landed in a compressed blob (the compressed bit is sticky) | Rewrite blobs whole with `kCtePutReplace`, or delete then put; never write ranges into compressed blobs |
| Rewrite with smaller result reads back rc 5 | plain put at offset 0 does not shrink the blob; stale trailing bytes follow the codec stream | Use `kCtePutReplace` (flags `0x1`) for whole rewrites |
| Data stored raw though compressible | put was partial, vectored, an append, or replica-addressed | Write whole blobs from offset 0 |
| Small reads are slow | each read decodes the whole blob | Use smaller blobs (fixed-size pages) |
| Compressor composed below write-back corrupts updated blobs | flushes are partial range writes | Put the compressor above write-back |
| `dynamic` always picks zstd | no model loaded | Expected; set `qtable_model_path`, or use a static codec |

## Verify it works

Compare logical and stored size. The compressor reports the logical size; the
pool under it reports the stored size.

```python
import os
os.environ["CLIO_CTE_POOL"] = "562.0"       # any pool at or above the compressor
import clio_cte_core_ext as cte
assert cte.clio_init(cte.RuntimeMode.kClient)
assert cte.initialize_cte("", cte.PoolQuery.Dynamic())
tag = cte.Tag("compress_check")
data = b"0123456789abcdef" * 65536          # 1 MiB, very compressible
tag.PutBlob("b0", data)
assert tag.GetBlobSize("b0") == len(data)   # logical size
assert tag.GetBlob("b0", len(data)) == data
assert tag.GetBlob("b0", 16, 4096) == data[4096:4112]   # partial read
```

Then run the same script with `CLIO_CTE_POOL` set to the pool below the
compressor (`567.0` in the default chain), and only `GetBlobSize("b0")`. It
must be far smaller than 1048576 (header plus codec bytes). Equal sizes mean
nothing was compressed: check `default_compress` and the startup log.
