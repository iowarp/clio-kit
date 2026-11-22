# CODECS.md: codecs, selection, models, GPU

Checked against clio-core `dev` (bd1dc3183), 2026-11-21.

## Codec registry

The wire id is stored in each compressed blob's header and in
`Context::compress_lib_`. It is frozen: a blob written with a codec still names
it on a build that lacks it (reading it there fails with rc 3). A name resolves
only if its library was found at configure time.

| name | wire id | kind | presets | library needed at build |
|---|---|---|---|---|
| `brotli` | 0 | lossless CPU | fast/balanced/best | brotli. **Unusable through the compressor**: id 0 means "no codec" |
| `bzip2` | 1 | lossless CPU | fast/balanced/best | bzip2 |
| `blosc2` (alias `blosc`) | 2 | lossless CPU | single mode | c-blosc2 |
| `fpzip` | 3 | lossy CPU | error-bound presets | LibPressio |
| `lz4` | 4 | lossless CPU | fast/balanced/best | lz4 |
| `lzma` | 5 | lossless CPU | fast/balanced/best | xz / liblzma |
| `snappy` | 6 | lossless CPU | single mode | snappy |
| `sz3` | 7 | lossy CPU | error-bound presets | LibPressio |
| `zfp` | 8 | lossy CPU | error-bound presets | LibPressio |
| `zlib` | 9 | lossless CPU | fast/balanced/best | zlib |
| `zstd` | 10 | lossless CPU | fast/balanced/best | zstd |
| `nvcomp-lz4`, `nvcomp-snappy`, `nvcomp-zstd`, `nvcomp-gdeflate`, `nvcomp-deflate`, `nvcomp-ans` | 11-16 | lossless GPU | single mode | nvcomp + `CLIO_CORE_ENABLE_CUDA=ON` |
| `zfp-sycl` | 17 | lossy GPU (Intel XPU) | 8 / 16 / 24 bits per value | SYCL-enabled libzfp, `CLIO_CTE_ENABLE_ZFP_SYCL` |
| `cusz` | 18 | lossy GPU | abs. error 1e-2 / 1e-3 / 1e-4 | cuSZ |
| `ndzip` | 19 | lossless float GPU | single mode | ndzip |
| `cuszp` | 20 | lossy GPU | abs. error 1e-2 / 1e-3 / 1e-4 | cuSZp |

Presets: `1` fast, `2` balanced (default), `3` best. Single-mode codecs ignore
the preset.

## How the codec for a put is chosen

The context of each put carries `dynamic_compress_` (mode), `compress_lib_`
(wire id) and `compress_preset_`.

1. **Writer chose.** Mode `kCompressStatic` (1) with a wire id: exactly that
   codec and preset. Mode `kCompressDynamic` (2): the models pick. Mode
   `kCompressSkip` (0) and wire id 0: no codec named.
2. **Pool default.** If the writer named neither a codec nor a mode,
   `default_compress` fills them in (`dynamic` sets mode 2).
3. **Eligibility.** Only a whole-blob primary write is compressed: offset 0, no
   segment list, not an append, `replica_ == 0`, not emulated. Anything else is
   forwarded raw with the codec request cleared.
4. **Dynamic selection** (mode 2). Samples up to 64 KiB of the payload, as
   bytes or as 32-bit floats when `data_type_ == 1`, and computes Shannon
   entropy, mean absolute deviation and mean second derivative. Candidates are
   fixed: zstd balanced, zstd fast, lz4 fast, bzip2 best, zlib balanced.
   Predictions come from the Q-table model if `qtable_model_path` loaded, else
   the dense NN (`dnn_model_weights_path`, builds with Eigen3), else a heuristic
   ratio of 2.0 for all, which keeps the first candidate (zstd balanced).
   - Default objective: highest predicted ratio.
   - `max_performance_ = true`: lowest predicted compress + decompress +
     transfer time. Transfer time uses the fastest storage target's score.
   - Candidates with predicted PSNR below a non-zero `target_psnr_` are dropped.
   Lossy and GPU codecs are never chosen dynamically; request them statically.
5. **Profit check.** The stored image (header + codec output) must be at most
   87.5% of the raw size, else the raw bytes are stored.

`MultiPutBlob` with a codec (or the default) is split into single puts, each
judged as above. Without a codec, the batch is forwarded as one.

## Models

None ship in any package. The Q-table and LinReg predictors load from a model
directory; the dense NN loads a JSON weights file. They are produced offline
with the training-data generator in the clio-core source tree
(`context-transfer-engine/compressor/generator`). A failed load logs
`Failed to load Q-table model from: <path>` and falls back to the next
predictor.

## GPU codecs

GPU codecs serve device producers such as the paged GPU vector: a compressed
page can be decoded straight into the faulting device page. The module creates
its own CUDA context, streams and fixed device buffers at pool creation and
never allocates per operation. A request that does not fit a slot, or arrives
with every slot busy, falls back to the CPU equivalent or stores raw.
Several nvcomp-family pages faulted together are decoded in one batched launch.

| variable | default | effect |
|---|---|---|
| `CLIO_COMPRESS_GPU_CODEC` | on | `0` forces GPU codecs onto their CPU equivalent |
| `CLIO_COMPRESS_GPU_SLOTS` | `16` | concurrent GPU codec slots (each has its own stream) |
| `CLIO_COMPRESS_GPU_BUF_MB` | `8` MiB | device buffer per slot (input and output each) |
| `CLIO_COMPRESS_GPU_SCRATCH_MB` | `128` MiB | device scratch for compressed output, in 16 slabs |
| `CLIO_COMPRESS_GPU_BATCH` | on | `0` disables batched decode (A/B measurement) |
| `CLIO_COMPRESS_LINGER_US` | `300` us | how long the batcher waits to gather more pages |
| `CLIO_CODEC_TRACE`, `CLIO_CODEC_TIME` | unset | debug prints of codec decisions and timings |

## Costs

- Put: one compression pass plus one shared-memory buffer of about
  1.05x the payload. The scheduler starts from roughly 300 bytes/us of CPU for
  compression and 3x faster for decompression, then learns from completions.
- Get of a compressed blob: one forwarded read, a size probe, a read of the
  whole stored image, a full decode, then copies. A 4 KiB read of a 64 MiB blob
  decodes 64 MiB.
- Size probe: one extra 32-byte header read.
