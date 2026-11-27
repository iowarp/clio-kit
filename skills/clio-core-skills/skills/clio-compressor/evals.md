# Evals - clio-compressor

Scenarios recorded 2026-11-21 against clio-core `dev` (bd1dc3183). Rubric is
pass/fail per bullet. Run each WITHOUT the skill to capture the gap, then WITH
it to confirm the gap closes.

## S1 - enable compression on a pip install

Setup: Prompt: "I installed iowarp-core with pip. How do I turn on zstd
compression for everything written through my clio-fs mount?"

Expected:

- States the wheel does not ship `clio_cte_compressor`, and gives a route:
  source build with `-DCLIO_CTE_ENABLE_COMPRESS=ON`, `spack install
  iowarp+compress`, or conda with that CMake argument.
- Sets `default_compress: "zstd"` and explains FUSE writes name no codec, so
  nothing compresses without it.
- Places the compressor above write-back (`next_pool_id: "567.0"`) and
  re-points the shm cache (568.0) to 562.0, with compose order respected.

## S2 - nothing is getting smaller

Setup: Prompt: "I composed the compressor at 562.0 but my stored data is the
same size as before. Python client, default_compress is not set." 

Expected:

- Identifies the missing `default_compress` (Python cannot set a codec per put).
- Mentions the 87.5% profit threshold for incompressible data.
- Gives a check comparing `GetBlobSize` through 562.0 (logical) with the pool
  below it (stored).

## S3 - range writes into compressed blobs

Setup: Prompt: "We update 4 KiB records in place inside 64 MiB blobs that go
through the compressor. Sometimes reads fail afterwards. Why?"

Expected:

- Explains only whole-blob writes are compressed; a range write lands raw in
  the compressed image, and the compressed mark is sticky.
- Recommends whole-blob rewrites with `kCtePutReplace` (or delete then put), or
  smaller fixed-size blobs, or no compressor for this data.
- Notes each small read decodes the whole 64 MiB blob.

## S4 - picking a codec

Setup: Prompt: "Should I use default_compress: dynamic or brotli for best ratio
on our float simulation output?"

Expected:

- Says brotli cannot be used through the compressor (wire id 0 means no codec).
- Says dynamic without a trained model always picks zstd balanced, and no
  models ship.
- Suggests a static high-ratio lossless codec (`zstd` best, `lzma`, `bzip2`),
  or a lossy codec (`zfp`, `sz3`) requested statically from C++ when error is
  acceptable.
