# Evals - clio-bdev

Scenarios recorded 2026-11-21 against clio-core `dev` (bd1dc3183). Rubric is
pass/fail per bullet. Run each WITHOUT the skill to capture the gap, then WITH
it to confirm the gap closes.

## S1 - file tier that does not survive restart

Setup: Prompt: "I added `- {path: /data/tier.dat, bdev_type: file, capacity_limit: 0g, score: 0.3}` to my CTE core's storage list. Will my data survive a restart, and how big is the tier?" Config: pip install, one CTE core, no `durability` key.

Expected:

- Says `capacity_limit: 0g` is a config error for a file tier (0 means 80% of DRAM only for ram).
- Says the tier is volatile without `persistence_level`, so the core is inferred volatile and starts empty after restart.
- Recommends `persistence_level: temporary` or `long_term` plus `metadata_log_path`.
- Notes the allocator log `/data/tier.dat_node0.alloc_log` is replayed by `clio-run start` and deleted by `--fresh`.

## S2 - survive a dead disk

Setup: Prompt: "I have three NVMe drives in one node. Give me a compose config where the CTE survives one drive dying, and tell me how to test it." Config: user installed with pip.

Expected:

- States safe_bdev is not in the pip wheel; needs a source, conda or Spack build.
- Composes three file bdevs before a `clio_safe_bdev` pool with `max_failures: 1`, one member `parity: true`, and an `alloc_log`.
- The CTE storage entry uses `existing_pool_id`, `existing_pool_module: clio_safe_bdev` and `persistence_level: long_term`; keeps `num_replicas: 0`.
- Tests with a `<member file>.fail` marker and checks `faulty_members` in the `stats` monitor query; rebuilds with `RecoverBdev` / `clio_safe_bdev_recover`.
- Says beyond `max_failures` reads fail with an I/O error, never wrong bytes.

## S3 - S3 tier setup

Setup: Prompt: "Add an S3 bucket `my-archive` as the slowest CTE tier." Config: pip install.

Expected:

- Says the s3 type is not in the wheel; needs `-DCLIO_ENABLE_AMAZON_DRIVE=ON` with Poco.
- Uses `path: "my-archive/clio"` (bucket plus prefix), not a bare bucket, because `_node<N>` is appended.
- Names `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (required, in the runtime's environment), `AWS_DEFAULT_REGION` (default us-east-1), `S3_ALLOW_BUCKET_CREATE=1` for a missing bucket.
- Warns the s3 allocator is in memory only and restarts empty.

## S4 - fresh start leaks space

Setup: Prompt: "After `clio-run start --fresh` my file bdev with `alloc_log: /var/lib/clio/d0.alog` reports almost no remaining capacity. Why?"

Expected:

- Explains an explicit `alloc_log` is replayed even after `--fresh`, while CTE metadata is wiped, so old blocks stay allocated.
- Says only the default `<pool_name>.alloc_log` is deleted by `--fresh`.
- Fix: stop the runtime, delete the log (and for safe_bdev its `.members` / `.journal`), then start fresh.
