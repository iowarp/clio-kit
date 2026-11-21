# CONFIG.md: every key of a `clio_cte_core` compose entry

Checked against clio-core `dev` at bd1dc3183 (parser:
`context-transfer-engine/core/src/core_config.cc`; defaults:
`core/include/clio_cte/core/core_config.h`).

## How the entry is read

- `mod_name`, `pool_name`, `pool_query` (always `local`) and `pool_id` are
  read by the runtime. Every other key of the entry is handed to the core's
  `CreateParams::LoadConfig` as YAML.
- **Unknown keys are ignored silently.** A misplaced key (for example
  `organizer` under `performance:`) has no effect and logs nothing.
- **A validation error does not stop the runtime.** The core logs
  `Config error: ...` and `Failed to load config from string`, then starts
  with whatever was parsed before the error. Sections are parsed in this
  order: `performance`, `targets`, `storage`, `dpe`, then the top-level
  organizer and `durability` keys, and parsing stops at the first rejected
  value. A bad `fsync_mode`, or a value of the wrong YAML type in
  `performance`/`targets`, therefore drops the whole `storage` list, and the
  core starts with no targets (every put returns rc 11). Range checks
  (`neighborhood`, `stat_targets_period_ms`, ...) run after parsing; they log
  an error but the parsed values stay in place.
- Paths go through `${VAR}` expansion (`path`, `metadata_log_path`).
- Size strings: a number (decimals allowed) plus an optional suffix `b`,
  `k`/`kb`, `m`/`mb`, `g`/`gb`, `t`/`tb`, `p`/`pb` (case-insensitive, powers
  of 1024). No suffix means bytes.

## Top-level keys

| key | type | default | meaning and constraints |
|---|---|---|---|
| `durability` | string | empty (infer) | `volatile` or `persistent`; any other value is an error. When omitted: persistent only if the core has at least one device and every device has `persistence_level` `temporary` or `long_term`; otherwise volatile (including no devices). Can be changed at runtime with `AsyncSetDurability` until the next restart. |
| `organizer` | string | `none` | periodic data organizer: `none`, `frecency` (recency + frequency), `grayscott`, `cyclic`, `scatter`, `hotset`. Each round rescores blobs and moves them through ReorganizeBlob. |
| `organizer_tasks` | u32 | 1 | periodic organizer replicas; the blob space is hash-partitioned between them. Must be >= 1. |
| `organizer_period_ms` | u32 | 5000 | period between organizer rounds (ms). |
| `config_env_var` | string | `CLIO_SERVER_CONF` | name of the variable a standalone config load reads. Has no effect on a compose entry. |

## `storage:` (list of tiers)

Each item becomes one target. The core creates a `clio_bdev` pool for it
(pool id `<core major + item index>.<1 + node>`), or attaches an existing
pool. Bdev-level behavior (allocator logs, `io_depth`, `alignment`, object
store options, `hbm` and `pinned` build needs) is in the `clio-bdev` skill;
the keys below are the ones the core itself reads.

| key | type | default | meaning and constraints |
|---|---|---|---|
| `path` | string | none; required unless `existing_pool_id` | `ram::<name>` for memory tiers; a file path for `file`; `s3://bucket/prefix`, `gcs://...` or `daos://pool/cont/file` for those types. Must not be empty. The target is registered as `<path>_node<N>`. Give each core's devices distinct paths. |
| `bdev_type` | string | none; required unless `existing_pool_id` | `file`, `ram`, `hbm`, `pinned`, `noop`, `s3`, `gcs`, `daos`. Anything else is an error. `daos` needs a DAOS-enabled build. |
| `capacity_limit` | size | none; required unless `existing_pool_id` | tier capacity. `0`/`0g` is valid only for `ram` and means 80% of system DRAM; 0 on any other type is an error. A file tier grows lazily, so this is a cap. |
| `score` | float | -1.0 (automatic) | 0.0 (slowest) to 1.0 (fastest). Outside the range is an error. -1/absent: derived from measured bandwidth on a log scale. |
| `persistence_level` | string | `volatile` | `volatile`, `temporary` or `long_term` (both of the latter survive a restart). A persistent core refuses a `volatile` device (registration rc 16). Blocks on volatile devices are dropped at restart. **A disk tier must set this to be durable.** |
| `growth_unit` | size | `1GB` | file tiers: how far the file grows, and reserves, at a time. The first unit is reserved at creation, so on tmpfs each file tier costs this much RAM up front; use e.g. `64MB` there. |
| `existing_pool_id` | `"MAJOR.MINOR"` | unset | bind this tier to an already-composed pool that implements the bdev task interface (for example `clio_safe_bdev` at `350.0`) instead of creating one. Major must not be 0. That pool must be composed **before** the core. `path`, `bdev_type`, `capacity_limit` become optional; free space comes from the pool's own statistics. |
| `existing_pool_module` | string | empty | module name of that pool, used only in log messages. |

Not core keys on dev, ignored if present: `read_cache`, `alloc_log`,
`io_depth`, `alignment` (the last three are bdev keys for a hand-composed
`clio_bdev`).

## `performance:`

| key | type | default | meaning and constraints |
|---|---|---|---|
| `metadata_log_path` | path | empty (WAL off) | prefix of the metadata WAL and snapshot. Files: `<path>` (snapshot), `<path>.tmp` (snapshot being written), `<path>.blob.<N>` and `<path>.tag.<N>` (one shard per worker). The parent directory must exist and be writable. Unused on a volatile core. Without it a persistent core cannot find its data after a restart. |
| `transaction_log_capacity` | size | `32MB` | total WAL capacity, split evenly over the worker shards (min 4 KiB each). When the WAL grows past it, the next flush rebuilds the snapshot and truncates the WAL. |
| `flush_metadata_period_ms` | u32 | 5000 | period of the FlushMetadata task (ms): fsyncs every WAL shard; rebuilds the snapshot when the WAL is over capacity, or when unlogged score changes are pending and the last snapshot is over 1 min old. 0 disables the periodic task. Runs only when `metadata_log_path` is set. |
| `fsync_mode` | string | `durable` | `durable`: `SyncTag` (and clio-fs `fsync`) returns after the devices and WAL are synced. `deferred`: it returns at once, sets `deferred_ = 1`, and durability waits for the periodic flushes. Any other value is an error. |
| `stat_targets_period_ms` | u32 | 1000 | how often each target's capacity and free space are re-read (ms). Must be 10-60000. |
| `score_difference_threshold` | float | 0.05 | minimum score change for ReorganizeBlob to move a blob. 0.0-1.0. |
| `score_threshold` | float | 0.7 | 0.0-1.0. Validated, not used by the runtime on dev. |
| `max_concurrent_operations` | u32 | 64 | 1-1024. Validated, not used by the runtime on dev. |
| `target_stat_interval_ms` | u32 | 5000 | 1-60000. Legacy; validated, not used. |
| `flush_data_period_ms`, `flush_data_min_persistence` | - | - | removed: logged as a warning and ignored. Use the write-back cache's `flush_period_ms`. |

## `targets:`

| key | type | default | meaning and constraints |
|---|---|---|---|
| `neighborhood` | u32 | 4 (code default); the shipped config sets 1 | each container registers the persistent tiers of itself and the next `neighborhood - 1` nodes in ring order, capped at the node count. Memory tiers (`ram`, `hbm`, `pinned`) are always node-local. 1 = owner-local placement. Must be 1-1024. |
| `failover_to_successor` | bool | false | route a dead owner's blobs to the first live successor container, which serves the replication module's remote copy and records changes for hand-back. Needs replication with `remote_copies >= 1` to be useful. |
| `default_target_timeout_ms` | u32 | 30000 | 1-300000. Validated, not used by the runtime on dev. |
| `poll_period_ms` | u32 | 5000 | logged only; target statistics follow `performance.stat_targets_period_ms`. |

## `dpe:` (data placement engine)

| key | type | default | meaning |
|---|---|---|---|
| `dpe_type` | string | `max_bw` | `max_bw` (alias `maxbw`): the fastest target whose score is not above the blob's score, spilling to the next when full. `round_robin` (alias `roundrobin`) and `random` ignore score order. Any other value is an error. |

## Environment variables

| variable | read by | effect |
|---|---|---|
| `CLIO_SERVER_CONF` | runtime | config file holding the `compose:` section; default `~/.clio/clio.yaml` |
| `CLIO_STORAGE_ROOT` | path expansion | root of `${CLIO_STORAGE_ROOT}` in `path` and `metadata_log_path`; default `~/.clio`. `clio-run start --disk` sets it too |
| `CLIO_CTE_POOL` | client init | `major.minor`: bind the process-wide CTE client to this pool (for example `564.0`, the chain top) instead of the default resolution |
| `CLIO_IPC_MODE` | client init | `SHM`, `IPC` or `TCP`. Unset: SHM if a local runtime's segment exists, then IPC, then TCP. Only SHM clients can use the shm cache fast path |
| `CLIO_WITH_RUNTIME` | `clio_init` | `0`: attach to a running runtime; `1`: start one in-process |
| `CLIO_WAIT_SERVER` | client init | seconds to wait for the runtime (default 30 s) |

## Runtime log lines to look for

- `cte_core Create (<pool_name>): persistent core (stated), WAL on`
- `... volatile core (inferred from its bdevs), WAL off (no metadata_log_path)`
- `refusing volatile bdev '<name>': this core is persistent`
- `persistent bdev '<name>' registered in a volatile core; its data will not survive a restart`
- `Config error: ...` (a key rejected; see "How the entry is read")
- `fresh start discarded N metadata log file(s)` (`--fresh`)
- `metadata WAL is N bytes, Kx its capacity` (snapshots not completing)
