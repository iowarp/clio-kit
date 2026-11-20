# MODULES.md: the modules CLIO Core ships

Every entry in `compose:` has `mod_name` (the module's `chimod_lib_name`),
`pool_name`, `pool_query` (always `local` for these), and `pool_id`. Every other
key is passed to the module's `CreateParams::LoadConfig`. Default pool ids are
the constants that clients assume. Moving a pool is safe only if every pool
and client that names it moves too.

## Pool id map (dev)

| id | module | in default compose |
|---|---|---|
| 301.0 | `clio_bdev` (`ram::chi_default_bdev`) | yes (required) |
| 350.0 | `clio_safe_bdev` | example, commented out |
| 400.0 | `clio_cae_core` | yes |
| 401.0 | `clio_cae_summarizer` | example, commented out |
| 512.0 | `clio_cte_core` (persistent) | yes |
| 540.0 | `clio_cte_core` (volatile) | yes |
| 560.0 | `clio_cte_filesystem` | yes (built only with FUSE/POSIX/VFD adapters or `CLIO_CTE_ENABLE_FILESYSTEM`) |
| 561.0 | `clio_cte_replication` | yes |
| 562.0 | `clio_cte_compressor` | example, needs `CLIO_CTE_ENABLE_COMPRESS=ON` |
| 563.0 | `clio_cte_cache` | no (superseded by the write-back and shm caches) |
| 564.0 | `clio_cte_indexer` | yes |
| 565.0 | `clio_cte_stream` | yes |
| 566.0 | `clio_cte_checkpoint` | no |
| 566.0 | `clio_cte_core_persist_write_thru` | no. **Collides with checkpoint**: give one an explicit other id. |
| 567.0 | `clio_cte_core_persist_write_back` | yes |
| 568.0 | `clio_cte_core_shm_cache_writethru` | yes |

The admin's get-or-create matches pools by **name** first, then by id. A
second pool with a taken id is reported as success and never created. Check
for duplicate ids yourself.

---

## Runtime layer (`clio::run`)

### admin (`context-runtime/modules/admin`)
- Always present. Creates and destroys pools, stops the runtime, runs
  heartbeats and SWIM, and handles recovery and migration.
- Warm restart re-composes pools from `<conf_dir>/restart/*.yaml`.
- The address table is logged synchronously to `<conf_dir>/wal/`.
- Failure knobs are environment variables only:
  - `CLIO_PROBE_SILENCE_S` (30)
  - `CLIO_NET_RETRY_TIMEOUT_S` (30)
  - `CLIO_NET_DEAD_FAIL_FAST` (0)

  A filesystem deployment usually wants 5, 5 and 1: a prompt EIO is better than
  a 30 s stall.

### bdev: block device (`clio_bdev`)
- **Types:** `ram`, `file`, `hbm` (device memory), `pinned` (pinned host),
  `noop`, `s3`, `gcs`, `daos` (needs `-DCLIO_ENABLE_DAOS=ON`).
- **Keys:** `bdev_type`, `capacity` (`0g` = 80% of DRAM), `alloc_log`,
  `growth_unit`, `persistence_level`, `io_depth`, `alignment`.
- **Allocator durability:**
  - A file bdev logs its allocations to `<pool_name>.alloc_log` and rebuilds the
    allocator from it on every start, so a live block is never handed out
    twice.
  - `start --fresh` deletes the log, unless `alloc_log` is set explicitly.
  - A DAOS bdev keeps no log unless you give it a local `alloc_log`. Without
    one, a restart that keeps metadata can reuse live blocks.
- **Data durability:** a write completes when the I/O completes. It is not
  fdatasynced until `Sync`, which is what the CTE's fsync path calls.
- **Fault injection (tests):** while `<pool_name>.fail` exists, every I/O to
  the device fails.
- You rarely compose file bdevs by hand: a CTE core creates one per `storage[]`
  entry.

### safe_bdev: erasure-coded array (`clio_safe_bdev`)
- One logical device over member bdevs: data members plus up to `max_failures`
  Reed-Solomon parity members. Members are composed **before** it. They can be
  added at runtime (`AddBdev`) and rebuilt (`RecoverBdev`).
- **Keys:** `max_failures`, `alloc_log`, `distributed`, `intent_sync`,
  `members[{pool_name,node_id,pool_id_major,pool_id_minor,parity}]`.
- **Consistency:** a write is acked only after every stripe it touched has
  current parity.
- **Durability:**
  - Each write first records its stripes as dirty in the log. With
    `intent_sync: true` (the default), that record is fsynced, so parity stays
    consistent across a power loss.
  - `intent_sync: false` stays consistent across a process crash only, and
    creates small files about 10x faster.
  - A degraded write saves the down member's chunk to `<alloc_log>.journal`.
    This closes the write hole.
  - Membership, including runtime changes, is restored from
    `<alloc_log>.members`.
- **Failure behavior:** a faulty member's chunks are reconstructed from the
  survivors. Beyond `max_failures`, requests fail with EIO, **never wrong
  bytes**.
- **`distributed: true`** puts members on different nodes. A member whose node
  dies is marked faulty.
- **To use it as a CTE tier:** set `storage[].existing_pool_id: "350.0"` and
  `existing_pool_module: clio_safe_bdev`, plus an explicit `persistence_level`.

---

## CTE: Context Transfer Engine (`clio::cte`)

### core (`clio_cte_core`): blobs in tags over tiered targets
- **Data model:** tags (named containers) of blobs.
  - Each blob has one **owner container**, chosen by `hash(tag, blob)`.
  - Each tag is found by `hash(name)`.
  - There is no metadata server.
- **Durability class:** `durability: volatile|persistent`. If the key is
  omitted, it is inferred from the devices.
  - A persistent core refuses a volatile device. A volatile core accepts a
    persistent device with a warning.
  - A volatile core writes no WAL and starts empty.
  - The class can be changed at runtime with `AsyncSetDurability`, until the
    next restart.
- **Storage keys** (`storage[]`): `path`, `bdev_type`, `capacity_limit`,
  `score` (0 slow to 1 fast; -1 auto), `persistence_level`
  (`volatile|temporary|long_term`), `read_cache` (memory tiers only: holds read
  copies, never primaries, never WAL'd), `growth_unit`, `existing_pool_id`,
  `existing_pool_module`.
  - Bdev pool ids are `<core major + device index>.<1 + node>`.
- **Other keys:**
  - `dpe.dpe_type`: `max_bw` (prefers the hottest tier whose score is at most
    the blob's score; a new blob scores 1.0), `round_robin`, or `random`.
  - `targets.neighborhood`: persistent tiers of the next N-1 nodes are also
    registered. Memory tiers are always node-local.
  - `targets.failover_to_successor`.
  - `performance.metadata_log_path`, `transaction_log_capacity`,
    `flush_metadata_period_ms` (5000), `fsync_mode` (`durable|deferred`),
    `stat_targets_period_ms`.
  - At the top level: `organizer` (`none|frecency|grayscott|...`),
    `organizer_tasks`, `organizer_period_ms`.
- **Metadata WAL:**
  - The WAL is sharded per worker, with one global sequence number. Replay
    merges all shards in sequence order.
  - Each record is flushed to the OS when written, so it survives a process
    crash.
  - `fsync` happens every `flush_metadata_period_ms`, on `SyncTag`, and on
    close, so power loss can lose up to one period.
  - A snapshot is written when the WAL exceeds its capacity.
  - `--fresh` deletes the WAL and the snapshot.
- **Restart:**
  - Blocks on volatile targets are dropped.
  - Cache copies are dropped.
  - Restored blocks are reserved in the allocators.
- **Eviction:** `Evict` removes the lowest-score blobs from a tier. A put that
  needs room evicts only blobs written as *droppable*. Droppable is write-once:
  a later non-droppable put to the same blob is refused (rc 14).
- **Routing:**
  - Blob verbs go to the owner (DirectHash).
  - `TagQuery`, `BlobQuery`, `GetContainedBlobs`, `GetTagSize` and `Evict` run
    per container. Pass Broadcast for a cluster-wide answer.

### core_persist_write_back (`clio_cte_core_persist_write_back`, 567.0): the default policy cache
- Sits over two disconnected cores: `next_volatile` (required) and
  `next_persistent` (optional; omit it for a volatile-only cache).
- **Writes:**
  - A put lands in the volatile core on the blob's **owner node** and is acked
    there. Puts sent from other nodes are forwarded to the owner, so two nodes
    never hold diverging dirty copies.
  - Partial writes first fetch the rest of the blob from persistent.
  - Appends, replaces and conditional puts write the copy out and run on
    persistent.
- **Flushing:**
  - Dirty ranges, and only those, are flushed through `next_persistent`.
  - A flush runs periodically (`flush_period_ms`, 10000), when the cache is
    `flush_fill_pct` full, after `flush_dirty_mb` of new dirty data, on
    `SyncTag`/`FlushData`, before eviction, and at `clio-run stop` (bounded by
    `--grace-period`).
- **Reads:** served from the volatile copy if there is one, else from
  persistent. `fill` decides whether to copy the blob into the volatile core
  (`on_hit_count:N`, `on_read_miss` or `none`). `evict` is `lru` or `lfu`.
- **When the volatile core is full:** `fallback_to_persistent: true` writes
  straight to persistent. `false` fails the put with ENOSPC, so throughput
  never silently drops to disk speed.
- **`flush_pool`:** lands each flushed range through the shm cache so that it
  invalidates sharers once per range.

### core_persist_write_thru (`clio_cte_core_persist_write_thru`, 566.0)
- Takes the same keys as write-back. `flush_period_ms` and
  `fallback_to_persistent` are ignored.
- **Ack:** only after the persistent write. The volatile copy is optional and
  never dirty, so nothing is lost on a crash.
- **Cost:** measured slower than write-back on every workload of the Ares tier
  suite. Use it when every ack must already be durable.

### replication (`clio_cte_replication`, 561.0): reliability
- **Keys:**
  - `replication_factor` (total durable copies; default 1).
  - `remote_copies` (default factor − 1): copies on the next containers by
    hash.
  - `num_replicas` (default 0): extra fixed persistent copies on the owner's
    own node. They protect against losing a device, not a node.
  - `cache_score` (1.0), `replica_score` (0.2), `replicate_period_ms` (50),
    `handoff_log_path`.
- **Ordering:** the primary is written first. Remote copies are written
  **synchronously, before the ack**. Same-node `num_replicas` are written by an
  async sweep (synchronous when the period is 0).
- **Failover:** this needs the core's `failover_to_successor: true`.
  - While the owner is down, its successor serves reads and takes writes as
    shadow copies. It records them for hand-back.
  - On restart, the owner pulls the hand-backs before serving.
  - Without `handoff_log_path`, the hand-back record lives only in memory. If
    the successor restarts first, the record is lost.
- **Reads** fall back to a replica and re-cache the primary.
- **Space:** about `1 + num_replicas + remote_copies` per fsynced byte.

### core_shm_cache_writethru (`clio_cte_core_shm_cache_writethru`, 568.0)
- **What it is:** one shared-memory segment per node, mapped by local client
  processes. It caches tag and blob metadata (and, unless `md_only`, bytes). A
  hit completes inside the client with no task.
- **Binding:** `entry_pool_id` names the pool that clients bind to through
  `CLIO_CTE_CLIENT_INIT` (the indexer, by default). A TCP or remote client
  cannot map the segment and stays on the task path.
- **Coherence:**
  - Each entry's home node tracks which nodes cache it and invalidates them
    when it changes.
  - With `invalidate_on_flush: true` (valid only when the write-back cache's
    `flush_pool` is this pool), invalidation happens once per flushed range.
    Cross-node readers can then see stale cached metadata until the writer's
    data flushes.
- **Keys:** `capacity`, `md_only`, `data_pool`/`data_capacity` (extent mode),
  `flush_*`, `max_tags`, `max_blobs`, `max_fill`, `evict_sample`,
  `half_life_ms`, `nospace_wait_ms`.
- **No WAL of its own.** Dirty entries are never evicted. `SyncTag` waits on
  its log, and a flush error sticks to the tag.
- Design notes: `context-transfer-engine/core_shm_cache_writethru/DESIGN.md`
  and `HANDOFF.md`.

### indexer (`clio_cte_indexer`, 564.0): search
- Owns `SemanticSearch` (BM25 term index); the core no longer serves it.
- **Write cost:** a put costs only an O(1) dirty-key enqueue. A sweep
  (`index_sweep_period_ms`, 100) tokenizes asynchronously, and a search drains
  pending keys first, so search is read-your-writes.
- **Persistence:** its own snapshot and WAL at `index_log_path`. A restart
  restores the index and never rescans storage. Keys acked after the last WAL
  record stay stale until `ReindexScan` runs.
- **Placement:** compose it **above** the compressor, so that it reads logical
  bytes.
- **Memory:** the index is held in memory and grows with every matching blob.
  Scope it with `tag_re`/`blob_re`. `CLIO_INDEXER_PASSIVE=1` makes it a pure
  forwarder.

### compressor (`clio_cte_compressor`, 562.0)
- Compresses on the way down and decompresses on the way out.
- **Codec choice:** per put (`dynamic_compress_`: skip, static or dynamic).
  `default_compress` applies to writers that name no codec. **FUSE and Python
  name none**, so without it nothing is compressed.
- **Keys:** `default_compress`, `default_preset`, `tracking_enabled` (steers a
  tag's copy toward its last reader), model paths, `trace_folder_path`.
- **Placement:** below the indexer and above the policy cache.

### stream (`clio_cte_stream`, 565.0): byte streams
- **Layout:** 1 MiB page blobs, with an authoritative logical size per stream
  held by a home container.
- **Appends:** an append is staged on the writer's node and returns at once.
  Every 1 ms, each node ships its batch to the home, where the batch is ordered
  and copied into pages. `Flush` waits for this node's appends to merge.
- **Keys:** `next_pool_id` (the chain top), `staging_pool_id` (default:
  replication, durable and uncached), `log_path`. **Without `log_path`, file
  sizes do not survive a restart.**
- The filesystem creates the stream pool if it is not composed. Compose it
  before the filesystem to configure it.

### filesystem (`clio_cte_filesystem`, 560.0): clio-fs
- **Mapping:** paths map to tags, and offsets to 1 MiB page blobs, through
  `next_pool_id` (required).
- **Namespace:** there is no metadata server. Directory rows live in table
  blocks, each with a home container. The home pushes every change to the
  nodes that cache the block before it acks.
- **Directory journal:** `metadata_log_path` names a per-node journal of the
  blocks that node homes (`<path>.dirs.<container>`, fsynced on
  fsync/fsyncdir and every 1 s). **It is node-local: a node that never returns
  takes its directories with it.**
- **POSIX semantics:**
  - close-to-open on size.
  - fsync is the durability point. It returns EIO if a peer died during the
    unsynced window.
  - The FUSE attribute TTL is 1 s on a single node and 0 on multi-node
    (`CLIO_FUSE_ATTR_CACHE_S`).
  - `CLIO_FUSE_READAHEAD_PAGES` can serve a page that is stale by up to the
    attribute TTL.
- **Keys:** `dir_split_entries`, `dir_log_compact_bytes`, `case_insensitive`
  (fixed at volume creation), `data_score`/`data_score_head_pages` (keeps
  metadata and file heads on the fastest tier), `stream_pool_id`.

### checkpoint (`clio_cte_checkpoint`, 566.0)
- The fault handler behind `gv::Vector::Copy`'s lazy copy-on-write.
- A miss on the copy tag materializes the blob from the source tag.
- Only `next_pool_id`. Compose it only for gpu_vector checkpoints, and move it
  off 566.0 if write-thru is also composed.

### cache (`clio_cte_cache`, 563.0): node-local copy layer
- Built, but not in the default chain: the write-back cache and the shm cache
  cover its role.
- **Behavior:** write-through. It keeps a node-local untransformed copy in the
  core's `REPLICA_CACHE` slot, and the owner invalidates the other copies.
  Copies are dropped at restart.
- Prefer the default chain for new deployments.

---

## CAE: Context Assimilation Engine (`clio::cae`)

### core (`clio_cae_core`, 400.0)
- **Ingest:** `ParseOmni` imports files and HDF5 into CTE tags. CEE calls it.
  The `reference` format records source extents instead of copying: a CTE miss
  reads straight from the source file (zero-copy assimilation).
- **Routing:** it forwards its data path to `next_pool_id`. Point that at the
  chain top (564.0), so that imported data is indexed.
- **Security:** `allowed_roots` restricts the paths CAE may read and write.
  The runtime does not authenticate clients, so set it.
- **Never at 512.0** (method-id collision).

### summarizer (`clio_cae_summarizer`, 401.0)
- An interposer that calls an LLM (`label_api: ollama|openai`) for puts
  matching `label_matches` and stores `{blob}_label`.
- Inference is **synchronous on the worker**.
- To enable it, point CAE's `next_pool_id` at it.
- Read tokens from `label_api_key_file`/`label_api_key_env`, never from the
  YAML.
