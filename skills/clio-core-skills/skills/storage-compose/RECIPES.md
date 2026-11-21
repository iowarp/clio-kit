# RECIPES.md: complete compose sections

Each recipe is a whole `compose:` list, ordered bottom-up, with its guarantees
stated. Each module's skill (`clio-bdev`, `clio-cte-core`, ...) has the full
key table. Paths use `${CLIO_STORAGE_ROOT}`,
which defaults to `~/.clio` and is overridden by `clio-run start --disk`.

## R1: scratch only (no durability, fastest)

Everything is lost at stop. Use for intermediate data and for gpu_vector
working sets with no checkpoint.

```yaml
compose:
  - mod_name: clio_bdev
    pool_name: "ram::chi_default_bdev"
    pool_query: local
    pool_id: "301.0"
    bdev_type: ram
    capacity: "0g"
  - mod_name: clio_cte_core
    pool_name: cte_main
    pool_query: local
    pool_id: "512.0"
    durability: volatile
    storage:
      - path: "ram::cte_ram_tier1"
        bdev_type: "ram"
        capacity_limit: "16GB"
        score: 1.0
    dpe: { dpe_type: "max_bw" }
```

The volatile core sits at 512.0 here because clients address 512.0 by default.

## R2: single-node durable (the dev default, without CAE and the filesystem)

- **Ack:** in DRAM.
- **Durable:** on fsync, or within about 10 s.
- **Runtime crash:** loses unflushed writes.
- **Disk loss:** loses everything.

```yaml
compose:
  - { mod_name: clio_bdev, pool_name: "ram::chi_default_bdev", pool_query: local,
      pool_id: "301.0", bdev_type: ram, capacity: "0g" }
  - mod_name: clio_cte_core            # persistent
    pool_name: cte_main
    pool_query: local
    pool_id: "512.0"
    durability: persistent
    storage:
      - path: "${CLIO_STORAGE_ROOT}/cte_disk_tier.dat"
        bdev_type: "file"
        capacity_limit: "500GB"
        score: 0.2
        persistence_level: "temporary"
    performance:
      metadata_log_path: "${CLIO_STORAGE_ROOT}/cte_metadata_log"
      transaction_log_capacity: "32MB"
    dpe: { dpe_type: "max_bw" }
  - mod_name: clio_cte_core            # volatile
    pool_name: cte_volatile
    pool_query: local
    pool_id: "540.0"
    durability: volatile
    storage:
      - { path: "ram::cte_ram_tier1", bdev_type: "ram", capacity_limit: "0g", score: 1.0 }
    dpe: { dpe_type: "max_bw" }
  - mod_name: clio_cte_replication
    pool_name: clio_cte_replication
    pool_query: local
    pool_id: "561.0"
    next_pool_id: "512.0"
    replication_factor: 1
  - mod_name: clio_cte_core_persist_write_back
    pool_name: clio_cte_core_persist_write_back
    pool_query: local
    pool_id: "567.0"
    next_volatile: "540.0"
    next_persistent: "561.0"
    fill: on_hit_count:2
    evict: lru
    flush_period_ms: 10000
    flush_fill_pct: 80
    flush_dirty_mb: 256
    fallback_to_persistent: true
```

Clients target 567.0, the chain top. A client that addresses 512.0 directly
skips the volatile tier and the write-back policy.

### R2 with every ack durable

Replace the write-back entry with
`mod_name: clio_cte_core_persist_write_thru` (`pool_id: "566.0"`, with the same
`next_volatile` and `next_persistent`). Each put now pays persistent latency.

## R3: multi-node, survive a node (replicated)

Start from R2, with the same entries on every node through a shared config and
a hostfile. Change these keys:

```yaml
  # persistent core (512.0)
    targets:
      neighborhood: 1
      failover_to_successor: true      # serve a dead owner's blobs from its successor
  # replication (561.0)
    replication_factor: 2              # +1 synchronous copy on the next node by hash
    handoff_log_path: "${CLIO_STORAGE_ROOT}/cte_handoff_log"
```

Also set `CLIO_NET_DEAD_FAIL_FAST=1` in the runtime environment, plus shorter
`CLIO_PROBE_SILENCE_S` and `CLIO_NET_RETRY_TIMEOUT_S` if prompt errors matter
more than riding out restarts.

**Guarantees.**
- An acked put is on two nodes only after the write-back flush, or after
  fsync. Replication sits *below* the write-back cache, so a dirty volatile
  copy is still single-node.
- Use write-thru if each ack itself must be on two nodes.
- Space is about 2x per fsynced byte.

## R4: survive a disk without a second node (safe_bdev)

The member bdevs come first, then the array, then a persistent core that
attaches to it:

```yaml
  - { mod_name: clio_bdev, pool_name: "/mnt/nvme0/disk0.bin", pool_query: local,
      pool_id: "302.0", bdev_type: file, capacity: "1TB" }
  - { mod_name: clio_bdev, pool_name: "/mnt/nvme1/disk1.bin", pool_query: local,
      pool_id: "303.0", bdev_type: file, capacity: "1TB" }
  - { mod_name: clio_bdev, pool_name: "/mnt/nvme2/disk2.bin", pool_query: local,
      pool_id: "304.0", bdev_type: file, capacity: "1TB" }
  - mod_name: clio_safe_bdev
    pool_name: safe_array
    pool_query: local
    pool_id: "350.0"
    max_failures: 1
    alloc_log: "/var/lib/clio/safe_array.alog"
    members:
      - { pool_name: "/mnt/nvme0/disk0.bin", node_id: 0, pool_id_major: 302, pool_id_minor: 0 }
      - { pool_name: "/mnt/nvme1/disk1.bin", node_id: 0, pool_id_major: 303, pool_id_minor: 0 }
      - { pool_name: "/mnt/nvme2/disk2.bin", node_id: 0, pool_id_major: 304, pool_id_minor: 0, parity: true }
  # persistent core storage entry:
    storage:
      - path: "safe_array"
        existing_pool_id: "350.0"
        existing_pool_module: clio_safe_bdev
        score: 0.2
        persistence_level: "long_term"   # required: unset = volatile
```

Keep replication's `num_replicas: 0`.

## R5: POSIX filesystem (clio-fs) over R2 or R3

Add the full dev top of chain, in this order, after the R2 or R3 entries:
1. shm cache (568.0): `next_pool_id: "567.0"`, `entry_pool_id: "564.0"`,
   `md_only: true`, `invalidate_on_flush: true`. Also set the write-back
   cache's `flush_pool: "568.0"`.
2. indexer (564.0): `next_pool_id: "568.0"`, `index_log_path` set, and
   `tag_re`/`blob_re` scoped if search is not needed for all files.
3. stream (565.0): `next_pool_id: "564.0"`, `staging_pool_id: "561.0"`,
   **`log_path` set**, or file sizes do not survive a restart.
4. filesystem (560.0): `next_pool_id: "564.0"`, `stream_pool_id: "565.0"`,
   `metadata_log_path` set.

Mount with the FUSE adapter (`clio_cte_fuse`). fsync is the durability point.
Directories homed on a node that never returns are lost even with replication.

## R6: data ingest (CAE) and the GPU tier

- **CAE:** compose `clio_cae_core` at 400.0 last, with
  `next_pool_id: "564.0"` and `allowed_roots` listing the data directories.
- **GPU-resident tier for gpu_vector:** add an `hbm` device to the **volatile**
  core (540.0) at score 1.0. It is an ordinary tier. The
  gpu_vector page cache is a separate thing, sized by the vector; see the
  gpu-vector skill.
- **gpu_vector checkpoints:** compose `clio_cte_checkpoint` with
  `next_pool_id` pointing at the chain top. If write-thru is also present, give
  it an id other than 566.0.
