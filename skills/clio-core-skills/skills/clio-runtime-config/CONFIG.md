# Config file reference (Context Runtime)

Every key the Context Runtime's own parser reads, verified against clio-core
`dev` (bd1dc3183) (`context-runtime/src/config_manager.cc`, `ParseYAML`).
Module keys inside `compose:` entries belong to each module's skill.

The file is YAML. Unknown keys are ignored without a warning, so a typo
silently leaves the default in place. Paths may use `${HOME}`, `${USER}`,
`${CLIO_STORAGE_ROOT}` (default `~/.clio`) or any other `${VAR}`.

## Which file is read

| order | source | notes |
|---|---|---|
| 1 | `CLIO_SERVER_CONF=<path>` | Wins over everything. An empty file is reported as a load failure and the runtime falls back to built-in defaults. |
| 2 | `~/.clio/clio.yaml` | The normal file. `import iowarp_core` (including every `clio-run` call from the pip wheel) copies `<site-packages>/iowarp_core/data/clio_default.yaml` here if it does not exist. |
| 3 | none found | Built-in defaults: port 9413, 4 workers, **no compose section**, so only the admin pool exists. CTE clients then fail. |

Notes:

- With `CLIO_TEST_MODE` set (not `0`), step 2 is skipped. Leave it unset outside tests.
- The packaged `clio_default.yaml` is never read directly by the runtime. It is
  only the seed for `~/.clio/clio.yaml`. Editing it after the first import does
  nothing; edit `~/.clio/clio.yaml`.
- Clients read the same file. A client uses it for `networking.port` and
  `runtime.ipc_namespace`, so a client with a different config (or a different
  `CLIO_PORT`) cannot find the runtime.
- Environment variables are applied after the file and win over it (ENV.md).
- To return to the shipped defaults: delete `~/.clio/clio.yaml` and run
  `python -c "import iowarp_core"`.

## Top-level sections

| section | purpose |
|---|---|
| `networking:` | ports, hostfile, startup wait for peers |
| `runtime:` | worker threads, queues, shared-memory segment sizes, state directory |
| `swim:` | gossip failure detector (off by default) |
| `viz:` | web dashboard |
| `gpu:` | GPU task-queue sizing (GPU builds only) |
| `compose:` | list of pools to create at start |

There is no `logging:` section. Logging is set by `CTP_LOG_LEVEL` and
`CTP_LOG_OUT` (ENV.md).

## `networking:`

| key | type | default | meaning |
|---|---|---|---|
| `port` | int | `9413` | Base port. The runtime binds port (peer RPC), port+1 (local server; whoever binds it is this node's runtime), port+2 (control channel for liveness; optional) and port+3 (client ROUTER). Ports port, port+1 and port+3 must all be free or the runtime refuses to start. Keep it below 49152. `CLIO_PORT` overrides. |
| `neighborhood_size` | int | `32` | Width of a ManyToOne collective neighborhood. Broadcasts and ranges reach every container regardless. |
| `hostfile` | path | unset | Multi-node: one host per line. Unset = single node. See DEPLOY.md for the format. There is no environment variable for it. |
| `wait_for_restart` | int, s | `30` | How long a starting node waits for its peers to answer. |
| `wait_for_restart_poll_period` | int, s | `1` | Interval between those retries. |

## `runtime:`

| key | type | default | meaning |
|---|---|---|---|
| `num_threads` | int | `4` | Worker threads that execute tasks. `CLIO_NUM_THREADS` overrides. |
| `queue_depth` | int | `1024` | Initial depth of each worker lane. Lanes grow on demand. |
| `local_sched` | string | `default` | Task scheduler: `default` or `local`. Any other name logs a warning and uses `default`. |
| `first_busy_wait` | int, us | `1000` when a file is loaded without the key (`10000` with no file; the shipped file sets `10000`) | How long an idle worker spins before it sleeps. |
| `client_reap_grace_s` | int, s | `30` | A dead client's shared-memory segments are freed after it has been dead, idle and without in-flight tasks this long. `0` = never while running; they are cleaned at the next start. |
| `task_progress_interval_ms` | int, ms | `5000` | A cross-node task quiet this long is probed for liveness; repeats each interval. `0` disables (a lost reply then hangs the client). `CLIO_TASK_PROGRESS_INTERVAL_MS` overrides. |
| `learning_rate` | float | `0.2` | Learning rate of the per-task cost model used by the scheduler. |
| `conf_dir` | path | `/tmp/clio_$USER` (Windows: `%TEMP%\clio_<USERNAME>`; `CLIO_MEMFD_DIR` if set) | Runtime state directory. Holds `wal/pools.<node>.bin` (the pool log: pools to re-create on restart), `wal/domain_table.<pool>.<node>.bin` (address-table WAL) and `models/` (learned cost models). On a host that clears `/tmp` at boot, set it to a persistent path. |
| `main_segment_size` | size | `0` = auto | Shared segment for task data. Auto = RAM size, clamped to half the process memory budget (cgroup-aware), and to 1 GiB off Linux. Sparse on Linux: untouched pages cost nothing. `CLIO_MAIN_SEGMENT_SIZE` overrides. |
| `metadata_segment_size` | size | `0` = auto | Shared segment for the CTE metadata cache. Same auto rule and clamps as above. |
| `client_data_segment_size` | size | `512m` | Each client process's first data segment. More are added as it fills. `0` = default. `CLIO_CLIENT_DATA_SEGMENT_SIZE` overrides. |
| `ipc_namespace` | string | empty | Suffix on shared-memory segment names. Segment names already include the user and the port; set this only to run two runtimes that share both. `CLIO_IPC_NAMESPACE` overrides. |

Sizes accept a byte count or a number with a unit: `b`, `k`, `m`, `g`, `t`,
`p` (case-insensitive; `512m`, `1GB`). A value with an unknown unit is logged
as an error and ignored.

## `swim:`

SWIM gossips liveness probes and declares a silent peer dead; recovery then
moves the dead peer's containers. **Leave it off for batch jobs.** A wide
collective (a compose across 256 nodes, a full-machine flush) can starve probe
replies long enough to declare healthy nodes dead; this was measured at 256
nodes. Turn it on only for long-lived services, sized for the widest
collective that service issues.

| key | type | default | meaning |
|---|---|---|---|
| `enabled` | bool | `false` | Run the detector. |
| `direct_probe_timeout_sec` | float, s | `300.0` | Deadline for a direct probe reply. |
| `indirect_probe_timeout_sec` | float, s | `150.0` | Deadline for a relayed probe. |
| `suspicion_timeout_sec` | float, s | `3600.0` | Suspected -> dead; triggers recovery. |

Failure handling that does not depend on SWIM (`CLIO_PROBE_SILENCE_S`,
`CLIO_NET_RETRY_TIMEOUT_S`, `CLIO_NET_DEAD_FAIL_FAST`) is environment-only; see
ENV.md.

## `viz:` (web dashboard)

The admin module serves one HTTP dashboard per node. Each module that ships a
dashboard page is mounted under `/viz/<mod_name>`. Pages include Pools, Pool,
Node and Config.

| key | type | default | meaning |
|---|---|---|---|
| `enabled` | bool | off for an embedded runtime; on for `clio-run start` | Serve the dashboard. Setting it here overrides the CLI default. `CLIO_VIZ_ENABLE` wins over the file. |
| `port` | int | `8080` | TCP port. `0` = any free port. |
| `bind` | string | `127.0.0.1` | Bind address. The dashboard exposes runtime internals and has no authentication; `0.0.0.0` publishes it to the network. |
| `max_threads` | int | `16` | HTTP threads. Keep it above ~6 (a browser holds several keep-alive sockets). |

Precedence: `clio-run start --viz/--no-viz/--viz-port/--viz-bind` >
`CLIO_VIZ_*` > `viz:` > default.

## `gpu:` (GPU builds only)

| key | type | default | meaning |
|---|---|---|---|
| `blocks` | int | `1` | GPU blocks for the GPU task path. `CLIO_GPU_BLOCKS` overrides. |
| `threads_per_block` | int | `32` | Threads per block. `CLIO_GPU_THREADS` overrides. |
| `queue_depth` | int | `16` | Tasks per GPU queue. |

## `compose:`

A YAML list. Each entry creates one pool, in list order, when the runtime
starts (and when you run `clio-run compose start <file>`). A failed entry stops
the start: the runtime logs `Compose: Failed to create pool ...` and exits.

### Fields every entry has

| key | type | default | meaning |
|---|---|---|---|
| `mod_name` | string | required | Module library name (the ChiMod's `chimod_lib_name`), e.g. `clio_bdev`, `clio_cte_core`. The runtime loads `lib<mod_name>_runtime.so` (`.dll`/`.dylib`). |
| `pool_name` | string | required | Unique name. Lookup is by name first (below), across all modules. |
| `pool_id` | string `"major.minor"` | required | E.g. `"512.0"`. Quote it: unquoted `512.0` is read as a float and may lose the form. A missing or null id fails the create with `EINVAL`. |
| `pool_query` | string | `local` | Where the compose task runs. Use `local` in the server config: every node runs its own compose at start, so `local` puts the pool on every node. Other values: `dynamic`, `broadcast`, `direct_id:<n>`, `direct_hash:<n>`, `range:<off>:<count>`, `physical:<node>`. Use `broadcast` only with `clio-run compose start` on one node of a running cluster. |
| `restart` | bool | `false` | Record the pool in this node's pool log so the next `clio-run start` re-creates it. Needed for pools created later with `clio-run compose start`. Pools in the server config are re-created at every start anyway. |
| `container_visibility` | `public` or `private` | `public` | A private pool rejects RPCs from user clients; runtime-internal callers are always allowed. |
| `container_rpc_acl` | map method name -> `public`/`private` | none | Per-method override of `container_visibility`. |

All other keys of the entry are passed, as YAML text, to the module.

### Get-or-create matching

1. If a pool with the same `pool_name` exists, it is returned as-is. Its
   module, id and parameters are **not** checked. Changing a key and running
   `compose start` again changes nothing; destroy the pool first
   (`compose stop`/`rm`) or restart the runtime.
2. Else, if a pool with the same `pool_id` exists, it is returned.
3. Else the pool is created with one container per hostfile entry.

So two entries with the same `pool_name`, or the same `pool_id`, silently
collapse into one pool.

### Ordering

Pools are created one at a time, top to bottom. An entry must come **after**
every pool it names (`next_pool_id`, `next_volatile`, `next_persistent`,
`flush_pool`, `staging_pool_id`, `entry_pool_id`, `stream_pool_id`, ...).
Put bdev pools first, CTE cores next, interposers bottom-up, and clients of
the chain (CAE, stream, filesystem) last.

A client that asks for a composed pool while the compose is still running is
held (up to 300 s) until the pool exists, so clients may connect early.

### Default pool ids

These are the ids the modules' clients assume. Reuse them unless you run two
instances of a module.

| id | `mod_name` | role | in shipped `compose:` | skill |
|---|---|---|---|---|
| 1.0 | `clio_admin` | admin, always present, never composed | n/a | this one |
| 301.0 | `clio_bdev` | DRAM block device `ram::chi_default_bdev` | yes, required | clio-bdev |
| 302.0, 303.0 | `clio_bdev` | file / DAOS device examples | commented | clio-bdev |
| 350.0 | `clio_safe_bdev` | erasure-coded array | commented | clio-bdev |
| 400.0 | `clio_cae_core` | CAE | yes | clio-cae |
| 401.0 | `clio_cae_summarizer` | summarizer | commented | clio-cae |
| 512.0 | `clio_cte_core` | persistent CTE core (CTE client default) | yes | clio-cte-core |
| 540.0 | `clio_cte_core` | volatile CTE core | yes | clio-cte-core |
| 560.0 | `clio_cte_filesystem` | clio-fs namespace | yes | clio-fs |
| 561.0 | `clio_cte_replication` | replication | yes | clio-replication |
| 562.0 | `clio_cte_compressor` | compressor | commented | clio-compressor |
| 563.0 | `clio_cte_cache` | plain cache | no | clio-cte-caches |
| 564.0 | `clio_cte_indexer` | indexer (chain top) | yes | clio-indexer |
| 565.0 | `clio_cte_stream` | stream | yes | clio-fs |
| 566.0 | `clio_cte_checkpoint` and `clio_cte_core_persist_write_thru` | both default to 566.0; give one another id | no | clio-fs, clio-cte-caches |
| 567.0 | `clio_cte_core_persist_write_back` | write-back cache | yes | clio-cte-caches |
| 568.0 | `clio_cte_core_shm_cache_writethru` | node shm cache | yes | clio-cte-caches |

Shipped compose order: 301, 512, 540, 561, 567, (562), 568, 564, 400, 565, 560.
To design a chain, use `storage-compose`.
