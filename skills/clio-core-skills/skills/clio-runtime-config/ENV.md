# Environment variables (runtime and client)

Every `CLIO_*` variable the Context Runtime and its client library read,
verified against clio-core `dev` (bd1dc3183) by searching the source for
`getenv` / `GetCompat`. Environment variables win over the config file.
Variables read only by one module (`CLIO_CTE_*`, `CLIO_BDEV_*`, `CLIO_DAOS_*`,
...) are documented in that module's skill.

**Names that older notes use but no code reads:** `CLIO_X` (the transport is
`CLIO_IPC_MODE`), `CLIO_HOSTFILE` (use `networking.hostfile`),
`CLIO_LOG_LEVEL` (use `CTP_LOG_LEVEL`), `CLIO_SCHED_WORKERS` (use
`CLIO_NUM_THREADS`), `CLIO_SHM_SIZE`, `CLIO_RUNTIME_DATA_SEGMENT_SIZE`,
`CLIO_POLL_SERVER`. Setting them does nothing.

## Config, location and identity

| variable | default | read by | meaning |
|---|---|---|---|
| `CLIO_SERVER_CONF` | unset | runtime, client | Config file path. Wins over `~/.clio/clio.yaml`. |
| `CLIO_PORT` | `networking.port` (9413) | runtime, client | Base port. Runtime and client must agree. |
| `CLIO_STORAGE_ROOT` | `~/.clio` | runtime | Value of `${CLIO_STORAGE_ROOT}` in config paths (the shipped config puts the CTE disk tier, metadata log, indexer log and clio-fs log there), the legacy restart log and `bdev_perf/`. `clio-run start --disk <path>` sets it. |
| `CLIO_MEMFD_DIR` | `/tmp/clio_$USER` (Windows `%TEMP%\clio_<USERNAME>`) | runtime, client | Per-user directory for segment links, pid records and unix sockets; also the default `runtime.conf_dir`. Point it elsewhere when `/tmp` is small or shared. Runtime and clients must agree. |
| `CLIO_IPC_NAMESPACE` | `runtime.ipc_namespace` (empty) | runtime, client | Suffix on shared-memory segment names, to run two runtimes with the same user and port. |
| `CLIO_EPHEMERAL` | unset | runtime | Non-empty and not `0`: skip the `compose:` section and the pool-log replay; start with the admin pool only. `clio-run start --ephemeral` sets it. |
| `CLIO_TEST_MODE` | unset | runtime, client | Tests only. Ignores `~/.clio/clio.yaml` and binds loopback. |
| `CLIO_RESTART_LOG` | `${CLIO_STORAGE_ROOT}/restart_log.bin` | runtime | Path of the legacy restart log (compose files registered by old versions; replayed on a recovering start). |
| `CLIO_REPO_PATH` | unset | runtime | Extra directories searched for module libraries (`lib<mod_name>_runtime.so`), `:`-separated (`;` on Windows). Searched after the runtime's own module directory and before `LD_LIBRARY_PATH`. Needed for modules you built yourself. |
| `CLIO_CHIMOD_ABI_CHECK` | on | runtime | `0` loads a module library that lacks the ABI fingerprint (built against an older runtime). Otherwise such a library is skipped with a warning. |

## Client connection

| variable | default | meaning |
|---|---|---|
| `CLIO_IPC_MODE` | unset = auto | Transport: `SHM`, `IPC` or `TCP` (any other value means TCP). Auto probes, fastest first: SHM if this host has the runtime's main segment, else IPC if the unix socket exists, else TCP. |
| `CLIO_SERVER_ADDR` | `127.0.0.1` | Address of the runtime for TCP clients. Set it to reach a runtime on another host. |
| `CLIO_BIND_ADDR` | see below | Bind address for the runtime's client ROUTER, the control channel, a single-node runtime's node 0, and each client's response listener. |
| `CLIO_WAIT_SERVER` | `30` s | How long a client waits for the runtime at init. `0` = fail at once, `-1` = wait forever. `clio-run status` uses 2 s and `stop` 5 s unless this is set. |
| `CLIO_CLIENT_RETRY_TIMEOUT` | `60` s | How long a client retries after losing the runtime. `0` = fail at once, `-1` = forever. |
| `CLIO_CLIENT_TRY_NEW_SERVERS` | `0` | Number of other hostfile entries a client tries when its runtime is gone (loads the hostfile). |
| `CLIO_WITH_RUNTIME` | unset | `1`/`true`/`TRUE`: a client process tries to become the runtime; if the port is already taken it attaches as a client. Any other value: client only. Unset: the program's default (Python `clio_init(..., default_with_runtime=False)`). |
| `CLIO_ZMQ_LOCAL_IPC` | on for macOS, off elsewhere | `1` puts the local client ROUTER on a unix socket instead of TCP port+3. |

`CLIO_BIND_ADDR` default, first match wins:

1. `CLIO_BIND_ADDR` if set.
2. `CLIO_TEST_MODE` set: `127.0.0.1`.
3. A hostfile is configured: `0.0.0.0`.
4. `CLIO_SERVER_ADDR` is not loopback (a client of a remote runtime):
   `0.0.0.0`, so the runtime can connect back to this client.
5. Otherwise `127.0.0.1`.

So a **single-node** runtime serves only local clients until you set
`CLIO_BIND_ADDR=0.0.0.0`. A loopback bind address also makes the runtime
ignore a configured hostfile. The runtime does not authenticate clients or
peers.

## Runtime sizing and tuning

| variable | default | meaning |
|---|---|---|
| `CLIO_NUM_THREADS` | `runtime.num_threads` (4) | Worker threads (>= 1). |
| `CLIO_MAIN_SEGMENT_SIZE` | `runtime.main_segment_size` (auto) | Main task-data segment, bytes or `512m`. `0` = auto. |
| `CLIO_CLIENT_DATA_SEGMENT_SIZE` | `runtime.client_data_segment_size` (512m) | First data segment of each client. `0` = 512 MB. |
| `CLIO_TASK_PROGRESS_INTERVAL_MS` | `5000` | Liveness probe of quiet cross-node tasks; `0` disables. |
| `CLIO_SHM_IN_SHARDS` | `0` = one per worker | Number of inbound shared-memory request rings. `1` = one ring. |
| `CLIO_SHM_ASYNC_SEND` | off | `1`: send SHM responses from a background thread. Helps only under heavy back-pressure; measured ~3x worse latency otherwise. |
| `CLIO_SHM_CLIENT_SPIN_US` | `50` us | How long a SHM client spins for a reply before sleeping. |
| `CLIO_GPU_BLOCKS`, `CLIO_GPU_THREADS` | `gpu.blocks`, `gpu.threads_per_block` | GPU builds only. |
| `CLIO_WIN_TIMER_MS` | unset | Windows: request this timer resolution (ms). |
| `CLIO_BOOST_STACK_SIZE` | build default | Fiber stack size, builds with Boost coroutines only. |

## Failure detection (peers)

| variable | default | meaning |
|---|---|---|
| `CLIO_PROBE_SILENCE_S` | `30` s | A peer silent this long (no probe reply, no bytes) is declared dead, and tasks waiting on it fail. |
| `CLIO_NET_RETRY_TIMEOUT_S` | `30` s | How long a send to an unreachable peer is retried before its task fails with the network-timeout code. |
| `CLIO_NET_DEAD_FAIL_FAST` | `0` | `1`: a task addressed to a peer already declared dead fails at once instead of waiting out the retry window. |

For a filesystem deployment that prefers a prompt `EIO` over a 30 s stall per
operation while a node is down, set `CLIO_PROBE_SILENCE_S=5`,
`CLIO_NET_RETRY_TIMEOUT_S=5`, `CLIO_NET_DEAD_FAIL_FAST=1`. Keep the defaults
to ride out peer restarts. SWIM (`swim:` in the config) is separate and off.

## Peer transport (node to node)

| variable | default | meaning |
|---|---|---|
| `CLIO_PEER_TRANSPORT` | `zmq` | `zmq`, `socket` (plain TCP) or `thallium` (Mercury; needs a build with Thallium, e.g. Spack `+mochi`; otherwise logs an error and uses zmq). **Every node must use the same value.** |
| `CLIO_LBM_THALLIUM_PROTOCOL` | `ofi+tcp;ofi_rxm` | Mercury protocol: `ofi+cxi` (Slingshot), `ofi+verbs;ofi_rxm` (InfiniBand/RoCE), `na+sm` (one node). |
| `CLIO_LBM_THALLIUM_LISTEN` | unset | Device or IP to listen on (`cxi0`, `hsn0`, an IP). |
| `CLIO_LBM_THALLIUM_AUTH_KEY` | unset | Mercury auth key (Slingshot `svc_id:vni`). |
| `CLIO_LBM_THALLIUM_BULK_THRESHOLD` | `64k` | Segments at least this size use RDMA bulk transfer. |
| `CLIO_LBM_THALLIUM_RPC_THREADS` | `4` | RPC handler threads. |
| `CLIO_LBM_THALLIUM_PROGRESS_THREADS` | `1` | `1` = dedicated progress thread; `0` = shared. |
| `CLIO_LBM_THALLIUM_TIMEOUT_MS` | `10000` ms | Send timeout; `0` = forever. |
| `CLIO_LBM_THALLIUM_RECV_QUEUE_MSGS` / `_BYTES` | `1024` / `1g` | Bound on received messages waiting per listener. |
| `CLIO_LBM_THALLIUM_RECV_QUEUE_WAIT_MS` | `1000` ms | How long a sender is held when that queue is full. |
| `CLIO_LBM_THALLIUM_RESOLVE_TIMEOUT_MS` | `5000` ms | Wait for a peer's address reply. |
| `CLIO_ZMQ_IO_THREADS` | `8` | ZeroMQ I/O threads. |
| `CLIO_ZMQ_MAX_SOCKETS` | `8192` | ZeroMQ socket limit. |
| `CLIO_NET_SEND_THREADS` | `0` | Extra network send threads. Experimental. |
| `CLIO_NET_TIERS`, `CLIO_NET_TIER_LANES`, `CLIO_NET_TIER_ZMQ_FROM` | unset | Size-tiered send lanes. Experimental; leave unset. |

## Logging and dashboard

| variable | default | meaning |
|---|---|---|
| `CTP_LOG_LEVEL` | `info` (build default) | `debug`, `info`, `success`, `warning`, `error`, `fatal`, or `0`-`5`. |
| `CTP_LOG_OUT` | unset (console) | Also write the log to this file. |
| `CLIO_VIZ_ENABLE` | unset | `1`/`0`: serve the dashboard. Wins over `viz.enabled`. |
| `CLIO_VIZ_PORT` | `8080` | Dashboard port; `0` = any free port. |
| `CLIO_VIZ_BIND` | `127.0.0.1` | Dashboard bind address. |
| `CLIO_VIZ_MAX_THREADS` | `16` | Dashboard HTTP threads. |
| `CLIO_VIZ_PATH` | unset | `:`-separated directories to serve dashboard assets from (edit pages without rebuilding). |

## Debugging and tests

| variable | meaning |
|---|---|
| `CLIO_ALLOW_PTRACE=1` | Linux: let `gdb -p` attach to the runtime under `ptrace_scope=1`. |
| `CLIO_FORCE_NET=1` | Route every non-Local task over the network path even on one node (benchmarks). |
| `CLIO_REQUIRE_RUNTIME=1` | HDF5 VOL/VFD adapters only: fail instead of silently passing through to the native file when no runtime is reachable. Set it in every measurement. Not read by the runtime or the Python client. |
| `CLIO_TEST_PARTITION_FILE`, `CLIO_NET_TRACE`, `CLIO_SHM_TRACE`, `CLIO_TRACE_PROBES`, `CLIO_COLL_PROF`, `CLIO_NET_QPROF`, `CLIO_EVLAT`, `CLIO_YAML_WARMUP` | Developer tracing and fault injection. Leave unset. |
