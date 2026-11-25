---
name: clio-runtime-config
description: 'Use when installing, starting, configuring or troubleshooting the CLIO Core Context Runtime itself: where clio.yaml comes from, networking/runtime/swim/viz keys, compose entry basics, CLIO_* environment variables, clio-run start/stop/status/compose, --fresh, multi-node hostfiles, Docker, client connection. Triggers on "clio-run start", "~/.clio/clio.yaml", "CLIO_SERVER_CONF", "hostfile", "port 9413", "runtime not running". Not for designing a storage chain; use storage-compose.'
metadata:
  bundle: clio-core
  servers: none
  provenance: designed
  eval-status: scenarios-recorded
---

# Running and configuring the Context Runtime

The Context Runtime is the daemon (`clio-run start`) that every CLIO Core
module runs inside. One runtime runs per node. It reads one YAML config file,
creates the pools listed in its `compose:` section, and serves clients
(Python, C++, FUSE, HDF5 adapters) over shared memory, a unix socket or TCP.
The admin module (pool 1.0) is always present and is its control plane.

Facts here were checked against clio-core `dev` (bd1dc3183) on 2026-11-21.
Source paths in parentheses are provenance for maintainers only.

## Files in this skill

| file | read when |
|---|---|
| **CONFIG.md** | editing the config: which file is read, every key of `networking:`, `runtime:`, `swim:`, `viz:`, `gpu:`, the generic `compose:` fields, get-or-create matching, ordering, default pool ids |
| **ENV.md** | any `CLIO_*` / `CTP_LOG_*` variable: transport, port, bind address, segment sizes, failure detection, peer transport, logging, dashboard; plus names that do nothing |
| **CLI.md** | any `clio-run` command or flag, exit codes, what `--fresh` deletes |
| **DEPLOY.md** | multi-node bring-up and hostfile format, Docker, connecting a client (Python, C++), the admin module and warm restart |

## Module skills

This skill covers the runtime and the fields every compose entry has. Each
module's own keys are in its skill:

| skill | covers |
|---|---|
| `storage-compose` | designing a whole chain: which modules, order, guarantees |
| `clio-bdev` | `clio_bdev` devices (301.0, file/DAOS tiers) and `clio_safe_bdev` |
| `clio-cte-core` | the CTE core (512.0, 540.0) and the blob API |
| `clio-cte-caches` | write-back (567.0), write-thru, shm cache (568.0), cache (563.0) |
| `clio-replication` | replication (561.0) |
| `clio-compressor` | compressor (562.0) |
| `clio-indexer` | indexer (564.0) |
| `clio-fs` | filesystem (560.0), stream (565.0), checkpoint, FUSE |
| `clio-cae` | CAE (400.0) and the summarizer (401.0) |

## Quick start (pip)

```bash
pip install iowarp-core
python -c "import iowarp_core"     # seeds ~/.clio/clio.yaml from the packaged default
nohup clio-run start > ~/clio-run.log 2>&1 &
clio-run status                    # "clio runtime (port 9413): RUNNING (pid N)", exit 0
clio-run compose list              # pools from the compose: section
clio-run stop
```

`clio-run start` stays in the foreground; it has no daemon flag. The dashboard
is at `http://127.0.0.1:8080` while it runs.

## Where the config comes from

1. `CLIO_SERVER_CONF=<file>` if set.
2. Else `~/.clio/clio.yaml`. The first `import iowarp_core` (any `clio-run`
   from the wheel does this too) copies
   `<site-packages>/iowarp_core/data/clio_default.yaml` there if missing. That
   packaged file is the commented reference; the runtime never reads it
   directly.
3. Else built-in defaults with **no compose section**: only the admin pool
   exists and every CTE client fails.

Environment variables override the file. Clients read the same file for the
port, so runtime and clients must agree on `CLIO_SERVER_CONF` and `CLIO_PORT`.

## Config file at a glance

```yaml
networking:
  port: 9413                 # binds 9413, 9414, 9415 (optional), 9416
  # hostfile: /etc/clio/hostfile   # multi-node; unset = single node
  wait_for_restart: 30       # s to wait for peers at start
runtime:
  num_threads: 4             # worker threads
  # conf_dir: /var/lib/clio  # pool log + address-table WAL; default /tmp/clio_$USER
  # main_segment_size: 1g    # default auto (RAM, clamped to half the memory budget)
  # client_data_segment_size: 512m
swim:
  enabled: false             # keep off for batch jobs
# viz: {enabled: true, port: 8080, bind: "127.0.0.1"}
compose:
  - mod_name: clio_bdev      # library name
    pool_name: "ram::chi_default_bdev"   # unique; matched first
    pool_query: local        # each node composes its own container
    pool_id: "301.0"         # "major.minor", quoted
    bdev_type: ram           # module keys follow (clio-bdev skill)
    capacity: "2g"
```

There is no `logging:` section: set `CTP_LOG_LEVEL` (`debug` ... `fatal`) and
`CTP_LOG_OUT=<file>`. Full tables: CONFIG.md.

## Compose rules every entry follows

- Required: `mod_name`, `pool_name`, `pool_id` (`"major.minor"`). Use
  `pool_query: local` in the server config.
- Get-or-create: an existing pool with the same `pool_name` is reused
  **unchanged** (no check of module, id or keys); else one with the same
  `pool_id`; else a new pool with one container per hostfile entry.
- Order: an entry must come after every pool it names (`next_pool_id`,
  `staging_pool_id`, ...). The first failing entry aborts `clio-run start`.
- `restart: true` matters only for pools added later with
  `clio-run compose start`: it puts them in the pool log so the next start
  re-creates them.
- Default ids are in CONFIG.md; `clio_cte_checkpoint` and
  `clio_cte_core_persist_write_thru` both default to 566.0.

## Workflow

1. **Install and seed the config** (quick start above). Edit
   `~/.clio/clio.yaml`, or keep your own file and export `CLIO_SERVER_CONF`
   for the runtime **and** every client.
2. **Size it.** `runtime.num_threads` to the cores you give CLIO. In a
   container or on a small machine, set explicit capacities for DRAM tiers and
   `ram` bdevs (`0g` means "from machine DRAM").
3. **Choose the modules** with `storage-compose`, then fill each entry with
   its module skill.
4. **Start**: `clio-run start` (recovers state) or `clio-run start --fresh`
   (discards this node's state; see CLI.md for exactly what). On a cluster,
   follow DEPLOY.md: same config, same hostfile order, start on every node.
5. **Verify**: `clio-run status` exit 0, `clio-run compose list` shows every
   pool, the dashboard's Pools page shows a container per node, and a client
   round trip works:

   ```python
   import iowarp_core, clio_cte_core_ext as cte
   assert cte.clio_init(cte.RuntimeMode.kClient)
   assert cte.initialize_cte("", cte.PoolQuery.Dynamic())
   ```

6. **Stop** with `clio-run stop` on every node. For durability tests use
   `kill -9` and a plain `clio-run start`, never `--fresh`.

## Connecting clients

- Same host: nothing to set. The client auto-picks SHM, then the unix socket,
  then TCP. Force one with `CLIO_IPC_MODE=SHM|IPC|TCP`. (`CLIO_X` is not read.)
- Another host: `CLIO_SERVER_ADDR=<runtime host>` on the client. A single-node
  runtime binds loopback, so start it with `CLIO_BIND_ADDR=0.0.0.0`; with a
  hostfile it already binds `0.0.0.0`. The runtime connects back to the
  client, so the client must be reachable from it.
- A client waits `CLIO_WAIT_SERVER` (30 s) for the runtime, then fails with
  `CRITICAL ERROR: Cannot connect to local server.`
- `CLIO_WITH_RUNTIME=1` turns a client into an embedded runtime if none is
  running. It dies with the process and starts with fresh semantics. Use a
  real `clio-run start` for data you keep.

## Pitfalls and failure signatures

| symptom | cause | fix |
|---|---|---|
| `refusing to start half-bound`, or `start` exits at once | port 9413, 9414 or 9416 is in use (another runtime, another user's runtime, another program) | `clio-run status`; stop the other runtime, or give this one another base port with `networking.port` / `CLIO_PORT` (clients need the same value) |
| Client: `Cannot connect to local server` after 30 s | runtime not started yet, crashed, or started with another config/port | start the runtime first; check `clio-run status`; export the same `CLIO_SERVER_CONF`/`CLIO_PORT` in both shells |
| Runtime up, but CTE calls fail and `compose list` shows only `admin` | no config found (`CLIO_TEST_MODE` set, wrong `CLIO_SERVER_CONF`, empty file) or `--ephemeral` | check the `Config at:` log line; `import iowarp_core` to seed `~/.clio/clio.yaml` |
| `Compose: Failed to create pool X ... return code: N`, start exits | bad module keys, a missing module library, an entry ordered before a pool it names, or a missing `pool_id` (`EINVAL`) | read the module's error just above; fix order/keys; check the module ships in your install |
| Changed a compose key, nothing changed | get-or-create by `pool_name` reused the old pool | `clio-run compose stop <file>` then `start`, or restart the runtime |
| `Permission denied` on a `/tmp/...` path at start; `RegisterTarget` rc=1 | files or `/tmp/clio_$USER` left by an earlier run as root (e.g. `sudo -E clio-run start`) | `sudo chown -R $USER /tmp/clio_$USER` and the listed files; never run the runtime with sudo |
| `status` exit 2 `STOPPED (stale artifacts)` | earlier crash or `kill -9` | harmless: the next `clio-run start` removes leftovers from dead processes |
| SIGBUS or OOM kill under load; `clamping` lines in the log | container/cgroup memory limit; DRAM tiers sized from host RAM | raise `mem_limit`; set explicit tier capacities; `shm_size` does not help (segments are memfd) |
| Hostfile node never starts / joins the wrong node | `#` comment lines in the hostfile become host names; entries out of order between nodes; node not listed | host names only, identical file on every node |
| Healthy nodes declared dead mid-run, tasks return route errors | `swim.enabled: true` with timeouts shorter than a wide collective | keep SWIM off for batch jobs |
| A fs operation stalls ~30 s while a node is down | default retry window | `CLIO_NET_RETRY_TIMEOUT_S=5 CLIO_PROBE_SILENCE_S=5 CLIO_NET_DEAD_FAIL_FAST=1` |
| Pools added with `compose start` gone after reboot | pool log lives in `runtime.conf_dir` = `/tmp/clio_$USER` | set `runtime.conf_dir` to persistent storage, or add the pools to the config's `compose:` |
| Data gone after restart | started with `--fresh`, or an embedded runtime (`CLIO_WITH_RUNTIME=1`) | plain `clio-run start`; durability itself is module config (`storage-compose`) |

## Availability

The runtime, `clio-run`, the admin module and `clio_bdev` ship in every
install (pip `iowarp-core`, conda, Spack, deb, `iowarp/deploy-cpu`). The pip
wheel has no C++ headers or CMake package: C++ clients and your own modules
need a source, conda or Spack install.
