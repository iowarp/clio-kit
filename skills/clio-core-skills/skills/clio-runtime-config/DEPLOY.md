# Deploying the runtime: multi-node, Docker, clients, admin

Verified against clio-core `dev` (bd1dc3183).

## Multi-node

### Hostfile

Set `networking.hostfile: /path/to/hostfile` in the config. There is no
environment variable for it.

```
node01
node02
node[03-08]
10.0.0.20;10.0.0.21
```

- One host per line, or several separated by `;`.
- `prefix[a-b,c]suffix` expands a range; leading zeros are kept when both
  ends have the same width (`node[01-16]-ib`).
- An entry may be `host:port` so several runtimes share one host; a runtime
  takes the entry whose port equals its own `networking.port`.
- **No comments and no blank-line tricks.** Whitespace is stripped, and a line
  starting with `#` becomes a host name. Keep the file to host names only.
- Node ids are the entry positions, starting at 0. Every pool gets one
  container per entry, and container id = node id.
- A node finds itself by matching its hostname (full or short) or one of its
  interface IPs against the entries. A node that is not listed fails to start.

### Bring-up

1. Put the **same** config file and the **same** hostfile on every node (same
   path or the same contents). What must match on all nodes:
   - the hostfile, in the same order (node ids come from the order);
   - `networking.port` / `CLIO_PORT`;
   - the `compose:` list: same `pool_name`, `pool_id`, module and order on
     every node (each node composes its own containers with
     `pool_query: local`);
   - `CLIO_PEER_TRANSPORT` and, for Thallium, the protocol;
   - the installed CLIO Core version.
   Paths inside module entries (`${CLIO_STORAGE_ROOT}/...`, device paths)
   must exist on each node but are each node's own.
2. Open ports base..base+3 (9413-9416 by default) between nodes, plus 8080 if
   you want the dashboard remotely.
3. Start the runtime on every node, at about the same time:

   ```bash
   # with pdsh
   pdsh -w node[01-08] 'nohup clio-run start > ~/clio-run.log 2>&1 &'
   # or under a batch job, one per node
   mpirun -np 8 --map-by ppr:1:node clio-run start &
   ```

   A starting node waits `networking.wait_for_restart` (30 s) for peers.
4. Check each node: `clio-run status` (exit 0) and
   `clio-run compose list` (same pools everywhere).
5. Stop: `clio-run stop` on every node (`pdsh -w ... clio-run stop`).

With a hostfile the runtime binds `0.0.0.0` on its peer and client ports.
Nothing authenticates peers or clients; keep the cluster network private.

### Node failure

| knob | default | effect |
|---|---|---|
| `CLIO_PROBE_SILENCE_S` | 30 s | silent peer declared dead; tasks waiting on it fail |
| `CLIO_NET_RETRY_TIMEOUT_S` | 30 s | sends to an unreachable peer are retried this long |
| `CLIO_NET_DEAD_FAIL_FAST` | 0 | 1 = tasks to a dead peer fail at once |
| `swim.enabled` | false | gossip detector that also moves containers off dead nodes; off for batch jobs |

A restarted node rejoins by running `clio-run start` again (recovering) with
the same config. Data availability while a node is down depends on the chain
(see `clio-replication`).

## Docker

The image `iowarp/deploy-cpu:latest` (Ubuntu 24.04) has CLIO Core in
`/usr/local`, a user `iowarp`, and the shipped default config at
`/home/iowarp/.clio/clio.yaml`. Its default command is a shell; run the
runtime with `clio_run start` (or `clio-run start`).

```yaml
services:
  iowarp:
    image: iowarp/deploy-cpu:latest
    hostname: iowarp
    volumes:
      - ./clio.yaml:/etc/iowarp/clio.yaml:ro
      - clio-data:/home/iowarp/.clio-data
    environment:
      - CLIO_SERVER_CONF=/etc/iowarp/clio.yaml
      - CLIO_STORAGE_ROOT=/home/iowarp/.clio-data   # persistent tiers/logs on a volume
      - CLIO_BIND_ADDR=0.0.0.0                       # else the published port is unreachable
    ports:
      - "127.0.0.1:9413-9416:9413-9416"              # loopback-only on the host
    mem_limit: 8g
    command: ["clio_run", "start"]
    restart: unless-stopped
volumes:
  clio-data:
```

Memory:

- On Linux the shared segments are `memfd` objects, **not** files in
  `/dev/shm`. `shm_size` does not limit them; the container's memory limit
  does. The main and metadata segments clamp themselves to half the
  container's memory budget.
- Pages written to segments and DRAM tiers count against `mem_limit`.
  Running out shows as SIGBUS or the OOM killer, not a clean error. Give DRAM
  tiers and `ram` bdevs explicit capacities well below the limit instead of
  `0g` (which sizes from machine DRAM).
- `runtime.conf_dir` defaults to `/tmp/clio_iowarp` inside the container, which
  is lost with the container. Point it at a volume if pools composed later
  with `restart: true` must survive a container restart.

Clients in another container cannot use SHM. Use TCP: set
`CLIO_SERVER_ADDR=<runtime container or host>` in the client; the runtime
connects back to the client, so the client must be reachable from the
runtime's container (same Docker network, or `CLIO_BIND_ADDR` on the client
set to an address the runtime can reach).

Multi-node in Docker: give containers fixed IPs on one network, list the IPs
in a hostfile mounted into every container, and set `networking.hostfile`.

## Connecting a client

A client is any process that calls CLIO: Python, a C++ program, the FUSE
mount, the HDF5 adapters. It reads the same config file (for the port) and
finds the runtime on this host unless `CLIO_SERVER_ADDR` says otherwise.

Transport (`CLIO_IPC_MODE`; unset = auto, which picks the first that works):

| mode | when | notes |
|---|---|---|
| `SHM` | same host as the runtime, same user | lowest latency (~5x faster than IPC, ~190x faster than TCP, measured) |
| `IPC` | same host | unix socket `<memfd dir>/clio_<port>.ipc` |
| `TCP` | any host | runtime's client port is base+3 (9416). Needs the runtime bound beyond loopback. |

### Python (pip wheel)

```python
import iowarp_core                    # sets library paths; seeds ~/.clio/clio.yaml
import clio_cte_core_ext as cte

ok = cte.clio_init(cte.RuntimeMode.kClient)          # waits up to CLIO_WAIT_SERVER (30 s)
assert ok, "no runtime: run `clio-run start` first"
assert cte.initialize_cte("", cte.PoolQuery.Dynamic())
# ... use the CTE client; see the clio-cte-core skill
```

`clio_init(mode, default_with_runtime=False, is_restart=False)`:
`default_with_runtime=True` (or `CLIO_WITH_RUNTIME=1`) makes this process
start a runtime if none is running, and attach if one is. That runtime lives
and dies with the Python process, does not serve the dashboard, and starts
with `--fresh` semantics unless `is_restart=True`. Prefer a separate
`clio-run start` for anything you want to keep.

`clio_runtime_ext` (also in the wheel) exposes `clio_init(mode)` (0 = client),
`clio_finalize()`, `async_monitor(pool_query, query)` and
`stop_runtime(pool_query, grace_period_ms=5000)`.

### C++ (source, conda or Spack install; the wheel has no headers)

```cpp
#include "clio_runtime/clio_runtime.h"

int main() {
  if (!clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient)) return 1;
  // ... module clients ...
  return 0;  // CLIO_RUNTIME_FINALIZE runs at exit
}
```

Link `clio::run::cxx` plus the module clients (`find_package(clio-core)`).
`CLIO_INIT(mode, default_with_runtime=false, is_restart=false)` has the same
semantics as the Python call.

### Is the runtime up?

```bash
clio-run status; echo "exit=$?"      # 0 RUNNING, 1 STOPPED, 2 STOPPED with leftovers
clio-run compose list                # the pools you expect, with their ids
clio-run viz status                  # dashboard URL, default http://127.0.0.1:8080
```

The dashboard's Pools page lists every pool and its containers per node.

## The admin module (`clio_admin`, pool 1.0)

Always present; created by the runtime itself, never composed; the runtime
refuses to start without it. It is the runtime's control plane:

- **Pools**: get-or-create and destroy pools (`compose`, client `Create`
  calls), list containers, migrate containers between nodes.
- **Networking**: sends and receives task inputs/outputs between nodes and to
  TCP/IPC clients; client connect and heartbeat.
- **Liveness**: heartbeat, peer silence detection, SWIM (if enabled),
  recovery plans, the task-progress probe for quiet cross-node tasks.
- **Lifecycle**: stop runtime (drain + teardown), flush, node add (`--induct`),
  shutdown announcement, reaping dead clients' segments.
- **Monitoring**: `worker_stats`, `pool_stats://...`, `system_stats` queries,
  and the web dashboard.

**Warm restart.** On a recovering `clio-run start`:

1. every `compose:` entry is created through its module's restart path, so
   each module replays its own logs;
2. the **address-table WAL** (`<conf_dir>/wal/domain_table.<pool>.<node>.bin`)
   is replayed, so containers that were migrated or recovered onto other nodes
   stay where they were;
3. the **pool log** (`<conf_dir>/wal/pools.<node>.bin`) is replayed: every
   durable pool (compose `restart: true`, or created by a client as
   persistent) not already present is re-created in creation order.

`--fresh` deletes both logs for this node. Both live in `runtime.conf_dir`,
which defaults to `/tmp/clio_$USER`; on a host that clears `/tmp` at boot,
pools that are not in the config file are forgotten after a reboot unless
`conf_dir` points at persistent storage.
