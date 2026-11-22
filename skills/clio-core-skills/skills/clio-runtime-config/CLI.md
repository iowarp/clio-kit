# `clio-run` command reference

Verified against clio-core `dev` (bd1dc3183) (`context-runtime/util/clio_run*.cc`).
`clio_run` is a legacy alias of the same binary. The pip wheel installs both
as console scripts that set the library path and run
`<site-packages>/iowarp_core/bin/clio-run`.

Every command acts on the runtime of **this node**, found through the config
file and `CLIO_PORT`. Use the same `CLIO_SERVER_CONF`/`CLIO_PORT` as the
runtime, or the command talks to a different (or no) runtime.

| command | does |
|---|---|
| `clio-run start [flags]` | start the runtime in the foreground |
| `clio-run stop [--force] [--grace-period <ms>]` | stop this node's runtime |
| `clio-run status` | is the runtime up? |
| `clio-run compose <start\|stop\|rm\|list> ...` | create / destroy pools from a compose file |
| `clio-run config export --path <file>` | write the live effective config |
| `clio-run monitor [-i N] [-o] [-j] [-v]` | worker statistics |
| `clio-run migrate --pool-id <M.m> --container-id <C> --node-id <N>` | move a container |
| `clio-run viz <start\|stop\|status>` | control the dashboard |
| `clio-run refresh <repo>` | module developers: regenerate autogen files |
| `clio-run version` (`--version`, `-V`) | print the version |
| `clio-run --help`, `clio-run <cmd> --help` | usage |

Legacy forms still accepted: `clio-run runtime start|stop|status`,
`clio-run repo refresh`, `clio-run compose <file>` (= `compose start`),
`clio-run compose --unregister <file>` (= `compose rm`). `clio-run restart`
was removed: it prints a message and exits 1; use `start`.

## `start`

```
clio-run start [--fresh] [--ephemeral] [--induct] [--disk <path>]
               [--viz | --no-viz] [--viz-port <n>] [--viz-bind <addr>]
```

Runs in the **foreground** until `clio-run stop`, SIGTERM or SIGINT. There is
no daemon flag; background it yourself:

```bash
nohup clio-run start > ~/clio-run.log 2>&1 &
```

Startup: read config -> clean dead leftovers in the per-user directory -> bind
ports -> load modules -> create the admin pool -> compose every `compose:`
entry in order -> (recovering start) replay the address-table WAL and the pool
log -> serve the dashboard. A failed compose entry makes `start` exit 1.

| flag | effect |
|---|---|
| (none) | **Recover**: every compose pool takes its module's restart path (metadata logs, allocator logs, journals are replayed), and durable pools in this node's pool log are re-created. |
| `--fresh` | Discard this node's persistent state and start empty (below). |
| `--ephemeral` | Skip `compose:` and the pool log; only the admin pool exists. Compose explicitly afterwards. Sets `CLIO_EPHEMERAL=1`. |
| `--induct` | After start, broadcast `AddNode` so the running nodes register this one. |
| `--disk <path>` | Storage root for this run (creates the directory); sets `CLIO_STORAGE_ROOT`. Affects config paths written as `${CLIO_STORAGE_ROOT}/...`. |
| `--viz` / `--no-viz` | Serve / do not serve the dashboard. Default for `start` is on, unless `viz.enabled` or `CLIO_VIZ_ENABLE` says otherwise. |
| `--viz-port <n>` | Dashboard port (implies `--viz`). Default 8080; `0` = any free port. |
| `--viz-bind <addr>` | Dashboard bind address (implies `--viz`). Default `127.0.0.1`. |

### What `--fresh` deletes

Only this node's state, only the state the runtime and its modules track:

- the pool log `<conf_dir>/wal/pools.<node>.bin` (pools to re-create) and this
  node's address-table WAL `<conf_dir>/wal/domain_table.*.<node>.bin`;
- the legacy restart log is not replayed;
- each composed module starts fresh instead of restarting, so it discards its
  own logs: the CTE core deletes its metadata log files (snapshot and shards at
  `metadata_log_path`), a file bdev deletes its default
  `<pool_name>.alloc_log` (an explicitly configured `alloc_log` is kept and
  replayed), and replication, stream and clio-fs do not replay their logs.

It does **not** delete the config file, the backing data files of file bdevs
(their bytes stay on disk, unreferenced and later overwritten), the
`bdev_perf/` statistics or the learned cost models. Data in DRAM tiers is lost
on any stop, fresh or not.

An application that embeds the runtime (`CLIO_WITH_RUNTIME=1`, or
`clio_init(..., default_with_runtime=True)`) starts it with fresh semantics
unless it passes `is_restart=True`.

## `stop`

```
clio-run stop [--force] [--grace-period <ms>]
```

Sends a stop task to this node's runtime and waits for the process to exit.

| flag | default | effect |
|---|---|---|
| `--grace-period <ms>` | `5000` | Budget for draining in-flight tasks before teardown. `0` means 5000. |
| `--force` | off | Ungraceful immediate stop (no drain). |

`stop` waits up to grace + 25 s (10 s with `--force`). If the runtime does not
answer or does not exit, `stop` escalates (Linux/macOS): SIGTERM, wait 25 s,
then SIGKILL, then removes the dead runtime's leftover segments and sockets.
Windows has no escalation; it reports an error. Exit code 0 = stopped,
1 = still running. `stop` stops one node; run it on every node of a cluster.

## `status`

Prints one line and sets the exit code:

| exit | line | meaning |
|---|---|---|
| 0 | `clio runtime (port 9413): RUNNING (pid N)` | answered a heartbeat within 2 s |
| 1 | `... STOPPED (clean)` | no runtime, nothing left behind |
| 2 | `... STOPPED (stale artifacts)` + `stale: <path>` lines | no runtime, but leftover segments/sockets (a crash). The next `start` cleans them. |

An extra `UNRESPONSIVE - process N is alive but did not answer` line means a
runtime process exists but is wedged; `clio-run stop` will escalate.

## `compose`

```
clio-run compose start <file.yaml>     # create every pool in the file's compose: list
clio-run compose stop  <file.yaml>     # destroy them; keep them in the pool log
clio-run compose rm    <file.yaml>     # destroy them and drop them from the pool log
clio-run compose list [--restartable]  # pools in the local runtime / pool-log entries
```

- The file uses the same `compose:` format as the server config (CONFIG.md);
  other sections in it are ignored.
- `start` creates pools in order and stops at the first failure (exit 1).
  Get-or-create: an existing pool with the same name is reused unchanged.
- Entries with `restart: true` are recorded in each node's pool log and come
  back on the next recovering `clio-run start`.
- `pool_query: local` creates the pool on this node only; on a cluster use
  `broadcast` (from one node) or run the command on every node.
- `stop`/`rm` destroy by `pool_id` across the cluster; `rm` does not delete
  the file.
- `list` prints `Active containers (N):` then `<pool_name>  (pool_id=M.m)`.
  `--restartable` reads the pool logs in `<conf_dir>/wal` without contacting
  the runtime.

## `config export`

`clio-run config export --path <file>` writes this node's effective live
configuration (live pools, safe-bdev members) as YAML. Use it to capture a
runtime that was composed by hand.

## `monitor`

`clio-run monitor [-i <s>] [-o] [-j] [-v]`: worker statistics of this node
(queued, blocked and periodic tasks per worker). `-i` refresh interval in s
(default 1), `-o` once and exit, `-j` raw JSON, `-v` verbose.

## `migrate`

`clio-run migrate --pool-id 512.0 --container-id 1 --node-id 2` moves one
container of a pool to another node and updates the address table (recorded in
the address-table WAL).

## `viz`

```
clio-run viz start [--port <n>] [--bind <addr>]   # start if not running, print URL
clio-run viz stop                                 # stop the dashboard; runtime keeps running
clio-run viz status                               # print URL or "not running"
```

Exit: 0 = running (start/status) or stopped (stop); 1 = not running / failed;
2 = runtime not reachable. `--bind` accepts loopback or the address the
runtime's configuration names.

## `refresh`

`clio-run refresh <chimod_repo_path>` regenerates the method and dispatch
files of module source trees. It is for module developers with the source
and headers (not in the pip wheel); see the `chimod-dev` skill.
