# Evals - clio-runtime-config

Scenarios recorded 2026-11-21 against clio-core `dev` (bd1dc3183). Rubric is
pass/fail per bullet. Run each WITHOUT the skill to capture the gap, then WITH
it to confirm the gap closes.

## S1 - client cannot reach the runtime on another machine

Setup: Prompt: "I pip installed iowarp-core on two machines. The runtime runs
on host A with the default config. On host B I set CLIO_X=TCP and my Python
script hangs for 30 seconds and then says it cannot connect. What's wrong?"

Expected:

- Says `CLIO_X` is not read by any code; the transport variable is
  `CLIO_IPC_MODE` (and auto-selection already picks TCP when no local runtime
  exists).
- Tells the user to set `CLIO_SERVER_ADDR=<host A>` on host B.
- Says a single-node runtime binds loopback by default and must be started
  with `CLIO_BIND_ADDR=0.0.0.0` (no hostfile case).
- Mentions that the runtime connects back to the client, so host B must be
  reachable from A, and that ports 9413-9416 must be open; both sides must use
  the same port.

## S2 - what does --fresh delete

Setup: Prompt: "Before a benchmark I want a clean slate. Should I delete
~/.clio, or is clio-run start --fresh enough? What exactly does --fresh
remove?"

Expected:

- Says `--fresh` discards this node's pool log and address-table WAL
  (`<conf_dir>/wal`) and makes each module start fresh (CTE metadata log
  deleted, file bdev default `.alloc_log` deleted, replication/stream/clio-fs
  logs not replayed).
- Says it keeps the config file and the bytes of file-bdev data files (left
  unreferenced), and that an explicit `alloc_log` is still replayed.
- Warns not to delete `~/.clio` wholesale because `clio.yaml` lives there (and
  is re-seeded only from the packaged default, losing edits).
- Notes that a plain `clio-run start` recovers, and `restart` no longer exists.

## S3 - multi-node bring-up

Setup: Prompt: "Write me the steps and config to run the CLIO runtime on 4
nodes node01..node04 of our cluster. I wrote a hostfile with a comment header
'# cluster nodes' and the four names."

Expected:

- Sets `networking.hostfile` in the config (no `CLIO_HOSTFILE` variable).
- Flags that `#` lines are not comments in the hostfile and must be removed;
  offers `node[01-04]` range syntax.
- Requires the same config, same hostfile order, same port and same compose
  list on every node, with `pool_query: local`.
- Starts `clio-run start` on every node (pdsh/mpirun), mentions the 30 s
  `wait_for_restart`, and verifies with `clio-run status` and
  `clio-run compose list` on each node.
- Recommends leaving `swim.enabled: false` for batch jobs.

## S4 - compose change has no effect

Setup: Prompt: "I changed capacity in my compose file from 2g to 8g for pool
my_ram and ran `clio-run compose start extra.yaml` again. It says success but
the dashboard still shows 2g."

Expected:

- Explains get-or-create: an existing pool with the same `pool_name` is
  returned unchanged, without checking its keys.
- Gives the fix: `clio-run compose stop extra.yaml` (or `rm`) then `start`
  again, or restart the runtime.
- Mentions `restart: true` if the pool must come back after the next
  `clio-run start`, and that the pool log lives in `runtime.conf_dir`
  (default `/tmp/clio_$USER`).
