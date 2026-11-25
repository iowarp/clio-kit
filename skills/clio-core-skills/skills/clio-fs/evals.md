# Evals - clio-fs

Scenarios recorded 2026-11-21 against clio-core `dev` (bd1dc3183). Rubric is
pass/fail per bullet. Run each WITHOUT the skill to capture the gap, then WITH
it to confirm the gap closes.

## S1 - Mount on Linux from a pip install

Setup: Prompt: "I did pip install iowarp-core on Ubuntu. How do I mount CLIO as a directory, and how do I unmount it cleanly?"

Expected:

- Says to install the system FUSE 3 packages (`fuse3`, `libfuse3-3`) because the wheel does not bundle libfuse3.
- Starts the runtime first with `clio-run start`, then mounts with `CLIO_WITH_RUNTIME=0 clio_cte_fuse <mnt> -f`.
- Explains that without `CLIO_WITH_RUNTIME=0` the daemon may host a private runtime other clients cannot see.
- Unmounts with `fusermount3 -u <mnt>` (not Ctrl+C) before `clio-run stop`.
- Does not claim the daemon creates the filesystem pool; notes the compose must contain `clio_cte_filesystem` (the default config does).

## S2 - What fsync guarantees across nodes

Setup: Prompt: "On a 4-node CLIO mount with the default config, my app got EIO from fsync after another node rebooted. Is that a bug? What was lost?"

Expected:

- Explains fsync returns EIO when a node that may have held the file's unsynced writes died or was down since the last fsync, and that this is intended.
- States that with the default `replication_factor: 1` bytes written since the last fsync that lived on that node may be lost; earlier fsynced data is safe.
- Says a retry of fsync without new writes succeeds, and the app should rewrite the data since its last fsync.
- Recommends `replication_factor` >= 2 with `failover_to_successor: true` to avoid it.
- Mentions `fsync_mode` is a CTE core `performance` key (`durable` default, `deferred` returns at once).

## S3 - Sizes and namespace survive restart

Setup: Prompt: "Review my compose: stream at 565.0 with no log_path, filesystem at 560.0 with metadata_log_path set. Will file sizes and directories survive clio-run stop / start? What about if I decommission node 3 for good?"

Expected:

- Says to set `log_path` on the stream entry; the filesystem's `<metadata_log_path>.stream` is only used when the filesystem creates its own stream pool.
- Explains that without it, restart falls back to inode-record sizes (last close/fsync/truncate) and merged deferred appends can be cut off.
- States directory blocks journal to `<metadata_log_path>.dirs.<container>` on the home node (fsynced every 1000 ms and on fsync/fsyncdir).
- States that a permanently lost node with one copy takes its directory blocks, inode records and pages with it (#1263); replication >= 2 plus failover lets a stand-in serve the CTE mirror.
- Warns that `clio-run start --fresh` discards all of it.

## S4 - Mount hangs during a large copy

Setup: Prompt: "Copying 16 GB into ~/clio-mnt on a 4 GB VM: writes started taking about two minutes each and ls hangs. No errors. What happened and how do I recover?"

Expected:

- Identifies that the runtime likely died (OOM) and that the FUSE daemon blocks instead of returning EIO.
- Names the indexer's in-memory term index (564.0 in the default chain) as the likely memory growth.
- Recovery: lazy unmount (`fusermount3 -uz`), kill `clio_cte_fuse`, `clio-run start` without `--fresh`, remount.
- Mitigation: point the stream and filesystem `next_pool_id` at 568.0 (below the indexer) or narrow the indexer scope, and cap memory.
- Notes that data written but not fsynced before the crash may be lost or read as EIO.
