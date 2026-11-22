# Evals - clio-cte-core

Scenarios recorded 2026-11-21 against clio-core `dev` (bd1dc3183). Rubric is
pass/fail per bullet. Run each WITHOUT the skill to capture the gap, then WITH
it to confirm the gap closes.

## S1 - every put returns rc=11

Setup: Prompt: "I pip-installed iowarp-core and edited ~/.clio/clio.yaml so the
CTE core at 512.0 has one NVMe tier. Now every PutBlob from Python returns 11.
What is wrong?" Config: `durability: persistent`, one `file` tier with no
`persistence_level`.

Expected:

- Explains rc 11 as "no target with space", which is also the code when no
  target is registered.
- Identifies the missing `persistence_level`: the tier is volatile, so the
  persistent core refuses it (rc 16 in the log, `refusing volatile bdev`).
- Gives the fix (`persistence_level: temporary` or `long_term`) and says to
  set `performance.metadata_log_path` for restart durability.
- Offers a check: `AsyncListTargets().result()` / `AsyncGetCapacity()` on
  `Client(PoolId(512, 0))`, or the runtime log.

## S2 - write the two core entries

Setup: Prompt: "Write the compose entries for a persistent CTE core on
/mnt/nvme (500 GB) and a 64 GB DRAM volatile core, with metadata that survives
a reboot."

Expected:

- Two separate `clio_cte_core` entries (512.0 `durability: persistent`, 540.0
  `durability: volatile`), not one core mixing DRAM and disk.
- The disk tier has `bdev_type: file`, `capacity_limit`, `score`, and
  `persistence_level: temporary` or `long_term`.
- `performance.metadata_log_path` is set on the persistent core only.
- Notes that a write-back cache (`clio-cte-caches`) is needed to move data
  between the cores, and that cores are composed before modules naming them.

## S3 - talk to the top of the chain from Python

Setup: Prompt: "My Python script uses clio_cte_core_ext Tag and Client. How do I
make sure it writes through the indexer at 564.0 and not straight into the
core at 512.0? Also, how do I fsync a tag?"

Expected:

- `CLIO_CTE_POOL=564.0` before `initialize_cte()` (aims `Tag` and
  `get_cte_client()`), or `cte.Client(cte.PoolId(564, 0))`.
- Warns that a TCP/IPC client without `CLIO_CTE_POOL` is bound to 512.0 and
  bypasses the caches.
- States that `SyncTag` has no Python binding on dev; offers C++
  `AsyncSyncTag` (rc 5/28 on failure) or `os.fsync` through a clio-fs mount.

## S4 - what does an ack mean on a bare core

Setup: Prompt: "I use only the persistent CTE core, no caches. What do I lose
on kill -9, on power loss, and with clio-run start --fresh?"

Expected:

- Ack = bytes written to the owner's device (not fdatasynced) and the WAL
  record written to the OS after the data.
- kill -9 loses no acked data (WAL records are flushed per record); score
  changes up to ~1 min can be lost.
- Power loss can lose unsynced device writes and up to
  `flush_metadata_period_ms` (5 s) of WAL; `SyncTag` closes the window.
- `--fresh` deletes the WAL shards and snapshot: everything is gone.
