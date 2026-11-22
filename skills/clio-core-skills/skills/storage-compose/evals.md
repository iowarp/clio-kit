# Evals - storage-compose

Scenarios recorded 2026-11-20 against clio-core `dev` (86efd1a7e). Rubric is
pass/fail per bullet. Run each WITHOUT the skill to capture the gap, then WITH
it to confirm the gap closes.

## S1 - "make it durable"

Setup: Prompt: "I added a file tier to the CTE core in my compose section; my
data should survive a restart now, right?" Config: one core, a `file` tier with
no `persistence_level`, no `metadata_log_path`.

Expected:

- Flags that the tier is volatile without `persistence_level` and that its
  blocks are dropped at replay.
- Flags that without `metadata_log_path` the bytes cannot be found after restart.
- Recommends the persistent + volatile core split with a write-back cache
  rather than mixing tiers in one core.
- Proposes a kill -9 / restart (without `--fresh`) check.

## S2 - state the guarantees

Setup: Prompt: "What exactly does an acknowledged put mean in the default
config, and what do I lose if a node loses power?"

Expected:

- Says the ack means the bytes are in the owner's volatile core (write-back).
- Names the loss window: unflushed dirty data (up to `flush_period_ms`) plus
  un-fsynced device writes and WAL (up to `flush_metadata_period_ms`).
- Names fsync / SyncTag as the durability point and write-thru as the
  alternative.

## S3 - survive a node

Setup: Prompt: "Four nodes; I need reads and writes to keep working if one node
dies."

Expected:

- Sets `replication_factor: 2` (or `remote_copies >= 1`) AND the core's
  `failover_to_successor: true`, and explains that one without the other is
  not enough.
- Sets `handoff_log_path`.
- Notes that write-back dirty data is single-node until flushed, and offers
  write-thru if each ack must be replicated.
- Does not enable SWIM with short timeouts for a batch job.

## S4 - compose order and pool ids

Setup: A config listing the indexer before the shm cache it points at, and both
`clio_cte_checkpoint` and `clio_cte_core_persist_write_thru` on default ids.

Expected:

- Reorders so each pool follows every pool it names.
- Detects the 566.0 collision and explains that the second pool is silently
  never created.

## S5 - disk failure on one node

Setup: Prompt: "Single big node with three NVMe drives; survive losing a
drive."

Expected:

- Uses `clio_safe_bdev` with a parity member, attached via `existing_pool_id`,
  with an explicit `persistence_level`.
- Keeps `num_replicas: 0`.
- Validates with a `<pool_name>.fail` file.

## Baseline failure modes to watch for (RED)

- Calling a deployment durable without naming the failure and the window.
- Putting CAE at 512.0, or pointing a core client at a non-core pool.
- Recommending `flush_data_period_ms` on a dev-era config (ignored there).
