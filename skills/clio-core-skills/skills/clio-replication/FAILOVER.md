# FAILOVER.md: node outage, hand-back, and the multi-node test

Checked against clio-core `dev` (bd1dc3183), 2026-11-21.

## Placement of copies

- **Owner.** `owner = hash(tag_id, blob_name) mod N`, where N is the number of
  containers in the pool (one per hostfile entry, fixed even while nodes are
  down). Container id equals node id.
- **Remote copies.** Containers `owner+1 ... owner+remote_copies`, mod N. They
  are stored in each successor's core as a *shadow copy* (a blob that node does
  not own) at `replica_score`. The successor's own `num_replicas` apply to it.
- **On-node replicas.** Replica indexes `1 ... num_replicas` in the owner's own
  core, flagged fixed and persistent (never migrated, never on a volatile tier).
- **Stand-in.** While the owner's node is dead, the first live container after
  it, `owner+1, owner+2, ...`. That is the node holding the first remote copy
  when it is alive.

## Timeline of an outage (`replication_factor: 2`, failover on)

1. **Node A stops answering.** Operations on A's blobs wait.
   `CLIO_PROBE_SILENCE_S` (30 s) after A went quiet, peers declare it dead. A
   task already sent to A fails after `CLIO_NET_RETRY_TIMEOUT_S` (30 s), or at
   once with `CLIO_NET_DEAD_FAIL_FAST=1`. A batched put whose owner died in
   flight is resent up to 10 times, 1 s apart.
2. **Failover.** The core under replication now routes A's blobs to the
   stand-in B (the next live node). B serves reads from its shadow copy.
3. **Writes during the outage.** B writes its shadow copy only (no further
   remote copy), records a hand-back entry (blob name, delete or write, wall
   time), appends it to `handoff_log_path.<B>` if set, and invalidates every
   live node's cached copy of the blob (B does not know who caches it).
4. **Re-copy list.** Owners that tried to mirror to a dead node note those
   blobs in memory and re-copy them from the primary when it is back
   (sweep every 500 ms). This list is lost if that owner restarts meanwhile.
5. **A restarts with `clio-run start`** (not `--fresh`). Its replication
   container:
   - sends `HandoffPull` to every live peer and waits at most 5 s, then lets
     compose continue;
   - holds client writes to blobs it owns until all pulls answered, or 120 s
     passed (log: `never answered the hand-back pull`);
   - on the first size probe or read of each owned blob, compares its primary
     with the remote copy and heals the primary if the copy holds more (a
     `kill -9` can leave the WAL behind the copy, which was written before the
     ack).
6. **Hand-back.** Peers push each recorded blob back to A (4 MiB chunks). If A
   came back without a restart (a healed partition), B's 500 ms sweep pushes
   instead. A push older than a write A took itself since returning is dropped.
7. **Steady state.** B's hand-back entries are cleared; the log is rewritten as
   a snapshot once it passes 4 MiB.

If B restarts before A returns: with `handoff_log_path`, B replays its log and
still hands back. Without it, the entries are gone and A returns with its
pre-outage data for those blobs.

## What is not handled

- No automatic re-replication after a permanent loss. The stand-in keeps
  serving, at reduced redundancy, for as long as the hostfile names the dead
  node.
- No rack or failure-domain placement.
- Before a node is declared dead, its blobs are unavailable (not served stale).
- Container migration on failure only happens with SWIM enabled (off by
  default; its defaults take about an hour).

## Multi-node verification procedure

Needs 3 nodes (A, B, C) with one hostfile, the same config on each,
`replication_factor: 2`, `failover_to_successor: true` on the core under
replication, and `handoff_log_path` set. For a quick run set
`CLIO_PROBE_SILENCE_S=5`, `CLIO_NET_RETRY_TIMEOUT_S=5` and
`CLIO_NET_DEAD_FAIL_FAST=1` in every runtime's environment.

Use a client bound to 561.0 (`CLIO_CTE_POOL=561.0`) so the test measures
replication itself, not the write-back cache above it. To test the full chain
instead, fsync (`SyncTag`, or `fsync` on clio-fs) after step 1.

1. **Write.** From C, write 200 blobs of 1 MiB with known content under one tag
   and record their checksums. About a third of them are owned by each node.
2. **Kill an owner.** `kill -9` the runtime on A (do not use `clio-run stop`).
3. **Wait** past `CLIO_PROBE_SILENCE_S`.
4. **Read all 200 from C.** Every checksum must match. The blobs A owned are now
   served by B. A failure here means failover is off, or the copies were never
   made (check B's log for `remote copy ... failed` from step 1).
5. **Write during the outage.** Overwrite 20 blobs, including some A owns, with
   new content. Delete 5 more of A's blobs.
6. **Optional:** restart B now (`clio-run stop` then `clio-run start`) to prove
   the hand-back log survives a stand-in restart.
7. **Restart A** with `clio-run start` (no `--fresh`). Look in A's log for
   `replayed ... handoff records` (on B, if step 6) and `handed N blob(s) back
   to container` on the peers.
8. **Read everything from A and from C.** Overwritten blobs return the new
   content, deleted blobs are gone, the rest are unchanged.
9. **Kill the node whose successor was A** (C, if the hostfile order is A, B,
   C), wait, and read its blobs from B: they are served from A, including the
   ones overwritten in step 5. This shows the copies A missed while down were
   re-made after it returned.

Record the time from the kill to the first successful read: it is the
unavailability window you can promise.
