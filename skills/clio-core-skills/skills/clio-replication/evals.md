# Evals - clio-replication

Scenarios recorded 2026-11-21 against clio-core `dev` (bd1dc3183). Rubric is
pass/fail per bullet. Run each WITHOUT the skill to capture the gap, then WITH
it to confirm the gap closes.

## S1 - copies but no availability

Setup: Prompt: "I set replication_factor: 2 on a 4-node cluster and killed one
node. Reads of its files hang for 30 seconds and then fail. Isn't replication
supposed to fix this?" Config: replication at 561.0, core at 512.0 without
`failover_to_successor`.

Expected:

- Identifies the missing `targets.failover_to_successor: true` on the core
  named by replication's `next_pool_id` (512.0), on every node.
- Explains that replication makes the copies and failover routes to them; one
  without the other gives no availability.
- Explains the 30 s wait as `CLIO_PROBE_SILENCE_S` / `CLIO_NET_RETRY_TIMEOUT_S`
  and offers `CLIO_NET_DEAD_FAIL_FAST=1` plus lower values.

## S2 - what does an ack mean in the default chain

Setup: Prompt: "With replication_factor 2 in the default clio.yaml, if my put
returns OK, is the data on two nodes?"

Expected:

- Says no: the default chain puts write-back above replication, so the ack
  means the owner's volatile core only.
- Names the points where it reaches two nodes: write-back flush
  (`flush_period_ms`, 10 s), fsync/`SyncTag`, graceful `clio-run stop`.
- Offers write-thru or addressing 561.0 directly for replicated acks.
- Notes a dead successor is skipped: the put succeeds with one copy fewer and is
  re-copied when it returns.

## S3 - permanent node loss

Setup: Prompt: "One of our 8 nodes died for good. We run replication_factor 2
with failover. Is all data still safe, and do we need to do anything?"

Expected:

- Data the dead node owned survives on its successor and is still served.
- States there is no automatic re-replication: those blobs now have one copy.
- Warns writes taken during failover were single-copy.
- Does not claim a rebuild command exists.

## S4 - multi-node test plan

Setup: Prompt: "Give me a test that proves our 3-node replicated CTE survives a
node crash and gets the writes back afterwards."

Expected:

- Uses `kill -9`, not `clio-run stop`, and restarts with `clio-run start`
  without `--fresh`.
- Writes during the outage (overwrite and delete), then checks them after the
  owner returns (hand-back).
- Sets `handoff_log_path` and includes a stand-in restart or explains why it
  matters.
- Fsyncs first or binds the client to 561.0, so the write-back cache does not
  hide the result.
