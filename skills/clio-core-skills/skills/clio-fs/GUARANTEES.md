# clio-fs guarantees

Stated for the FUSE mount over the shipped default chain: indexer 564.0 ->
shm cache 568.0 (metadata only) -> write-back cache 567.0 -> volatile RAM
core 540.0, flushed through replication 561.0 (`replication_factor: 1`) to the
persistent core 512.0, with `metadata_log_path` set on the core and the
filesystem. Other chains change the answers; see the storage-compose skill.

## What each call means

| call | returns when | survives |
|---|---|---|
| `write()` | each 1 MiB page range is accepted by the chain: the write-back cache holds it in RAM on the page's owner node. The file's new size is held by the daemon until close/fsync | application crash. **Not** a runtime crash or power loss until flushed |
| `write()` with `O_APPEND` on multi-node (`deferred`) | the bytes are staged as a blob in `staging_pool_id` (561.0, persistent path); the stream home merges them within about 1 ms | runtime crash (re-queued from the staged blob on restart) |
| `close()` | size published to the file's stream home and inode record (close-to-open) | every node's next `open` sees bytes and size |
| `fsync()` / `fdatasync()` | in order: drain pending publish, publish the size, wait for pending replication, sync the file's tag (dirty copies flushed to a persistent tier, devices fsynced, CTE metadata WAL fsynced), fsync the stream size log, sync the parent directory | runtime crash and power loss, for that file and its name |
| `fsyncdir()` | the directory's blocks and journal are durable | power loss of creates/renames/unlinks in it |
| namespace ops (`mkdir`, `rename`, `unlink`, `create`) | the home applied the change, wrote it (journal record + CTE mirror blob), and pushed it to every node caching the directory | runtime crash (journal writes reach the OS per record); power loss after the next 1000 ms journal sync or `fsync` |

Errors:

- `write()` returns **ENOSPC** when no tier can take a page, EIO otherwise.
  Bytes of earlier pages in the same call are kept and counted.
- `close()` reports an error latched by an asynchronous publish; release
  errors never reach `close(2)` (FUSE limitation).
- `fsync()` returns **EIO** if a node that may have held this file's unsynced
  writes was declared dead, or is down, since the window opened (the first
  write after the previous fsync). Reported once, like Linux errseq. Writes
  made after the peer went silent went to live successors and fsync cleanly.
- `fsync()` returns ENOSPC when no persistent tier had room.
- Reads of a page whose owner just died retry for about 10 s, then EIO. A
  missing page reads as zeros (a hole); any other read failure is EIO, never
  zeros.

`fsync_mode` (CTE core entry, `performance.fsync_mode`):

| value | `fsync()` behaviour |
|---|---|
| `"durable"` (default) | as above |
| `"deferred"` | returns at once; the write-back cache's periodic flush (`flush_period_ms`, default 10000 ms) makes data durable. A crash can lose up to that window. Any other value refuses the config |

The POSIX/STDIO interceptors write synchronously and their `fsync` makes the
file durable the same way. `MPI_File_sync` does nothing.

## What each failure loses (defaults)

| failure | lost or unavailable |
|---|---|
| application crash | nothing it already `write()`-returned. Its open handles are released; unlinked-but-open files are destroyed |
| `clio_cte_fuse` crash | bytes stay stored, but sizes of files written and not yet closed or fsynced stay at the last published size (the daemon held the larger size). Mount shows "Transport endpoint is not connected" |
| runtime `kill -9` | dirty pages in the RAM write-back copy not yet flushed (flushed every 10000 ms, on fsync, at 256 MiB new dirty data, or at 80% RAM fill). Fsynced files survive whole. Namespace changes survive (journal records reached the OS). Open handles do not survive a restart. The mount hangs until remounted |
| power loss | as kill -9, plus up to 1000 ms of directory journal, up to 5000 ms of stream size log and CTE metadata WAL. Fsynced files survive |
| disk loss (persistent tier) | everything on it, including inode records and directory mirrors it held, unless `num_replicas` or remote copies exist |
| temporary node loss | that node's pages, inode records and directory blocks are unavailable (EIO after network timeout, or at once with `CLIO_NET_DEAD_FAIL_FAST=1`); files homed elsewhere keep working. Mutations of state the dead node cached can stall about 12 s (10 s lease + 2 s skew) |
| permanent node loss | with one copy: its pages, inode records, directory journal and directory blocks it homed (#1263). With `replication_factor` >= 2 and `failover_to_successor: true`, a stand-in serves the replicated CTE mirror of each directory block and the inode records |
| `clio-run start --fresh` | this node's whole state: CTE metadata, directory journal, stream log. Never use it to "fix" a hang |

Restart (`clio-run start`, no flag) replays the CTE snapshot and WAL, the
directory journal, and the stream log; it re-queues staged appends, finishes
interrupted merges, destroys orphaned inodes, and holds size requests until
each file's size is reconciled with its inode record (at most 120 s). RAM-tier
bytes that were never flushed are reported as lost: reads of them return EIO.

**Sizes across a restart.** Set `log_path` on the stream entry. The default
config composes the stream at 565.0 with `log_path` commented out, and a
composed stream pool ignores the filesystem's `<metadata_log_path>.stream`. A
restart then falls back to each file's inode-record size (stored at every
close, fsync and truncate); deferred appends merged after that record was
written can be cut off.

## Consistency across nodes

- **Namespace:** every directory block and inode has one home that applies
  changes in order and pushes them to every node caching it before
  acknowledging. When a namespace call returns, every node sees it. Cached
  copies are trusted for a 10 s lease; if the home is unreachable after
  that, the operation fails instead of serving stale data.
- **Data:** close-to-open. A reader that opens after the writer's `close`
  (or `fsync`) sees its bytes and size on any node, given the multi-node
  default `CLIO_FUSE_ATTR_CACHE_S=0`. Concurrent overlapping writes from
  different nodes are not ordered; applications must synchronize.
- **Appends:** each `O_APPEND` write lands whole, never interleaved. One
  node's appends keep their order; appends from different nodes are ordered
  by their origin's timestamp (see STREAM_CHECKPOINT.md).
- **Rename** is a multi-step protocol, not one atomic step. A crash in the
  window can leave the file under both names (two hard links), never a
  dangling name. `RENAME_NOREPLACE` is probe-then-rename (small race);
  `RENAME_EXCHANGE` returns EINVAL.
- **`O_EXCL`**: of N racing creators anywhere, exactly one wins.
- **Search names** (CTE tag names used by `TagQuery`/indexer) are published
  asynchronously in batches: eventually consistent. Directory blocks are the
  authority.

## Crash test (verify the guarantees)

```bash
clio-run start &
CLIO_WITH_RUNTIME=0 clio_cte_fuse ~/clio-mnt -f &
dd if=/dev/urandom of=/tmp/ref bs=1M count=32
cp /tmp/ref ~/clio-mnt/a && sync ~/clio-mnt/a        # coreutils sync FILE = fsync
cp /tmp/ref ~/clio-mnt/b                             # no fsync
pkill -9 -x clio_run                                 # runtime dies; mount now hangs
fusermount3 -uz ~/clio-mnt; pkill -9 -x clio_cte_fuse
clio-run start &                                     # NOT --fresh
CLIO_WITH_RUNTIME=0 clio_cte_fuse ~/clio-mnt -f &
cmp /tmp/ref ~/clio-mnt/a && echo "a intact"         # must pass
cmp /tmp/ref ~/clio-mnt/b || echo "b lost unflushed data (allowed)"
```

`a` must be intact. `b` may be short, may fail reads with EIO (lost RAM
blocks), or may be intact if a flush ran before the kill. Multi-node
variant: keep a file open on node 2's mount, write 64 MiB (pages spread over
all nodes) without fsync, `kill -9` node 1's runtime, then `fsync` the same
descriptor on node 2: EIO is the correct answer. A second `fsync` with no new
writes returns 0.
