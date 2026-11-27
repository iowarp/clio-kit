# safe_bdev: the erasure-coded array (`clio_safe_bdev`)

Not in the pip wheel. Build from source, conda or Spack. In a wheel-only
install, composing `mod_name: clio_safe_bdev` fails because the module library
is absent.

## What it guarantees

- **Ack:** a `Write` returns success only after the data and the parity of
  every stripe it touched are written to the members, under a per-stripe lock.
  Like a plain bdev, that is not yet fdatasynced.
- **Sync:** `Sync` re-encodes every dirty stripe, syncs every active member,
  then fsyncs the log. After a successful `Sync` the data is durable and
  parity-protected. The CTE's fsync path calls it.
- **Failures:** up to `max_failures` members may be dead at once. Reads of a
  dead member's chunks are decoded from any N of the N + m columns, reading
  only the requested range. Writes continue degraded.
- **Beyond the budget:** a request that needs too many dead members fails
  (return code 1, `io_error_` = 2, device fault). It never returns wrong
  bytes. A restart with more members down than `max_failures` refuses to
  create the array (return code 4).
- **Crash consistency:** before any member write, the touched stripes are
  logged as intents. With `intent_sync: true` (default) the intent is
  fsynced first, so parity stays consistent across a power loss. After a
  crash, every stripe with a live intent is marked dirty and re-encoded
  before it is used for reconstruction. A degraded write saves the down
  member's chunk to `<alloc_log>.journal`, closing the write hole.
- **Without `alloc_log` there is no persistence at all:** membership,
  allocator, intents and the journal are all off.

| Event | Effect with `alloc_log` set |
|---|---|
| client crash | nothing lost |
| runtime crash (kill -9) | nothing acked lost; mid-write stripes re-encoded at start |
| power loss | unsynced data may be lost, parity stays consistent (`intent_sync: true`); with `false` consistency holds for process crashes only |
| 1..`max_failures` disks lost | no data lost; array runs degraded until rebuilt |
| more than `max_failures` disks | reads that need them fail with EIO; array refuses to start |
| node lost (`distributed: true`) | its members are marked faulty; same budget applies |
| `clio-run start --fresh` | the array's `alloc_log` is still replayed (it always recovers), while the CTE metadata above is wiped: space leaks. Delete `<alloc_log>`, `.members`, `.journal` and the member files for a true reset |

## Compose

Members first, then the array, then the CTE core that attaches to it.

```yaml
compose:
  - { mod_name: clio_bdev, pool_name: "/mnt/nvme0/disk0.bin", pool_query: local,
      pool_id: "302.0", bdev_type: file, capacity: "1TB", persistence_level: long_term }
  - { mod_name: clio_bdev, pool_name: "/mnt/nvme1/disk1.bin", pool_query: local,
      pool_id: "303.0", bdev_type: file, capacity: "1TB", persistence_level: long_term }
  - { mod_name: clio_bdev, pool_name: "/mnt/nvme2/disk2.bin", pool_query: local,
      pool_id: "304.0", bdev_type: file, capacity: "1TB", persistence_level: long_term }
  - mod_name: clio_safe_bdev
    pool_name: safe_array
    pool_query: local
    pool_id: "350.0"
    max_failures: 1
    alloc_log: "/var/lib/clio/safe_array.alog"
    intent_sync: true
    members:
      - { pool_name: "/mnt/nvme0/disk0.bin", node_id: 0, pool_id_major: 302, pool_id_minor: 0 }
      - { pool_name: "/mnt/nvme1/disk1.bin", node_id: 0, pool_id_major: 303, pool_id_minor: 0 }
      - { pool_name: "/mnt/nvme2/disk2.bin", node_id: 0, pool_id_major: 304, pool_id_minor: 0, parity: true }
  - mod_name: clio_cte_core
    pool_name: cte_main
    pool_query: local
    pool_id: "512.0"
    storage:
      - path: "safe_array"
        existing_pool_id: "350.0"
        existing_pool_module: clio_safe_bdev
        score: 0.2
        persistence_level: long_term    # required: unset = volatile
    performance:
      metadata_log_path: "${CLIO_STORAGE_ROOT}/cte_metadata_log"
```

Keys read by `safe_bdev::CreateParams::LoadConfig`:

| Key | Type | Default | Meaning and constraints |
|---|---|---|---|
| `max_failures` | integer | `1` | simultaneous member failures tolerated, and the most parity members accepted. Data members + `max_failures` must be at most 256 |
| `members` | list | none; at least one required | members are **referenced, not created**; each must be composed earlier |
| `members[].pool_name` | string | empty | the member bdev's pool name (path for file members) |
| `members[].node_id` | integer | `0` | node hosting the member; used only with `distributed: true` |
| `members[].pool_id_major` | integer | none | member pool id major. Without it the pool id is null |
| `members[].pool_id_minor` | integer | `0` | |
| `members[].parity` | bool | `false` | seat as a parity column. More parity members than `max_failures` is refused |
| `alloc_log` | path | empty (no persistence) | array WAL. Also creates `<alloc_log>.members` (membership manifest) and `<alloc_log>.journal` (degraded-write journal). Always replayed, even after `--fresh` |
| `distributed` | bool | `false` | `true`: each member is its pool's container on node `node_id`; member I/O is routed there, the array runs on the first member's node, and a member whose node dies is faulted. Management tasks sent to another node return rc 40 ("not home") |
| `intent_sync` | bool | `true` | `false`: intents reach the kernel but are not fsynced. Consistent across a process crash, not a power loss; about 10x faster small-file creates |

Constants (not configurable): chunk 64 KiB; a 64 KiB superblock at offset 0
of every member; background parity build every 50 ms; log flush/compaction
every 50 ms; up to 2^32 slots (256 TiB) per member; at most 256 members per
role.

The member bdevs' own allocators are not used: the array addresses member
offsets directly. Give file members `persistence_level: long_term` (or any
non-volatile label) anyway so their default logs are not removed.

## Layout, in brief

- Data member d, slot s holds a 64 KiB chunk at byte `65536 + s * 65536`.
  A stripe is slot s on every data member; parity is slot s on every parity
  member. Parity placement is dedicated, not rotated.
- The logical offset encodes the location (`chunk = off / 65536`,
  `member = chunk / 2^32`, `slot = chunk % 2^32`). There is no chunk map, so
  offsets stay valid when members are added, removed or rebuilt.
- Allocation is round-robin, one chunk at a time, over data members with room.
  Members may differ in size; each data member's usable slots are clamped to
  the smallest parity member's slot count. Capacity reported to the CTE is the
  sum of active data members' clamped slots.
- Small rewrites update parity from the delta; otherwise the stripe is
  re-encoded whole. Frees never touch parity.

## Failure injection (testing)

- Create `<member backing file>.fail` (for example `/mnt/nvme0/disk0.bin.fail`)
  to make that file member fail every read, write and sync with a device
  fault. The array faults the member on its next I/O, logs it once, persists
  the fault, and serves reads by reconstruction. Delete the marker to let the
  disk "come back"; the member stays faulty until rebuilt.
- `<member backing file>.nospace` makes writes fail with ENOSPC; that fails
  only the request, it does not fault the member.
- Transient errors and rejected ranges also fail only the request. Media
  errors and vanished devices fault the member.
- `CLIO_SAFE_BDEV_FAULT_SKIP_PARITY=1` in the runtime's environment: writes
  land data but not parity and the builder idles, simulating a crash between
  the two. Test only.

## Membership changes and rebuild

| Operation | What happens |
|---|---|
| `AddBdev` as data | reads the member's superblock (writes one if blank, refuses one owned by another array); joins the round-robin; no data moves. If an existing member reports a predicted lifetime under 7 days, the new disk replaces that member instead (migration) |
| `AddBdev` as parity | refused once parity count = `max_failures`; otherwise every written slot is marked dirty and the background builder computes the new column |
| `RemoveBdev` | marks a member faulty (`was_faulty` = 1) or removed (0). Nothing is copied; reads reconstruct around it. A member keeps its index; the array never shrinks |
| `RecoverBdev` | seats a replacement pool in the failed member's position and rebuilds every slot below the high-water mark (data from survivors + parity, parity by re-encoding). Writes during the rebuild are redone onto the replacement. An interrupted rebuild resumes at the next start. A replacement must hold at least the failed member's used slots |
| restart | replays the log and `.members`, reads every superblock, refuses foreign members, seats dead ones as down, fails if more than `max_failures` are down, resumes recoveries |

Operator tools (source builds):

```bash
# replace a dead member (creates a persistent file bdev, then RecoverBdev)
clio_safe_bdev_recover --array 350.0 --failed 302.0 \
    --member /mnt/nvme3/disk0b.bin --capacity 1099511627776 --new-id 305.0
# grow the array (AddBdev); --add parity raises tolerance instead
clio_safe_bdev_recover --array 350.0 --add data \
    --member /mnt/nvme4/disk4.bin --capacity 1099511627776 --new-id 306.0
```

Exit codes: 0 ok, 1 bad usage, 2 member create failed, 3 rebuild failed.
It connects as a client using the usual `CLIO_SERVER_CONF` environment.

A member added at runtime must exist again after a restart. The recover tool
creates it as a persistent pool. If you add members another way, also add
them to the compose file (or compose them with `restart: true`).

## Verify

```bash
curl -s "http://127.0.0.1:8080/api/pools/350.0/monitor?query=stats&routing=local"
```

The reply holds `max_failures`, `data_count`, `parity_level`,
`total_capacity`, `remaining_capacity`, `written_slots`, `dirty_slots`,
`faulty_members`, `recovery_active`, `recovery_ops_remaining`, and a
`members` list with each member's role, index, pool id, state and
`recovering` flag. The dashboard page `/viz/clio_safe_bdev` shows the same and
offers add / remove / replace.

Healthy steady state: `faulty_members` 0, `dirty_slots` 0 (shortly after the
last write), `parity_level` equal to the number of parity members.

Disk-loss drill: write data, `touch <member>.fail`, read the data back
(should match, `faulty_members` 1), then replace the member and wait for
`recovery_ops_remaining` 0. Disk-failure behaviour is validated on Linux only.

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| "safe_bdev Create: no member bdevs supplied" (rc 1) | `members` missing or not a list | add members |
| "... data members + max_failures N > 256" | too many columns | reduce |
| Create rc 4: "N members are down but max_failures is M" | too many dead members at start | bring the disks back (delete `.fail` markers, remount); data beyond the budget cannot be decoded |
| member refused as foreign | its superblock belongs to another array | use a blank file |
| Persistent CTE core refuses the tier, rc 16 | storage entry lacks `persistence_level` | add `long_term` |
| Membership lost after restart | no `alloc_log` | set it |
| Space leak after `--fresh` | array log always replays | delete the array's log files too |
| Twice the bytes written per fsync | `num_replicas` > 0 above the array | set it to 0 |
| Pool fails to load in a pip install | module not in the wheel | source / conda / Spack build |
