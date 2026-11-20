# INTERPOSER.md: writing a CTE data-path layer

An interposer is a chimod that speaks the **CTE core's own task interface**.
Clients point a `clio::cte::core::Client` at it, and it handles some core verbs
(usually Put and Get) while forwarding everything else to `next_pool_id`. Every
layer of the shipped chain is one: replication, the write-back and write-thru
caches, the shm cache, the indexer, the compressor, checkpoint and the CAE
summarizer. Use `context-transfer-engine/replication/` as the reference
implementation.

## The base class

`clio::cte::core::CoreInterposer`
(`context-transfer-engine/core/include/clio_cte/core/core_interposer.h`)
provides:
- `interposer_next_pool_`. Set it in Create from your config's `next_pool_id`.
  If it is null, the canonical core (512.0) is used.
- `ScheduleTask`, delegated to the core's own. Blob verbs hash to the blob's
  owner container, exactly as at the core. **Do not override it for core
  verbs.** If you do, chain operations resolve on the submitting node, and
  cross-node readers never see each other's blobs.
- `GetTaskStats`, the core's cost model. Override it only for your own verbs.
- `ForwardToCore(method, task)`, which runs the task on the next pool's
  **local** container verbatim.
- `ForwardSaveTask`, `ForwardLoadTask`, `ForwardLocal*Task`, `ForwardNewTask`,
  `ForwardNewCopyTask` and `ForwardAggregateOut`. Call these from the dispatch
  `default:` cases so that forwarded verbs keep the core's wire format.

## Rules

1. **Method ids of interposed core verbs MUST equal the core's**: `kPutBlob: 15`,
   `kGetBlob: 16`, `kDelBlob: 18`, `kDelTag: 19`, `kGetBlobSize: 23`,
   `kMultiPutBlob: 48`, `kSyncTag: 56`, and so on. Copy them from
   `context-transfer-engine/core/clio_mod.yaml`.
2. **Your own verbs are numbered ≥ 100.** A collision shadows a forwarded core
   verb, and the bug looks like the wrong handler crashing.
3. **The dispatch is hand-maintained.** Interposers do not use plain
   autogen for the forwarded verbs: unknown methods must fall through to the
   `Forward*` helpers. Copy the pattern from an existing interposer; the
   policy caches share `cache_base/.../cache_lib_exec.inc`.
4. **Compose order.** The pool you forward to must exist before you, and on
   every node: compose with `pool_query: local`, after it.
5. **Forwarding is in-process and same-node.** Routing already sent the task to
   the owner's node, so `ForwardToCore` is a local call. If you need another
   node, for example a replica, use a client and a `DirectId`/`DirectHash`
   query, and await it.
6. **Preserve the core's semantics for verbs you do not change.** Pass return
   codes through. A Put you transform must still report the logical size to
   the caller.
7. **State your layer's guarantees** in its `clio_mod.yaml` header comment and
   in `clio_default.yaml`:
   - what an ack means
   - what you keep in memory and how it is made durable (a WAL?)
   - what is lost on a crash
   - how you stay coherent across nodes

   Downstream users compose from those comments.
8. **Restart.** Anything you cache is either rebuilt from a log or dropped.
   Never trust in-memory state across a restart.
9. **Test multi-node.** Interposer bugs hide on one node, because every verb is
   local there. Run a two-runtime test (two ports on one host via
   `host:port` hostfile entries), with the writer on one node and the reader
   on the other. Then add a kill-and-restart test for anything that claims
   durability.
