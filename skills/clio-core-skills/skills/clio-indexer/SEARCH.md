# Searching: API reference and examples

All search goes to the CTE core's `SemanticSearch` method (id 35) on the
indexer pool (564.0) or any pool that forwards into it (the CAE core at 400.0
forwards search too). The indexer's own ids (100, 101) are not search.

## 1. Connect from Python

Both Python modules ship in the pip wheel.

```python
import os
os.environ.setdefault("CLIO_WITH_RUNTIME", "0")   # attach to `clio-run start`
os.environ.setdefault("CLIO_CTE_POOL", "564.0")   # bind to the indexer layer
import clio_cte_core_ext as cte
cte.clio_init(cte.RuntimeMode.kClient, False)
cte.initialize_cte("", cte.PoolQuery.Dynamic())
client = cte.get_cte_client()
```

Set both variables **before** the import. A same-host client that can map the
shm cache would bind to 564.0 anyway; a TCP or remote client stays on the core
(512.0), where search silently returns `[]`. Setting `CLIO_CTE_POOL` is always
safe.

## 2. `SemanticSearch` (Python, `clio_cte_core_ext`)

```python
results = client.SemanticSearch(
    tag_regex,            # str, full-match regex on tag names
    blob_regex,           # str, full-match regex on blob names
    query_text,           # str, tokenized like the index
    k=10,                 # int, max results; 0 = no cap
    pool_query=cte.PoolQuery.Broadcast(),  # Local() = this node only
)
# -> list[SemanticSearchResult], sorted by descending score
for r in results:
    r.tag_id.major_, r.tag_id.minor_   # TagId of the blob's tag
    r.blob_name                         # str
    r.score                             # float, BM25, higher is better
```

Async form: `fut = client.AsyncSemanticSearch(tag_regex, blob_regex,
query_text, k, pool_query)`, then `fut.result()` -> same list. Both release
the GIL while waiting.

The Python result has **no tag name** (the C++ struct does). Map ids to names
yourself:

```python
by_id = {}
for name in client.TagQuery(".*notes/.*", 0):          # list[str]
    tid = cte.Tag(name).GetTagId()
    by_id[(tid.major_, tid.minor_)] = name
```

Read a hit's bytes: `tag = cte.Tag(r.tag_id); size = tag.GetBlobSize(r.blob_name);
data = tag.GetBlob(r.blob_name, size, 0)` (returns `bytes`).

Errors: a bad regex or a pool that does not serve search returns `[]`; no
exception is raised. Check the runtime log.

## 3. Result semantics (BM25)

- **Working set:** documents in the node's index whose tag name matches
  `tag_regex` AND blob name matches `blob_regex` (`std::regex_match`, full
  string; ECMAScript syntax). `"doc"` does not match `doc7`; `"doc.*"` does.
- **Scoring:** Okapi BM25, k1 = 1.5, b = 0.75. Document frequency and average
  document length are computed **over the working set only**, so a score means
  "rank within this filter". Duplicate query words count once.
- **Every matched document is returned**, query words or not; non-matching
  ones score exactly 0 and sort last. An empty query returns matched documents
  with score 0. Filter `score > 0` for "real" hits.
- **Broadcast:** each node scores its own slice with its own statistics; the
  runtime merges by raw score, sorts, and trims to `k`. Scores from different
  nodes are therefore only roughly comparable. `Local()` searches only the
  node that receives the task.
- **Read-your-writes:** each node drains its dirty set before scoring.
- **Cost:** every query walks every document in each node's index to apply the
  regexes, then scores the matched ones. No blob bytes are read at query time.
- Scores are comparable only within one query.

## 4. Searching files written through the FUSE mount

Each file is a tag named by its path inside the mount. Its bytes are 1 MiB
page blobs named `0`, `1`, ... A small blob `~i` holds attributes. Search
pages only:

```python
hits = [h for h in client.SemanticSearch(".*notes/.*", "[0-9]+",
                                          "sea ice extent anomaly", 5)
        if h.score > 0]
for h in hits:
    print(f"{h.score:6.2f}  {by_id.get((h.tag_id.major_, h.tag_id.minor_))}"
          f"  (MiB {int(h.blob_name)})")
```

A large file can match on several pages; group by tag for one row per file.
Run `client.TagQuery(".*", 0)` once to see how paths are spelled on your
install.

## 5. The CEE route (`clio_cee`)

`clio_cee.ContextInterface` wraps search, listing, retrieval, import and
deletion. Use it when you want names or bytes, not scores.

```python
import os
os.environ.setdefault("CLIO_WITH_RUNTIME", "0")
os.environ.setdefault("CLIO_CTE_POOL", "564.0")
import clio_cee as cee
ctx = cee.ContextInterface()

# import two texts through the CAE (indexed in the default chain)
rc = ctx.context_bundle([
    cee.AssimilationCtx(src="string::ice", dst="iowarp::climate_docs",
                        format="string",
                        src_data="Arctic sea ice extent fell to a record low."),
    cee.AssimilationCtx(src="string::co2", dst="iowarp::climate_docs",
                        format="string",
                        src_data="Atmospheric CO2 reached 421 ppm."),
])
assert rc == 0
print(ctx.context_query("climate_docs", ".*", max_results=2,
                        prompt="sea ice minimum"))      # ['ice', 'co2']
packed = ctx.context_retrieve("climate_docs", ".*", max_results=1,
                              prompt="sea ice")         # [bytes-as-str]
ctx.context_destroy(["climate_docs"])
```

| method | signature | returns |
|---|---|---|
| `context_query` | `(tag_re, blob_re, max_results=0, prompt="", time_begin=0, time_end=0)` | `list[str]` blob names. Mode: temporal if either time bound != 0 (epoch ns), else BM25 if `prompt` non-empty, else name listing. With a prompt, `max_results=0` means **10**. |
| `context_retrieve` | `(tag_re, blob_re, max_results=1024, max_context_size=268435456, batch_size=32, prompt="", time_begin=0, time_end=0)` | A list with **one string**: the matched blobs' bytes concatenated in rank order, no separators, stopping before the first blob that would exceed `max_context_size` bytes. `[]` if nothing matched. Zero-size blobs are skipped. |
| `context_bundle` | `(list[AssimilationCtx])` | `0` or the CAE error code. See the `clio-cae` skill. |
| `context_destroy` | `(list[str] tag names)` | `0`, or `1` if any delete failed. Removes the tags and their index documents. |
| `context_splice` | `(new_ctx, tag_re, blob_re)` | Not implemented; returns 1. |

`context_query` returns blob names only, without tag names, and loses scores.
With a prompt it still returns zero-score blobs after the real matches. CEE
always searches with Broadcast.

## 6. C++

```cpp
#include <clio_cte/core/core_client.h>
clio::cte::core::Client idx(clio::run::PoolId(564, 0));
auto fut = idx.AsyncSemanticSearch(".*", "^doc[0-9]+$",
                                   "moon lunar telescope", /*k=*/5,
                                   clio::run::PoolQuery::Broadcast());
fut.Wait();
if (fut->GetReturnCode() != 0) { /* 1 = bad regex */ }
for (const auto &r : fut->results_) {
  // r.tag_id_, r.tag_name_, r.blob_name_, r.score_ (double)
}
```

C++ headers and CMake packages are not in the pip wheel; build against a
source, conda or Spack install (`find_package(clio-core)`, link
`clio::cte::core_client`, and `clio::cte::indexer_client` for ReindexScan).

## 7. Getting old data in: backfill and `ReindexScan`

Data enters the index three ways:

1. Writes through 564.0 (normal case).
2. **First-insertion backfill:** the first write to a tag after the indexer is
   composed (and, without `index_log_path`, after every restart) enqueues all
   existing blobs of that tag. From Python, writing any small blob into a tag
   indexes the whole tag.
3. **`ReindexScan`** (method 101), C++ only:

```cpp
#include <clio_cte/indexer/indexer_client.h>
clio::cte::indexer::Client ix(clio::cte::indexer::kIndexerPoolId,   // 564.0
                              clio::run::PoolId(512, 0));
auto scan = ix.AsyncReindexScan(".*notes/.*", ".*");  // Broadcast by default
scan.Wait();
uint64_t n = scan->blobs_enqueued_;   // summed over nodes
```

Each node enumerates its own (tag, blob) pairs matching both regexes,
intersected with the module's `tag_re`/`blob_re`, and **enqueues** them. The
work happens in the next sweep or search. Use it after a crash window, a
`--fresh` mistake, a scope change, a bulk load in passive mode, or data written
below 564.0. Narrowing `tag_re`/`blob_re` takes effect at the next restart
(restored documents outside the new scope are dropped); widening it needs a
`ReindexScan` for data written while it was out of scope.
