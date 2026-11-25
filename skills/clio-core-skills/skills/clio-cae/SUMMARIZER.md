# The summarizer (`clio_cae_summarizer`, 401.0)

An interposer on the CTE core's task interface. It forwards every core call
to `next_pool_id`. For each **PutBlob** (method 15) whose tag and blob name
match a rule, it sends the put's payload to an LLM and stores the reply as
`<blob_name>_label` in the same tag. With no rules it is a pure passthrough.
Not in the pip wheel: build from source with the CAE enabled and libcurl +
nlohmann_json installed (otherwise it builds, but every summary is empty and
the log says `produced no output`).

## Compose entry

Place it after the chain top it forwards to and before the CAE entry, then
point the CAE at it:

```yaml
  - mod_name: clio_cae_summarizer
    pool_name: clio_cae_summarizer
    pool_query: local
    pool_id: "401.0"
    next_pool_id: "564.0"                     # chain top: labels get indexed
    label_endpoint: "http://127.0.0.1:11434"  # Ollama
    label_api: ollama
    label_prompts:
      summarize: "Summarize the following text in one short sentence."
    label_matches:
      - tag_re: "\\.txt$"                     # tag names ending in .txt
        blob_re: "^chunk_[0-9]+$"             # CAE file chunks, not description
        model: "gemma3:1b"
        prompt: "summarize"
        context_length: 4096
        num_predict: 64

  - mod_name: clio_cae_core
    pool_name: cae_main
    pool_query: local
    pool_id: "400.0"
    next_pool_id: "401.0"                     # CAE -> summarizer -> indexer
```

Hosted OpenAI-compatible endpoint (example: ALCF Inference Endpoints):

```yaml
    label_endpoint: "https://inference-api.alcf.anl.gov/resource_server/metis/api/v1"
    label_api: openai
    label_api_key_file: "${HOME}/.alcf_access_token"
    label_api_key_env: ALCF_ACCESS_TOKEN
    label_matches:
      - tag_re: "\\.txt$"
        blob_re: ".*"
        model: "Mistral-Large-3-675B-Instruct-2512"
        prompt: "summarize"
        context_length: 32768
        num_predict: 256
```

## Keys (`SummarizerConfig::LoadConfig`)

| key | type | default | meaning and constraints |
|---|---|---|---|
| `next_pool_id` | `"major.minor"` | unset: `512.0` | Pool every call (and the `_label` write) goes to. Use the chain top (564.0) so labels are searchable. |
| `label_endpoint` | URL | `""` | Base URL. Ollama: POST `{endpoint}/api/generate`. OpenAI: POST `{endpoint}/chat/completions`. |
| `label_api` | `ollama` \| `openai` (case-insensitive) | `ollama` | Any other value **fails pool Create**. |
| `label_api_key_file` | path, `${VAR}`/`~` expanded | `""` | openai only. File holding the bearer token. Wins over the env var. Re-read on every request. |
| `label_api_key_env` | env var name | `""` | openai only. Variable holding the token, read in the **runtime's** environment on every request. Neither set: no `Authorization` header (warning at Create). |
| `label_prompts` | map name -> text | empty | Prompt templates. The request text is `"{prompt}\n\n{chunk}"`. |
| `label_matches` | list | empty (passthrough) | Rules, first match wins. |
| `label_matches[].tag_re` | ECMAScript regex, **substring** (`regex_search`) | `""` (matches all) | Against the tag's full name. Unlike the indexer, not anchored: use `^...$` to anchor. An invalid regex disables only that rule. |
| `label_matches[].blob_re` | regex, substring | `""` | Against the blob name. |
| `label_matches[].model` | string | `""` | Model id sent to the server. |
| `label_matches[].prompt` | string | `""` | Key into `label_prompts`. Unknown key: rule skipped with a warning. |
| `label_matches[].context_length` | int, tokens | `4096` | Ollama `num_ctx`, and the chunking budget for both APIs. `<= 0` disables chunking. |
| `label_matches[].num_predict` | int, tokens | `0` (no cap) | Ollama `num_predict` / OpenAI `max_tokens`, per chunk. |

Parsing is best-effort: a malformed value aborts the rest of the parse
silently. Check the Create log line: `label_matches=N, label_endpoint='...',
label_api='...'`.

## Behavior and cost

- **Synchronous.** The handler forwards the put, then (if a rule matches)
  calls the model on the same worker, stores `_label`, and only then
  completes the put. A matching put's latency includes every inference call.
  Each request has a 5 s connect timeout and a 120 s total timeout.
- **Chunking.** Byte budget per request = `context_length * 3/4 * 3` minus
  the prompt length (about 9 KB at 4096 tokens). A larger payload is split;
  each chunk is one request; replies are joined with blank lines. Failed
  chunks are skipped; all failed = no label.
- **Payload only.** It summarizes the bytes in that one PutBlob, not the
  whole blob: a partial write labels only the written fragment, and each
  1 MiB CAE chunk gets its own label. MultiPutBlob and other verbs are not
  summarized. Only traffic addressed to 401.0 is seen: the FUSE mount and
  clients on 564.0 bypass it.
- **No loops.** `_label` is written to `next_pool_id`, so it is never
  summarized again. A rule with `blob_re: ".*"` still matches `x_label` if
  something else writes it through 401.
- **Failures never fail the put.** Inference or store errors are logged as
  warnings.
- **Proxy.** Loopback endpoints (`localhost`, `127.0.0.1`, `::1`) are never
  proxied; others follow `https_proxy`/`no_proxy` of the runtime.

Throughput rule of thumb: with N workers, at most N matching puts are in
inference at once, and other tasks queued on those workers wait behind them.
Scope rules to small, text-only, low-rate data.

## Reading labels

```python
import os
os.environ.setdefault("CLIO_WITH_RUNTIME", "0")
os.environ.setdefault("CLIO_CTE_POOL", "564.0")
import clio_cee as cee
ctx = cee.ContextInterface()
print(ctx.context_query("notes\\.txt", ".*_label"))     # ['chunk_0_label', ...]
print(ctx.context_retrieve("notes\\.txt", "chunk_0_label"))
print(ctx.context_query(".*", ".*_label", 5, "sea ice"))  # search the summaries
```

## Verify

1. `curl http://127.0.0.1:11434/api/tags` lists the model (Ollama).
2. Start the runtime; the log shows the summarizer Create line with
   `label_matches=1`.
3. Import a small `.txt` file through the CAE (SKILL.md, "Verify it works")
   with a `dst` tag that matches `tag_re`, then list `.*_label` blobs as
   above. No label, a `Summarizer: chunk 1 of 1 failed` warning, and a
   `transport error` or `HTTP <code>` warning means the endpoint, model or
   token is wrong. The chunk warning with **no** transport/HTTP warning means
   a build without libcurl/nlohmann_json (requests are never sent).
