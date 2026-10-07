# Web MCP Server

Install the launcher using the [CLIO Kit setup guide](../../setup.md) before
using the commands below. See [agent integrations](../../README.md#agent-integrations)
for MCP and skill configuration.

The Web MCP exposes synchronous `search`, durable task-enabled `fetch`, and
`fetch_events` for the complete backend conversion log. The selected search
provider is fixed when the MCP starts; no tool call can silently switch it.

## Install

Connect a local stdio MCP to one unified CLIO Web Search deployment (replace
`http://localhost:8089` with the address of your deployment):

```bash
claude mcp add web -- clio-kit mcp-server web --remote-url http://localhost:8089
```

`--remote_url` is accepted as an alias for clients or scripts that prefer
underscores. The remote URL provides SearXNG search, DOI resolution, document
conversion, task-backend discovery, and per-agent Valkey credentials. If the
deployment requires authentication, set `WEB_REMOTE_TOKEN` in the MCP process.

For standalone keyless search without remote document conversion:

```bash
claude mcp add web -- clio-kit mcp-server web --provider ddg
```

### What works without a remote deployment

| Capability | Standalone (`--provider ddg`, no `--remote-url`) | With `--remote-url` |
|---|---|---|
| `search` | DuckDuckGo (`query`, `count`) | SearXNG with native selectors |
| `fetch` of an HTTP(S) URL (HTML to Markdown, plain text), inline or `to_file=True` | yes | yes |
| `fetch` as a task (create, status, result, cancel) | yes, in-memory task backend | yes, durable Valkey backend |
| `fetch` of a DOI (bare, `doi:`, or `doi.org` URL) | no: fails with "Document enrichment requires ..." | yes |
| PDF, Office, XML, and image conversion | no: binary content is only saved with `to_file=True` | yes |
| `fetch_events` | no: fails with the same error | yes |

Read `web://capabilities` to see what the running installation supports
(`document_enrichment`, `task_backend`).

Legacy `--address` and `--document-address` options remain compatible, but only
`--remote-url` enables automatic durable Valkey discovery.

## Task contract

`fetch(target)` accepts ordinary MCP calls and returns content inline for
clients without the tasks extension. Clients that negotiate tasks receive a task
handle immediately at the protocol level. `tasks/get` reports the latest
download or conversion message, terminal results are returned through the task,
and `tasks/cancel` cancels any active backend document conversion. There is no
fixed overall conversion timeout; only individual network requests have bounded
timeouts.

`fetch_events(conversion_id, after_sequence=0, limit=100)` returns the ordered,
persistent backend log when the latest task message is not enough to diagnose a
conversion. Failures describe the stage, cause, retryability, conversion ID, and
an actionable remediation without exposing raw third-party exception text.

`search(query, count=5)` remains synchronous because ordinary web search is a
bounded request-response operation. SearXNG installations additionally expose
`category`, `engines`, `language`, `time_range`, `pageno`, and `safesearch`.
There is intentionally no `deep_search` tool: multi-step research is agent
semantics, not a backend tool semantic.

## Development

```bash
uv sync --prerelease allow
uv run ruff check --fix .
uv run ruff format .
uv run pyright src tests
uv run pytest -m "not integration"
WEB_MCP_LIVE=1 WEB_REMOTE_URL=http://localhost:8089 uv run pytest -m integration
```
