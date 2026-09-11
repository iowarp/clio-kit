---
title: Web MCP
description: "Provider-fixed web search plus transparent URL, DOI, and document fetching"
---

import MCPDetail from '@site/src/components/MCPDetail';

<MCPDetail
  name="Web"
  icon="🔧"
  category="Utilities"
  description="Provider-fixed web search plus transparent URL, DOI, and document fetching"
  version="2.1.3"
  actions={["fetch", "fetch_events", "search"]}
  platforms={["claude", "cursor", "vscode"]}
  keywords={["web", "fetch", "search", "mcp", "llm-integration", "agentic-web"]}
  license="BSD-3-Clause"
  tools={[{"name": "fetch", "description": "Fetch an HTTP(S) URL or DOI inline or as a task. HTML and text are read locally; supported documents use CLIO Web Search conversion when configured.", "function_name": "fetch"}, {"name": "fetch_events", "description": "Query the full ordered backend event log for a document fetch conversion.", "function_name": "fetch_events"}, {"name": "search", "description": "Search the web using this installation's fixed ddg provider.", "function_name": "search"}]}
>

{/* clio-kit:usage:start */}

### Search and retrieve a source

Use `search` to find candidate pages, then `fetch` the specific page before
making claims about its contents. Record the source URL and retrieval context.
`fetch(target=...)` supports ordinary MCP calls, returning fetched content
inline. A task-capable client receives a task; follow its progress to completion
and read the terminal result before citing content. Prefer tasks for long conversions.
Task IDs and progress messages are not source content. Request `to_file=True`
for file output. PDF and structured-document conversion requires a configured
CLIO Web Search service; inspect `web://capabilities` for the active setup.

### Verify a research claim

Compare a project or dataset landing page with the corresponding paper or
catalogue metadata. A search snippet alone is not proof of the full claim.
Report unavailable pages and truncated responses explicitly.

{/* clio-kit:usage:end */}

</MCPDetail>
