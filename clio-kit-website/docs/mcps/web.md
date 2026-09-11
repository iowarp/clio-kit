---
title: Web MCP
description: "Web MCP server providing curated fetch + search tools for agentic web access"
---

import MCPDetail from '@site/src/components/MCPDetail';

<MCPDetail
  name="Web"
  icon="🔧"
  category="Utilities"
  description="Web MCP server providing curated fetch + search tools for agentic web access"
  version="1.1.0"
  actions={["fetch", "search"]}
  platforms={["claude", "cursor", "vscode"]}
  keywords={["web", "fetch", "search", "mcp", "llm-integration", "agentic-web"]}
  license="BSD-3-Clause"
  tools={[{"name": "fetch", "description": "Fetch an HTTP(S) URL with a streamed size cap and timeout, convert HTML to Markdown, and return the content inline or (to_file=True) write it to a local file and return its path.", "function_name": "fetch"}, {"name": "search", "description": "Search the web via a configurable provider (keyless DuckDuckGo by default; self-hosted SearXNG; optional BYO-key Brave or Tavily) and return ranked results. SearXNG supports category, engine, language, time-range, page, and safe-search selectors.", "function_name": "search"}]}
>

{/* clio-kit:usage:start */}

### Search and retrieve a source

Use `search` to find candidate pages, then `fetch` the specific page before
making claims about its contents. Record the source URL and retrieval context.
The fetch result is bounded; request file output when more content is needed.

### Verify a research claim

Compare a project or dataset landing page with the corresponding paper or
catalogue metadata. A search snippet alone is not proof of the full claim.
Report unavailable pages and truncated responses explicitly.

{/* clio-kit:usage:end */}

</MCPDetail>
