---
title: Scientific-Catalog MCP
description: "Operator-owned scientific dataset discovery for remote agents"
---

import MCPDetail from '@site/src/components/MCPDetail';

<MCPDetail
  name="Scientific-Catalog"
  icon="🔧"
  category="Data Processing"
  description="Operator-owned scientific dataset discovery for remote agents"
  version="1.1.4"
  actions={["scientific_dataset_search", "scientific_dataset_describe"]}
  platforms={["claude", "cursor", "vscode"]}
  keywords={[]}
  license="BSD-3-Clause"
  tools={[{"name": "scientific_dataset_search", "description": "Search operator-registered scientific datasets and return bounded intrinsic summaries.", "function_name": "scientific_dataset_search"}, {"name": "scientific_dataset_describe", "description": "Return one exact operator catalog record plus a top-level dataset_descriptor. Pass dataset_descriptor unchanged as jarvis_add_step config.dataset_descriptor; do not pass the surrounding dataset record.", "function_name": "scientific_dataset_describe"}]}
>

{/* clio-kit:usage:start */}

### Discover a site dataset

Configure `SCIENTIFIC_CATALOG_FILE` or `--catalog-file` with the operator's
catalogue. Call `scientific_dataset_search`, select a returned dataset ID and
use `scientific_dataset_describe` for its metadata and descriptor. This service
describes existing datasets; it does not download or render them.

When a JARVIS package accepts `dataset_descriptor`, pass the result's named
`dataset_descriptor` unchanged, not the surrounding catalogue record. Confirm
that referenced locations are accessible to the downstream runtime. See the
[catalogue format](https://github.com/iowarp/clio-kit/blob/main/clio-kit-mcp-servers/scientific-catalog/README.md).

{/* clio-kit:usage:end */}

</MCPDetail>
