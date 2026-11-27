---
title: Avogadro MCP
description: "Avogadro 2 for building and editing structures in 3D, streamed into a view; edits and selections come back as files."
---

import MCPDetail from '@site/src/components/MCPDetail';

<MCPDetail
  name="Avogadro"
  icon="⚛️"
  category="Utilities"
  description="Avogadro 2 for building and editing structures in 3D, streamed into a view; edits and selections come back as files."
  version="1.0.0"
  actions={["avogadro_status", "open_in_avogadro", "read_avogadro", "avogadro_snapshot", "close_avogadro"]}
  platforms={["claude", "cursor", "vscode"]}
  keywords={["MCP", "chemistry", "Avogadro", "molecular editor", "MCP Apps"]}
  license="BSD-3-Clause"
  tools={[{"name": "avogadro_status", "description": "Whether Avogadro 2 is installed here (and where), whether the virtual display and VNC server the streamed view needs (Xvfb, x11vnc) are present, what to install when something is missing, and whether Avogadro is running.", "function_name": "avogadro_status"}, {"name": "open_in_avogadro", "description": "Open Avogadro in a view, with a structure file (XYZ, SDF/MOL, PDB, CIF, CJSON, \u2026) or empty to build from scratch. The person works in it; when they choose Extensions > CLIO > Send to CLIO, the view tells you, and read_avogadro writes what they sent. Starts Avogadro the first time (seconds).", "function_name": "open_in_avogadro"}, {"name": "read_avogadro", "description": "What the person last sent from Avogadro (Extensions > CLIO > Send to CLIO), written as an XYZ structure file plus Avogadro's CJSON under output_dir/chemistry/avogadro/, with its formula and the atoms they had selected (0-based indices, file order).", "function_name": "read_avogadro"}, {"name": "avogadro_snapshot", "description": "A PNG of Avogadro's 3D view as it is now, saved and returned as an image.", "function_name": "avogadro_snapshot"}, {"name": "close_avogadro", "description": "Stop Avogadro and its virtual display; what was not sent is lost.", "function_name": "close_avogadro"}]}
>

{/* clio-kit:usage:start */}

### Usage

Available tools: `avogadro_status`, `open_in_avogadro`, `read_avogadro`, `avogadro_snapshot`, `close_avogadro`.

Inspect the tool input schema in your MCP client before calling it. Use a small known input, check the result and any reported errors, then expand to the intended workload.

{/* clio-kit:usage:end */}

</MCPDetail>
