---
title: Pubchem MCP
description: "Look compounds up in PubChem by name (candidates when a name is ambiguous) and fetch their 3D structures as SDF."
---

import MCPDetail from '@site/src/components/MCPDetail';

<MCPDetail
  name="Pubchem"
  icon="⚗️"
  category="Utilities"
  description="Look compounds up in PubChem by name (candidates when a name is ambiguous) and fetch their 3D structures as SDF."
  version="1.0.0"
  actions={["resolve_name", "get_compound", "fetch_structure"]}
  platforms={["claude", "cursor", "vscode"]}
  keywords={["MCP", "chemistry", "PubChem", "SMILES", "SDF", "compound"]}
  license="BSD-3-Clause"
  tools={[{"name": "resolve_name", "description": "Find the PubChem compound a name means. A name PubChem maps to exactly one compound returns status 'resolved' with its CID, title, IUPAC name, formula, molecular weight (g/mol), isomeric and canonical SMILES, InChIKey and stereo centre counts. A name that maps to several compounds or to none ('xylene') returns status 'ambiguous' with up to max_candidates candidates (found by PubChem's name dictionary and word search) for the person to choose from.", "function_name": "resolve_name"}, {"name": "get_compound", "description": "The PubChem record of one compound by CID: title, IUPAC name, formula, molecular weight (g/mol), isomeric and canonical SMILES, InChIKey, charge, stereo centre counts, components, and its first 20 synonyms (common names, CAS numbers).", "function_name": "get_compound"}, {"name": "fetch_structure", "description": "Download a compound's 3D conformer (with hydrogens) from PubChem as an SDF file under output_dir/chemistry/pubchem/cid<CID>-<name>/, plus a JSON compound record. Returns the structure record {path, format, formula, n_atoms, sha256}. When PubChem has no 3D conformer the result says so (has_3d false, fallback '2d_record') and writes the flat 2D record; embed 3D coordinates from the SMILES with ase.smiles_to_structure instead.", "function_name": "fetch_structure"}]}
>

{/* clio-kit:usage:start */}

### Usage

Available tools: `resolve_name`, `get_compound`, `fetch_structure`.

Inspect the tool input schema in your MCP client before calling it. Use a small known input, check the result and any reported errors, then expand to the intended workload.

{/* clio-kit:usage:end */}

</MCPDetail>
