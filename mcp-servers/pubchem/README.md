# PubChem MCP

Look compounds up in [PubChem](https://pubchem.ncbi.nlm.nih.gov/) by name and fetch
their structures for a simulation. When a name is ambiguous the server returns
candidates for the person to choose from. It never takes the first match.

## Tools

| Tool | What it does |
| --- | --- |
| `resolve_name(name, max_candidates=8)` | A name PubChem maps to exactly one compound returns `status: "resolved"` with the CID, title, IUPAC name, formula, molecular weight (g/mol), isomeric and canonical SMILES, InChIKey and stereo centre counts (defined and undefined). A name that maps to several compounds or to none returns `status: "ambiguous"` with up to `max_candidates` candidates, each with the same fields, its first synonyms and `found_by` (how it was found), and a `next_step` telling the agent to ask the person. |
| `get_compound(cid)` | The same record for one CID, with its first 20 synonyms (common names, CAS numbers). |
| `fetch_structure(cid, output_dir?)` | Writes PubChem's 3D conformer (with hydrogens) as SDF, plus a JSON compound record, and returns the structure record `{path, format, formula, n_atoms, sha256, dimension}`. When PubChem has no 3D conformer (metals, salts, mixtures, large flexible molecules), `has_3d` is false and `fallback` is `"2d_record"`: the flat 2D record is written and the text says to embed 3D coordinates from the SMILES with `ase.smiles_to_structure`. |

Example: `resolve_name("xylene")` returns `ambiguous`, because PubChem has no compound
named exactly "xylene". The candidates include o-xylene (CID 7237), m-xylene (CID 7929) and
p-xylene (CID 7809), followed by substituted xylenes such as 4-iodo-m-xylene.
`resolve_name("p-xylene")` resolves to CID 7809. `resolve_name("carvone")` resolves to the
racemic record (CID 7439), and the text notes that its stereo centre is undefined.

### How candidates are gathered

1. The exact name lookup (`/compound/name/<name>/cids`). One CID: resolved.
2. Otherwise: PubChem's autocomplete dictionary (`/rest/autocomplete/compound/<name>`).
   Suggestions that contain the name as a whole word come first, then the shortest. Each
   suggestion is looked up exactly.
3. PubChem's word search (`name_type=word`), in the order PubChem returns its results.
4. One batch request each for the candidates' properties and synonyms.

The order only arranges the list. It does not pick a compound for the person.

Outputs go to `<output_dir or $CLIO_KIT_ARTIFACTS or cwd>/chemistry/pubchem/cid<CID>-<name>/`.

## Needs

Network access to `pubchem.ncbi.nlm.nih.gov` (HTTPS, no key). Requests are kept to at
most 5 per second, as PubChem asks. A busy reply (503) or a network error is retried
three times with exponential backoff. After that the error is `pubchem_unavailable`.

Errors end with `[reason]`: `not_found` (no compound for that name or CID),
`pubchem_unavailable`, `bad_request`.

## Tests

```bash
uv run --project mcp-servers/pubchem pytest mcp-servers/pubchem -m "not integration"   # offline
uv run --project mcp-servers/pubchem pytest mcp-servers/pubchem -m integration         # live PubChem
```

The offline tests replay PubChem responses recorded from live calls
(`tests/fixtures/pubchem_responses.json`). To record them again, run
`uv run python -m tests.replay` from `mcp-servers/pubchem`.

## NOTICE

Part of the Chemistry in CLIO work, which ports ChemGraph (argonne-lcf/ChemGraph,
Apache-2.0) to clio-kit. ChemGraph uses `pubchempy`. This server is a new, small PUG REST
client written for clio-kit and contains no ChemGraph code. PubChem data is provided by
the NCBI. See PubChem's [data usage policy](https://www.ncbi.nlm.nih.gov/home/about/policies/).
