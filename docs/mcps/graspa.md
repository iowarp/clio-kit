---
title: Graspa MCP
description: "Gas adsorption in MOFs and zeolites by grand-canonical Monte Carlo with gRASPA: run, prepare for a GPU node, read and rank."
---

import MCPDetail from '@site/src/components/MCPDetail';

<MCPDetail
  name="Graspa"
  icon="🫧"
  category="System Management"
  description="Gas adsorption in MOFs and zeolites by grand-canonical Monte Carlo with gRASPA: run, prepare for a GPU node, read and rank."
  version="1.0.0"
  actions={["run_graspa", "prepare_graspa", "read_graspa", "rank_frameworks", "graspa_status"]}
  platforms={["claude", "cursor", "vscode"]}
  keywords={["MCP", "chemistry", "adsorption", "GCMC", "gRASPA", "MOF", "zeolite"]}
  license="BSD-3-Clause"
  tools={[{"name": "run_graspa", "description": "Compute the adsorption uptake of one gas (CO2, H2O or N2) in a framework CIF at a temperature (K) and pressure (Pa) by grand-canonical Monte Carlo with gRASPA, here. The cell is replicated so each width is at least twice the cutoff; the force field is UFF-like Lennard-Jones with TraPPE-style rigid adsorbates; framework charges 'none' (default) or 'cif'. Writes the run folder (inputs, gRASPA output, run.json, result.json) and reports the uptake in mol/kg with its block-average error bar and molecules per unit cell. Needs gRASPA and a GPU here (binary_missing otherwise; use prepare_graspa).", "function_name": "run_graspa"}, {"name": "prepare_graspa", "description": "Write a gRASPA run folder (framework CIF, force field, adsorbate, simulation.input with the unit cells, run.json) for one adsorbate at a temperature and pressure, without running it: for a GPU node elsewhere (a Slurm job that runs gRASPA in the folder with stdout saved to graspa_stdout.txt). Same settings as run_graspa.", "function_name": "prepare_graspa"}, {"name": "read_graspa", "description": "Read a finished gRASPA run folder (made by prepare_graspa or run_graspa, run here or on another machine): the uptake in mol/kg with its error bar, molecules per unit cell, mg/g, g/L and the Henry coefficient when gRASPA printed them. Writes result.json in the folder.", "function_name": "read_graspa"}, {"name": "rank_frameworks", "description": "Rank frameworks from gRASPA results (result.json files or folders holding them; one adsorbate): by uptake at the adsorption condition, or, with a desorption condition, by working capacity (uptake at adsorption minus at desorption). Repeats are averaged; a framework without a completed run at every requested condition is excluded with the reason. Writes ranking.csv; top_fraction selects the best share.", "function_name": "rank_frameworks"}, {"name": "graspa_status", "description": "Whether the gRASPA executable is available here, where it was found (GRASPA_COMMAND, CHEMGRAPH_GRASPA_EXECUTABLE or PATH), and how to get it.", "function_name": "graspa_status"}]}
>

{/* clio-kit:usage:start */}

### Usage

Available tools: `run_graspa`, `prepare_graspa`, `read_graspa`, `rank_frameworks`, `graspa_status`.

Inspect the tool input schema in your MCP client before calling it. Use a small known input, check the result and any reported errors, then expand to the intended workload.

{/* clio-kit:usage:end */}

</MCPDetail>
