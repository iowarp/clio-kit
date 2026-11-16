# clio-chemistry

Find a compound, build and change its structure, compute its energy, vibrations and
spectra, and simulate gas adsorption in porous crystals.

```bash
claude plugin install clio-chemistry@clio-kit
```

Install the launcher and register the marketplace first; see [setup](../../setup.md).

- **MCPs:** `pubchem` (names to compounds and 3D structures), `decimer` (a drawn
  structure in an image to SMILES), `ase` (SMILES to 3D, geometry changes, energies,
  optimizations, vibrations, thermochemistry, IR), `fdmnes` (XANES spectra), `graspa`
  (GCMC adsorption uptake and framework rankings).
- **Expected output:** structure files, result JSON and CSV files (trajectories,
  frequencies, spectra, rankings) with the method, settings and units recorded.
- **Prerequisites:** `pubchem` needs network access to PubChem; `decimer` downloads its
  model (285 MiB) on first use; `ase` builds its xTB, MACE or UMA environment on first
  use, and NWChem and ORCA are separate programs; `fdmnes` needs the FDMNES executable
  and `graspa` needs gRASPA on a GPU node (both from this repository's
  [Spack packages](../../spack-repo/README.md)).

This primary bundle names dependencies; it contains no copy of server code or
skill instructions. Edit its membership in `mcp-server-versions.toml` and
regenerate the manifest. The chemistry skills and agent are in the CLIO agent
marketplace. Consult the [validation guide](../../docs/marketplace.md#scientific-acceptance-boundaries)
for tested coverage; installing a bundle does not prove every backend operation.
