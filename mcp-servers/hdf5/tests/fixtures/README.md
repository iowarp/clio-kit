# Test fixtures

`conventions/`: one small file per HDF5 convention, each written by that
convention's own library on 2026-10-06, so the plugin's convention entries are
tested against real output rather than files built to match them.

| File | Written by |
|---|---|
| `n.nxs` | nexusformat 2.1.0 (root `file_name` attribute replaced and the file rewritten, to drop a local path) |
| `cf.nc` | netCDF4-python 1.7.4, netCDF-C 4.9.3; CF-1.8 attributes |
| `n.nwb` | pynwb 4.2.0 |
| `a.h5ad` | anndata 0.13.4 |
| `pmd_100.h5` | openPMD-api 0.17.1, file-based iteration encoding |
| `img.vtkhdf` | VTK `vtkHDFWriter`, ImageData |
| `store.h5` | pandas `to_hdf(format="table")` with PyTables 3.11.1 |

`agent_plugins_1_0_0/`: the Agent Plugins 1.0.0 manifest and MCP
configuration schemas, copied from
https://github.com/agentplugins/agent-plugins-spec at
ff8ab5e392cc87bd88d87c060815a87490e51003 (`schemas/1.0.0/`). Apache License 2.0.
