# Scientific dataset report

Create a small HDF5 dataset report with a CSV, figure and checked statistics.

```bash
claude plugin install clio-dataset-report@clio-kit
```

Install the CLIO Kit launcher first; the hook/helper also needs `python3`.
This plugin consumes HDF5, Pandas and Plot MCP plugins and `clio-agents` through
dependencies. It adds `creating-dataset-report` and a read-only verification
hook. Invoke `/clio-dataset-report:creating-dataset-report` with your file,
dataset path, column meanings/units and a new output directory.

The helper checks count, mean, median, min/max, unchanged source and output
presence. Limits: 64 MiB source/CSV and 100,000 CSV rows. Review data conversion,
figure meaning and scientific claims separately. The native hook runs after
Write/Edit of the opted-in report/manifest; shell/MCP writes require the explicit
final check described by the skill. It does not block writes or certify prose.

The complete native installation targets Claude Code. The skill and Python
helper can be copied to another compatible agent; configure MCPs separately,
and replace the host-specific review/hook route with explicit verification.
See [plugin guide](../../docs/plugins.md) and [authoring](../../docs/authoring.md).
