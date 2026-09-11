# Slurm MCP setup and verification

Install the launcher using the [CLIO Kit setup guide](../../../setup.md).
The MCP process needs working Slurm client commands and permission to access the
site's scheduler. Check `sinfo` and `squeue` in the same environment first.
For native setup, see the [Slurm installation guide](slurm_installation/SLURM_INSTALLATION_GUIDE.md).

```bash
clio-kit doctor --server slurm --connect
clio-kit mcp-server slurm -- --help
```

Configure your agent with command `clio-kit` and arguments
`["mcp-server", "slurm"]`. The agent starts the stdio subprocess. There are no
per-tool REST endpoints such as `/submit_slurm_job_handler` or `/health`.

## Verify a real job

1. Use `slurm_cluster` to inspect available partitions and queue state.
2. Write a small shell script with a known output, using a site-approved
   partition and resource request.
3. Call `slurm_submit` with the script's absolute path and retain its
   `scheduler_native_id`.
4. Call `slurm_describe` with that ID and `output="both"` until terminal;
   check the exit status and exact output.
5. For a cancellation test, submit a separate short sleep job and call
   `slurm_cancel` with matching `job_id` and `confirm_job_id`. Verify its eventual
   state; an accepted cancellation request is not a terminal-state guarantee.

The [agent contract](agent-contract-v3.md) defines all five default tools,
result limits and the optional legacy/admin profiles. The
[example scripts](../example_scripts/README.md) need local resource settings.

## Developer checks

From the repository root:

```bash
uv run --frozen --directory clio-kit-mcp-servers/slurm pytest -q
uv run --frozen --directory clio-kit-mcp-servers/slurm slurm-mcp --help
```

Tests and successful MCP initialization do not replace a real scheduler job.
Read stderr and the job's output files when diagnosing errors; keep stdout
reserved for protocol messages.
