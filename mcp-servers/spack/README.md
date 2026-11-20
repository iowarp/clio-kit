# Spack MCP

Install the launcher using the [CLIO Kit setup guide](../../setup.md) before
using the commands below. See [agent integrations](../../README.md#agent-integrations)
for MCP and skill configuration.

Spack MCP exposes a compact, structured package-management surface:

```text
spack_find
spack_locate
spack_search
spack_info
spack_prepare_site
spack_site_query
```

`spack_install` and `spack_environment` are admin tools (`--profile admin`).
Users install packages on a site through a jarvis install job
(`jarvis ppl install ... +submit`), which runs on a compute node, not through
this server.

The default server intentionally has no stateful `spack_load` tool. Each MCP
call runs in its own process context, so changing that process environment would
not affect a later agent or scheduler job. Pass installed specs to JARVIS through
`jarvis_run(spack_specs=[...])`; JARVIS captures a filtered environment delta,
persists it in the pipeline, and reloads it inside direct or scheduled execution.
`spack_locate` returns a canonical `load_spec` in `/<dag_hash>` form; pass that
value to `jarvis_run` so a later install cannot make the runtime selection
ambiguous. `spack_install` echoes the same `load_spec` on success, so a caller
never has to re-locate what it just installed.

An ordinary `spack_find` query with no installed matches is successful typed
data: `count` is `0` and `packages` is `[]`. Calling `spack_locate` for an absent
package instead returns the structured `not_installed` semantic, enriched with
whether a recipe is available to install (and in which repo) or exists in no
registered repo at all, so an agent can route the install to a jarvis install
job without a separate lookup. Other nonzero Spack invocations
remain structured MCP errors.

## Sites reached through clio-relay

On an HPC site, CLIO uses its own Spack that reuses what the site already has
(the site Spack's install tree and modules, such as a site MPI that spans
nodes) and builds the rest. This server never opens an SSH connection itself.
`spack_prepare_site` and `spack_site_query` work in two calls:

1. Called without `relay_result`, the tool returns `relay_call`: the clio-relay
   `exec` arguments (`site`, an argv `command`, `timeout_s`,
   `max_output_bytes`). `spack_prepare_site` also returns `approval`, the text
   to show the person before anything is written.
2. The agent calls clio-relay `exec` with those arguments unchanged, then calls
   the tool again with the same arguments and `relay_result` set to the exec
   structured result. The tool checks that the result belongs to the planned
   argv and parses it.

`spack_prepare_site(site, site_spack=...)` does the following, and is safe to
rerun (each step reports `present`, `unchanged` or `written`):

- shallow-clones the Spack release the site runs (read from the site Spack,
  or `spack_version`) into `<site_home>/spack`, and checks the release's
  pinned commit;
- writes `upstreams.yaml` pointing at the site install tree (asked from the
  site Spack) and its Lmod/Tcl module trees;
- writes `config.yaml` (caches under `<root>/var/clio`, build stage under
  `$tempdir/$user`), `concretizer.yaml` (`reuse: true`) and `bootstrap.yaml`
  (bootstrap root under `<root>/var/clio`, binary bootstrap only, never a
  clingo source build);
- runs `spack compiler find` and `spack external find --not-buildable` for
  build tools only. MPI is never externalized from PATH and is refused as a
  build tool;
- sets `packages:mpi:buildable:false` and records module MPIs (MPI modules
  outside the site Spack's own module trees, such as Ares's `mpich/4.1.1`) as
  candidate externals with both `prefix:` and `modules:`. The prefix comes from
  a light `module load` and `command -v mpiexec` in a throwaway login shell; a
  module whose prefix cannot be found is reported as a warning and not recorded.

All of this config lives in the clone's own `etc/spack` (Spack's site scope).
Every command runs with `SPACK_DISABLE_LOCAL_CONFIG=1`, which in Spack 0.23
drops both the system and the user scopes, and with
`SPACK_USER_CONFIG_PATH`/`SPACK_USER_CACHE_PATH` set to
`<site_home>/spack-config`, so the person's `~/.spack` and `~/spack` never leak
in. The report ends with what jarvis-cd must be told:

```text
JARVIS_SPACK_ROOT=<site_home>/spack
JARVIS_SPACK_CONFIG=<site_home>/spack-config
```

`spack_prepare_site` does not install jarvis-cd; clio-relay bootstraps it.

`spack_site_query(site, query)` researches the CLIO Spack: `find` (installed
and upstream packages), `info` (one recipe), `spec` (`spack spec -I -l`: which
nodes are installed, upstream, external or still to build) and `modules`
(`module -t avail`). The first `spec` on a fresh clone downloads the clingo
bootstrap binary (about 8 MB) into `<root>/var/clio/bootstrap`.

Login nodes are shared, so both tools do only light, short work there: git
clone, writing config, `compiler find`, `external find`, `find`, `info` and
`spec`. They never run `spack install` or a build, and nothing outlives the
relay call (relay's exec kills the process group when it ends). On a Secure
(guarded) relay site, exec allows only its read-only list, which has no `bash`,
`git` or `spack`; the tools then return `status: refused` and the agent tells
the person instead of working around the guard.

## Recipe discovery

`spack_find`/`spack_locate` only see what is already **installed**.
`spack_search(query)` and `spack_info(package)` answer the broader question of
recipe **availability**:

- `spack_search` enumerates every repo registered with `spack repo list` and
  walks each repo's recipe directory directly (never `spack list`, which
  is broken on at least one deployment this server targets) to fuzzy-match the
  query against real recipe names -- matching normalizes `-`/`_` and case, so
  a query like `py-numpy` matches a `py_numpy` repo directory (Spack's modern
  Python-module-safe layout) exactly, not merely as a fuzzy guess. Each match
  reports which repo(s) declare it and whether it is already installed. A repo
  whose recipe directory cannot be read (missing path, permission denied, ...)
  degrades that one repo's contribution -- named in `repos_unreadable` -- but
  never empties out or fails the rest of the search. Results beyond the
  25-match cap set `truncated: true` with the real `total_matches`. If `spack
  find` itself fails, only the installed-state half of the answer degrades
  (`installed_state_degraded` + `installed_state_degraded_reason`); recipe
  availability is unaffected.
- `spack_info` describes one recipe's versions, variants, and description.
  It probes `spack info` first; if that subcommand is unavailable, fails, or
  returns an incomplete parse (a missing expected section), it falls back to
  statically parsing the recipe's `package.py` (never imported or executed,
  only read as source) and marks the result `degraded: true` with a
  `degraded_reason` naming the actual cause -- never silently.

## Install concretization (admin)

`spack_install` is an admin tool for a local Spack. It must not run on an HPC
login node; site installs run as jarvis install jobs. It always selects the
Spack concretization policy explicitly:

- `spack_install(spec, reuse=true)` runs `spack install --reuse <spec>` and may
  reuse compatible installed packages or buildcaches while concretizing.
- `spack_install(spec, reuse=false)` runs `spack install --fresh <spec>` and
  excludes installed packages and buildcaches while concretizing.

Fresh concretization does not mean "blindly reinstall an existing concrete
hash." Agents should first call `spack_find`, install only when the required
software is absent, then call `spack_locate` and pass its canonical `load_spec`
to `jarvis_run`.

`spack_install` runs synchronously today (streaming/task augmentation is
deferred to the kit tasks-semantics slice, SEP-2663) with a configurable
`timeout_seconds`. The full build log is captured to disk -- unbounded, not
just the response's bounded tail -- under `SPACK_MCP_INSTALL_LOG_DIR` (default
a `spack-mcp/install-logs` directory under the system temp dir). On success the
result carries `prefix`, `load_spec`, `log_path`, and a bounded `log_tail`.
Failure is one of several typed, distinguishable errors, each naming the
recovery affordance:

- `recipe_not_found` -- every registered repo was successfully read and none
  declares this package; `detail` lists the repos searched (composes with
  `spack_search`'s discovery). Spack is never even invoked in this case.
- `availability_unknown` -- the catalog found no match, but at least one
  registered repo could not be read; `detail` names which repo(s), so the
  install is refused without guessing rather than silently trusting an
  unverified negative. Spack is not invoked here either.
- `build_failure` -- Spack ran and exited nonzero; `detail` carries the log
  path and a tail of the failure.
- `timed_out` -- exceeded `timeout_seconds`; `detail` carries the log path so
  progress can be inspected without re-running the install.
- `capture_failed` -- the bounded subprocess capture itself failed after
  Spack was launched; `detail` carries the log path.
- `log_unwritable` -- the install log directory or file could not be
  created/opened; Spack is never invoked in this case.

### Breaking change in the v2.3 -> v3 registry contract

`spack_install` moved from the default (user) profile to the admin profile,
and the user profile gained `spack_prepare_site` and `spack_site_query`. An
agent that called `spack_install` on the default server must route installs
to a jarvis install job, or run the server with `--profile admin`.

### Breaking change in the v2.1 -> v2.2 registry contract

`spack_install`'s output shape changed: v2.1's `packages` (a list) and
`stdout_excerpt` are **removed**, replaced by `package` (one object),
`prefix`, `load_spec`, `log_path`, and `log_tail`. This is a breaking change
to the tool's output schema, not an additive one -- an external agent bound
to `spack_install.output.packages`/`.stdout_excerpt` must update. No consumer
inside this repository (or the `clio-agent-marketplace` submodule) read
either removed field at the time of the v2.2 release.

Run the default server with:

```bash
uv tool install 'clio-kit'
clio-kit mcp-server spack
```

When Spack is not on the service PATH, configure the audited executable
explicitly instead of modifying the worker environment:

```bash
clio-kit mcp-server spack -- \
  --spack-command /path/to/spack
```

The path must resolve to an executable file. `SPACK_MCP_COMMAND` provides the
equivalent site-level override; an explicit CLI argument takes precedence.

Environment diagnostics persist only a conservative path/toolchain allowlist.
Sites can add exact non-secret variable names with the comma-separated
`CLIO_SPACK_ENV_ALLOWLIST` or `SPACK_MCP_ENV_ALLOWLIST`; sensitive and transient
names remain rejected even when configured.

An operator-only profile adds `spack_install` and a diagnostic that returns the
structured environment without claiming to mutate future processes:

```bash
clio-kit mcp-server spack -- --profile admin
```

Every successful tool returns a versioned Pydantic result. Command failures are
MCP error results whose text is a JSON `spack.mcp.error.v1` document. Specs are
passed as subprocess arguments rather than shell text, output is bounded, and
install timeouts terminate the process group.
