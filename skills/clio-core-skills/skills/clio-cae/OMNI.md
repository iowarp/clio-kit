# OMNI files, sources and the `clio-cae` CLI

## The `clio-cae` command

```
clio-cae <omni_file_path>     # load, run `run:` steps, submit, wait
clio-cae -V | --version       # print version, exit 0
clio-cae -h | --help          # usage, exit 0
```

There are no other subcommands or flags; any other argument count prints
usage and exits 1. Aliases: `clio_cae`, `clio_cae_omni`. Not in the pip wheel.

What it does:

1. Connects as a client to the runtime named by the usual client config
   (`CLIO_SERVER_CONF` or `~/.clio/clio.yaml`; `CLIO_PORT`,
   `CLIO_SERVER_ADDR`). The runtime must be running.
2. Parses the OMNI file. A YAML error or a missing required key aborts before
   anything is sent. Unknown per-transfer keys log a warning and are ignored.
3. Runs every `run:` step locally, in the client. If any fails, nothing is
   submitted.
4. Sends all transfers to the CAE pool **400.0** in one `ParseOmni` call and
   waits.
5. Exit 0 only if the assimilation code and the task code are both 0 **and**
   at least one transfer was scheduled. Otherwise exit 1 with the code and a
   hint. Dataset-level causes are in the **runtime** log, not the CLI output.

The success message suggests polling `cte_search`; no such command ships.
List the tag instead (`clio_cee` `context_query`, or `TagQuery`/`BlobQuery`
from `clio_cte_core_ext`).

## File format

Top level: a map with a required `transfers:` sequence. Other top-level keys
(`name`, `version`, ...) are ignored without a warning.

| key (per transfer) | type | required | meaning |
|---|---|---|---|
| `src` | string | yes | Source URL. The scheme (text before `://` or `::`) picks the assimilator; see the source table. |
| `dst` | string | yes | `iowarp::<tag>` for every source except `http(s)://` (which needs `file::<path>`) and `format: hdf5_vol` (`iowarp::` with no name). |
| `format` | string | yes | `binary`, `hdf5`, `hdf5_vol`, `string`, `globus`. Informational for most sources: the scheme decides. It matters for `hdf5_vol` and for which scheme a `run:` output gets (`hdf5*` -> `hdf5::`, else `file::`). |
| `depends_on` | string | no (`""`) | The exact `src` or `dst` string of an **earlier** transfer. Transfers already run in order; this is checked, not scheduled. A dependency on a later transfer, on a transfer `name`, or on nothing rejects the whole file (rc -3) before anything runs. |
| `range_off` | integer, bytes | no (0) | Start offset in the source. |
| `range_size` | integer, bytes | no (0 = to end of file) | Length. `range_off + range_size` past the end fails (rc -6). |
| `src_token` | string | no | Source credential (Globus access token; HTTP bearer token). `${VAR}` expanded, so keep secrets in the environment. |
| `dst_token` | string | no | Destination credential (Globus). `${VAR}` expanded. |
| `dataset_filter.include_patterns` | list of globs | no | HDF5 datasets to import. Empty = all. |
| `dataset_filter.exclude_patterns` | list of globs | no | HDF5 datasets to skip; applied first. Globs use `fnmatch` without flags: `*` **crosses `/`**, `?` and `[...]` work. |
| `run` | path | with `run_out` | Script run in the client before submission as `run <source path or URL> <run_out>`. `file::`/`hdf5::` prefixes are stripped from the first argument. Relative paths resolve against the OMNI file's directory; `${VAR}` expanded. Must be executable (Linux/macOS: `chmod +x`; Windows: run via `cmd /C`). |
| `run_out` | path | with `run` | File the script must write. It is deleted before the script runs; exit 0 without the file is an error. The transfer then ingests `file::<run_out>` (or `hdf5::` if `format` starts with `hdf5`), and later `depends_on` that named the old `src` are rewritten. |

### Full example

```yaml
name: nightly_ingest            # ignored by the loader
transfers:
  # 1. Raw binary, whole file
  - src: "file::/data/run42/output.bin"
    dst: "iowarp::run42_raw"
    format: binary

  # 2. A 64 MiB window of the same file
  - src: "file::/data/run42/output.bin"
    dst: "iowarp::run42_window"
    format: binary
    range_off: 1073741824
    range_size: 67108864

  # 3. Convert Parquet to HDF5 with a script, then import selected datasets
  - src: "file::/data/run42/table.parquet"
    dst: "iowarp::run42_table"
    format: hdf5
    run: "./parquet_to_h5.py"           # called as: parquet_to_h5.py /data/run42/table.parquet /tmp/table.h5
    run_out: "/tmp/table.h5"
    dataset_filter:
      include_patterns: ["/columns/*"]
      exclude_patterns: ["*_index"]

  # 4. Only after transfer 1 (exact src or dst of an earlier transfer)
  - src: "s3://my-bucket/run42/meta.json"
    dst: "iowarp::run42_meta"
    format: binary
    depends_on: "iowarp::run42_raw"

  # 5. Download a web file to local disk (not into CTE) ...
  - src: "https://data.example.org/run42/extra.bin"
    dst: "file::/data/run42/extra.bin"
    format: binary
    src_token: "${EXAMPLE_TOKEN}"
  # ... then import the downloaded file
  - src: "file::/data/run42/extra.bin"
    dst: "iowarp::run42_extra"
    format: binary
    depends_on: "file::/data/run42/extra.bin"
```

Transfers 3 and 4 need an HDF5- and S3-enabled build; the rest work with the
pip wheel's runtime on Linux/macOS (via `clio_cee`, since the wheel has no
`clio-cae`; `run:` steps and `dataset_filter` need the CLI).

Minimal smoke file:

```yaml
transfers:
  - src: "file::/home/me/data/hello.txt"
    dst: "iowarp::cae_smoke"
    format: binary
```

## Sources

| `src` scheme | what lands in CTE | build option | in pip wheel |
|---|---|---|---|
| `file::/abs/path` | tag `<dst>`: `description` + `chunk_0..` (1 MiB). Ranges apply. Read by the runtime process, checked against `allowed_roots`. | always | yes |
| `string::<blob>` | one blob `<blob>` holding `src_data` verbatim. API only: an OMNI file has no `src_data` key. | always | yes |
| `hdf5::/abs/path.h5` (`format: hdf5`) | one tag per dataset `<dst>/<dataset path>`: `description` (dtype, dims) + `chunk_N` (1.5 MiB). Each dataset is read whole into RAM first. Multi-node: datasets are spread round-robin across nodes, each node opening the file itself (it must be visible on every node). | `CLIO_CORE_ENABLE_HDF5=ON` + HDF5 | no |
| `hdf5::...` with `format: hdf5_vol` | the file staged in the HDF5 VOL connector's cache layout, under the tag the connector derives. `CLIO_VOL_CHUNK_SIZE` must match the application's. | same | no |
| `http://`, `https://` | **nothing in CTE**: a download to the local `file::<path>` in `dst`, by `curl` child processes (parallel ranged GETs). Skips if the destination already holds the exact bytes (`CAE_HTTP_OVERWRITE=1` forces). Credentials: `src_token` (bearer) or per-site sections in `~/.clio/cae_http_auth.ini` (`CAE_HTTP_AUTH_FILE`). Blocks a worker for the whole download. | non-Windows builds | yes (Linux/macOS) |
| `s3://bucket/key`, `s3::bucket/key` | like `file::` (chunked into one tag). Any S3-compatible endpoint. Credentials from the runtime's AWS environment. | `CAE_ENABLE_S3=ON` + Poco + AWS SDK for C++ | no |
| `gs://bucket/obj`, `gcs://...` | like `file::`. Application Default Credentials. | `CAE_ENABLE_GCS=ON` + google-cloud-cpp | no |
| `globus://<endpoint>/<path>`, `https://app.globus.org/...` | Globus transfer; destination `globus://...` or `file::<path>`. Token from `src_token` or `GLOBUS_ACCESS_TOKEN`. | `CAE_ENABLE_GLOBUS=ON` + Poco + nlohmann_json | no |
| anything else (ADIOS2 `.bp`, Parquet, CSV as tables, ...) | rejected: `Unsupported protocol` / rc -2 | none | - |

Cloud and Globus options default OFF and switch themselves off with a CMake
warning if their libraries are missing. Check the configure log for
`CAE S3 support: ENABLED` etc.

## Error codes (`result_code_`, `context_bundle` return)

| code | meaning |
|---|---|
| 0 | every transfer done |
| -1 | request could not be decoded; or (assimilator) `dst` is not `iowarp::` |
| -2 | no assimilator: unsupported or uncompiled scheme, **or the path policy refused the source**; or (file) empty tag name |
| -3 | `depends_on` cannot be satisfied (nothing ran); or (file) tag creation failed |
| -4 | (file) no source path; (hdf5) cannot open file |
| -5 | (file) source missing, unreadable or 0 bytes; (hdf5) dataset discovery failed |
| -6 | (file) range past end of file; (hdf5) one or more datasets failed |
| -7, -8 | (file) open / seek failed; (hdf5) re-open failed |
| -9 | (file) read or `description` put failed; (hdf5) `include_patterns` matched nothing |
| -10 | (file) a chunk PutBlob failed (tier full or chain error) |
| -13, -20 | (C++ `on_fault`) registry or setup failed |
| HTTP: -status | the server's HTTP status (e.g. -404); -13 transport failure; -3 bad `dst` |

The CLI prints the code and `error_message_`; the runtime log has the cause.
