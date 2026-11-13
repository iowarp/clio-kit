# CLIO extension files

Files for CLIO's `ai.iowarp.clio` extension namespace, referenced from the
`structured_files` entry in `../plugin.json`. Agent Plugins 1.0 (§8) puts
client-specific files in a top-level directory named after the namespace, so
they live here rather than at the paths in the RFC's sketch (`web/`,
`conventions/`). Other clients ignore this directory.

## viewer/

The HDF5 viewer page: h5web inside an iframe, talking to its host over the
postMessage protocol in gact-tui's `packages/hdf5-viewer/src/protocol.ts`.
It is built output and is not committed. To fill it:

```bash
cd gact-tui
pnpm --filter @clio/hdf5-viewer build
cp -r packages/hdf5-viewer/dist/. ../clio-kit/plugins/clio-hdf5/ai.iowarp.clio/viewer/
```

The RFC names a JavaScript module mounted as `<Viewer fileId locator onSelect />`.
gact-tui chose the iframe (RFC §12, decision 2), so the entry is an HTML page
instead, and the page's protocol carries `fileId`, `locator` and the
`select` message that stands in for `onSelect`.

## conventions/

One convention per file, in the RFC §9 entry format. Each entry was checked
against a file written by the convention's own library; HDF-EOS5, CGNS, XDMF,
MATLAB v7.3 and Keras are left out until one has been.

Fields from the §9 sketch: `id`, `container`, `version_hint`, `detect`,
`locate_primary` (`follow_attr`, `signal`, `axes`, `units`), `schema`,
`render_hint_from`.

The sketch could not express every verified convention, so the entries use a
few additions. They are proposals for the format, not settled fields:

| Addition | Why |
|---|---|
| `equals`, `contains` on a `detect` item | NWB and AnnData are identified by an attribute's value, not its presence (`neurodata_type: NWBFile`). |
| `any:` in `detect` | Items in `detect` must all match. netCDF-4 files older than netCDF-C 4.4 lack `_NCProperties` and need an alternative. |
| `extends` | CF sits on netCDF-4 and pandas on PyTables, so one file matches two entries. |
| `find` in `locate_primary` | Moves to the first object (breadth-first) carrying an attribute. NeXus `default` attributes are optional, and nexusformat does not write the root one. |
| `path` in `locate_primary` | A fixed location (`/X` in AnnData). |
| `datasets_with_attr` | netCDF variables are every dataset carrying `DIMENSION_LIST`, not one path. |
| `schema.source: in_file:<path>` | NWB embeds its schema in the file. |

Attribute references follow the sketch: `/@name` is a root attribute,
`@name` an attribute of the current object, and `/path@name` an attribute of
a fixed object. A bare `attr: name` in `detect` matches the root or any
object directly under it, which keeps detection to one level and no data
reads. Values starting with `@` are quoted, because YAML reserves `@` at the
start of a plain scalar; the sketch's unquoted `follow_attr: @default` does
not parse.
