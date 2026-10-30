# HDF5 Map Objects

## Purpose

The map API (**H5M**), introduced with the VOL work in HDF5 1.12 and unchanged in 1.14 and 2.x, provides **map objects** — a native key-value store alongside groups and datasets. Maps store typed key→value pairs with no predefined schema, purpose-built for large flat lookups where group-based key-value simulation is too slow.

**Related**: `references/vol_usage.md` (maps require a VOL connector implementing the map API). Consult the HDF5 VOL developer documentation when implementing a map-capable connector.

## What Problem Maps Solve

Groups can simulate key-value storage, but they are optimized for hierarchical navigation, not random lookup. Storing millions of key-value pairs as group members creates metadata B-tree overhead and poor per-key access performance. Map objects provide:
- Large, flat, unordered key-value collections
- Dynamic key sets (insert/delete at runtime)
- Typed keys and values (any HDF5 datatype)
- Native support in connectors built on KV storage backends (e.g., DAOS)

## Critical Architecture Constraint

**Map objects are defined at the API level but not implemented in the native HDF5 file format.** The native VOL connector does not support map objects. Without a map-capable VOL connector, all `H5M*` calls fail immediately.

Current connectors with map support:
- **DAOS VOL** (HDF Group / Intel): maps natively to DAOS distributed KV store
- **Pass-through VOL**: can wrap map-capable connectors

There is no fallback to native HDF5 storage. If your environment does not have a map-capable VOL connector, map objects are not available.

## C API Overview

| Function | Purpose |
|---|---|
| `H5Mcreate(loc, name, key_type, val_type, lcpl, mcpl, mapl)` | Create map, declare key/value types |
| `H5Mopen(loc, name, mapl)` | Open existing map by name |
| `H5Mput(map, key_type, key, val_type, val, dxpl)` | Insert or overwrite a key-value pair |
| `H5Mget(map, key_type, key, val_type, val, dxpl)` | Retrieve value by key |
| `H5Mexists(map, key_type, key, exists, dxpl)` | Test for key presence (`exists` is `bool *` since HDF5 2.0, previously `hbool_t *`) |
| `H5Mdelete(map, key_type, key, dxpl)` | Remove a key |
| `H5Miterate(map, idx, key_type, op, op_data, dxpl)` | Iterate over all keys |
| `H5Mget_count(map, count, dxpl)` | Number of entries |
| `H5Mclose(map)` | Release map handle |

h5py does not wrap H5M at any level. Use C/C++ against a map-capable VOL connector.

## Key and Value Datatypes

- Any fixed-size HDF5 type can be a key: integers, floats, fixed-length strings, compound types
- Variable-length strings as keys are supported if the VOL connector handles VL heap allocation
- Values accept any HDF5 type including VL, compound, and array types
- Key type and value type are **fixed at map creation** — schema cannot change after creation
- Prefer compact key types (integers, fixed-length strings) for best lookup performance in backend storage

## Property Lists

Map objects use dedicated property lists:
- `H5Pcreate(H5P_MAP_CREATE)` — map creation properties (currently VOL-connector-defined)
- `H5Pcreate(H5P_MAP_ACCESS)` — map access properties

Standard HDF5 property list lifecycle applies: create, set properties, pass to `H5Mcreate`/`H5Mopen`, close.

## Do / Do Not

**Do**:
- Verify your VOL connector implements `H5VL_MAP_*` operations before designing a map-based workflow
- Use typed integer or fixed-length string keys for predictable performance
- Call `H5Mexists` before `H5Mget` to avoid errors on missing keys
- Close maps with `H5Mclose` — same resource discipline as datasets and groups

**Do not**:
- Assume map objects work with the native HDF5 VOL — they do not
- Use maps for small collections (<100 entries) — group + attributes is simpler and universally portable
- Expect h5dump, HDFView, or h5py high-level API to read map objects — support is VOL-connector-specific
- Mix groups and maps to model the same concept within one file — choose one pattern
- Assume portability to standard `.h5` files — a DAOS-backed map file is not readable as native HDF5

## Portability

Map objects are **not portable** to the native HDF5 file format. A file written via DAOS VOL with map objects cannot be opened as a standard HDF5 file. This is a fundamental design constraint, not a tooling limitation.

If portability to h5py, h5dump, or HDFView is required, use groups or compound datasets instead.

## Maturity and Roadmap

Map objects have been stable at the API level since 1.12 and are unchanged in 1.14 and 2.x. Real-world usage is confined to DAOS-backed environments. As of HDF5 2.2.0 the native connector still has no map support (the 2.0 through 2.2 changelogs contain no H5M entries). Monitor HDF Group release notes before adopting maps in a new architecture.

## When to Use Map Objects

- DAOS-backed HDF5 workflows requiring native KV semantics
- Very large flat key-value stores (millions of entries) where group B-tree overhead is measurable and profiled
- Typed metadata registries embedded in workflows already committed to a map-capable VOL

## When NOT to Use

- Any workflow requiring standard HDF5 file portability
- Environments without DAOS or another map-capable VOL connector
- Small key-value collections — use group attributes
- h5py-only Python workflows (no high-level API)
- Situations where `h5dump` or standard tooling must be able to inspect the file

## Citations and References

- [HDF Group: Map Objects API (H5M)](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_m.html)
- [HDF Group: New Features in HDF5 1.12/1.14](https://docs.hdfgroup.org/hdf5/v1_14/index.html)
- [DAOS VOL Connector](https://github.com/HDFGroup/vol-daos)
- [HDF Group: VOL Connector Architecture](https://support.hdfgroup.org/documentation/hdf5/latest/group___f_a_p_l.html)
