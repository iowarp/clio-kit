# HDF5 Compound & Complex Datatypes

## Purpose

HDF5 supports a rich type system beyond numeric arrays: compound types (struct-like records), variable-length strings, enumerations, opaque, and array types. This guide covers their correct use, numpy mapping, and common failure modes.

**Related**: `references/filters.md` for compression (compound types are filterable; VL heap data does not shrink under filters), `references/io.md` for write performance.

## Type System Overview

| HDF5 Type | Numpy Analog | h5py Creation |
|---|---|---|
| `H5T_COMPOUND` | structured array / `np.dtype([...])` | Pass structured dtype to `create_dataset` |
| `H5T_STRING` fixed | `'S10'` bytes | `h5py.string_dtype(length=N)` or `'SN'` |
| `H5T_STRING` variable | Python str (read back as `bytes` in h5py >= 3, see below) | `h5py.string_dtype()` (length=None) |
| `H5T_ENUM` | int + lookup dict | `h5py.enum_dtype(dict, basetype)` |
| `H5T_OPAQUE` | raw bytes | `h5py.opaque_dtype(np.dtype('V16'))` |
| `H5T_ARRAY` | fixed sub-array per element | `np.dtype((np.float64, (3, 3)))` (NumPy sub-array dtype) |
| `H5T_VLEN` | ragged sequence | `h5py.vlen_dtype(basetype)` |

## Compound Types

Compound types store structured records — C structs or numpy structured arrays. Each field has a name, HDF5 type, and byte offset.

**In h5py**: Pass a numpy structured dtype directly to `create_dataset`. h5py maps it to `H5T_COMPOUND` automatically.

**Key rules**:
- Field names are case-sensitive and unique
- Nested compound types (struct-of-structs) are supported
- Numpy may add alignment padding that does not match the HDF5 field offsets — use `{'names': [...], 'formats': [...], 'offsets': [...], 'itemsize': N}` when round-tripping with C structs
- Reading returns a numpy structured array; individual fields accessed as `data['fieldname']`
- **Partial I/O**: HDF5 supports reading only a subset of compound fields (memory type differs from file type). In h5py read a subset with `dset['field']` or `dset.fields(['a','b'])[...]`; drop to the `h5py.h5t` low-level API only for custom memory types

## String Types

### Fixed-Length Strings
- Null-padded or null-terminated byte arrays — portable, predictable storage size
- Use `h5py.string_dtype(length=N)` or `dtype='SN'`
- Always read back as `bytes` in Python 3 — decode explicitly

### Variable-Length Strings
- Each element is a heap pointer — convenient, but the heap payload does not compress (filters see only the fixed-size descriptors, not the heap data)
- Use `h5py.string_dtype()` for variable length
- Prefer `h5py.string_dtype(encoding='utf-8')` for international text
- Dataset elements read back as `bytes` in h5py ≥ 3 (as `str` in h5py 2.x) — code that auto-decodes one breaks on the other. In h5py ≥ 3 call `dset.asstr()[...]` to get `str`, or `dset.astype('T')` (NumPy 2, h5py ≥ 3.14) for NumPy's native variable-width strings
- **SWMR restriction**: VL string datasets cannot be modified in SWMR mode

**Recommendation**: Fixed-length strings for performance-critical or archival data; VL strings only for human-readable metadata where length is genuinely unpredictable.

## Enumeration Types

Maps named labels to an integer base type — useful for categorical data (sensor states, status codes).

`h5py.enum_dtype({'OFF': 0, 'ON': 1, 'FAULT': 2}, basetype=np.uint8)`

- Storage is the base integer type — compact and filterable
- h5py reads enum fields back as plain integers; retrieve the enum dict via `h5py.check_enum_dtype(dset.dtype)`
- For portability to non-h5py readers, store the enum mapping in attributes alongside the dataset

## Opaque Types

Raw byte sequences with no HDF5-interpreted structure. Fixed size per element.

`h5py.opaque_dtype(np.dtype('V16'))`

- Can be filtered (HDF5 filters are type-agnostic), but compression ratios are poor without type knowledge
- Loses interoperability with any tool other than your own code
- **Preferred alternative**: store as a fixed-length bytes dataset with a `format` attribute documenting the encoding

## Array Types (H5T_ARRAY)

Fixed-shape sub-arrays embedded in each element. A dataset of shape `(N,)` with type `H5T_ARRAY[3][3]` of float reads as shape `(N, 3, 3)`.

`np.dtype((np.float64, (3, 3)))`

- More compact than a compound type with 9 float fields for the same data
- Chunking applies to outer dataset dimensions only — cannot filter on sub-array dimensions

## Variable-Length Sequences (H5T_VLEN)

Each element is a variable-length sequence of a base type stored on the heap.

`h5py.vlen_dtype(np.int32)`

- Reads as a numpy object array (ragged) — no vectorized operations
- **Heap data does not compress** — filters apply only to the fixed-size VL descriptors, not the heap payload (the dataset can still carry a filter pipeline)
- High per-element overhead; avoid for large uniform arrays
- Use only when element lengths are genuinely irregular and total count is small
- **C API**: free VL buffers with `H5Treclaim()` (`H5Dvlen_reclaim()` is the deprecated older spelling); h5py handles this automatically

## New in HDF5 2.x

- **Complex numbers**: HDF5 2.0 adds a native `H5T_COMPLEX` class (C99 complex). Creating such an object needs the library version bounds "high" setting to be `H5F_LIBVER_V200` or later, and older libraries cannot read it. For compatibility with older readers use an existing convention such as nc-complex (compound). h5py 3.16 can store NumPy complex numbers natively (needs HDF5 2.0+) but compound remains its default.
- **Reduced-precision floats**: predefined types `H5T_FLOAT_BFLOAT16LE/BE` and FP8 (`H5T_FLOAT_F8E4M3`, `H5T_FLOAT_F8E5M2`) arrived in 2.0; FP6 (`H5T_FLOAT_F6E2M3`, `H5T_FLOAT_F6E3M2`) and FP4 (`H5T_FLOAT_F4E2M1`) in 2.1. Conversions are emulated in software, and implicit conversion of the FP8/FP6/FP4 types may produce wrong values (a known problem in 2.2.0): use a memory type that matches the file type and convert outside HDF5, or verify values.
- **Strings**: h5py 3.14 can read and write NumPy 2 `StringDType` (`dtype='T'`).
- **Filters and VL data**: since HDF5 2.1 the library invokes a filter's `can_apply` and `set_local` callbacks for datasets with variable-length datatypes (previously skipped).

## Do / Do Not

**Do**:
- Match field offsets explicitly when round-tripping compound types to/from C structs
- Set string encoding explicitly (`utf-8`) for any file shared outside your team
- Store enum dictionaries in attributes for portability to non-h5py readers
- Use fixed-length strings inside compound types — VL strings in compounds are legal but complex

**Do not**:
- Use VL strings or VL sequences in performance-critical datasets — heap allocation is slow and kills compression
- Expect VL types to compress — the heap payload will not shrink, regardless of filter settings
- Use opaque types as a catch-all for hard-to-model data — the interoperability cost is rarely worth it
- Assume h5py 2.x and 3.x decode strings the same way (3.x returns `bytes` for VL string datasets) — test explicitly

## Common Pitfalls

**Alignment mismatch**: `np.dtype` with `align=True` may produce different itemsize/offsets than the HDF5 compound. Verify with `dtype.itemsize` vs. the HDF5 type's `get_size()`.

**Enum round-trip**: h5py does not auto-convert integer reads to named enum values. Retrieve and apply the enum dict manually.

**VL memory leak in C**: C code reading VL data must call `H5Treclaim()` (formerly `H5Dvlen_reclaim()`). Forgetting this leaks heap memory silently.

**Partial compound read**: In h5py use `dset['field']` or `dset.fields([...])` to read a subset of fields; a custom memory datatype via the low-level API is needed only for unusual layouts.

## Citations and References

- [HDF Group: HDF5 Datatypes](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_t.html)
- [h5py: Special Datatypes](https://docs.h5py.org/en/stable/special.html)
- [HDF Group: Strings in HDF5](https://support.hdfgroup.org/documentation/hdf5-docs/advanced_topics/UsingUnicode.html)
- [HDF Group: Compound Datatypes Tutorial](https://support.hdfgroup.org/documentation/hdf5/latest/_learn_basics_compound.html)
