---
title: "Special types — h5py 3.13.0 documentation"
source: "https://docs.h5py.org/en/stable/special.html"
author:
published:
created: 2025-02-23
description:
tags:
  - "clippings"
---
HDF5 supports a few types which have no direct NumPy equivalent. Among the most useful and widely used are *variable-length* (VL) types, and enumerated types. As of version 2.3, h5py fully supports HDF5 enums and VL types.

## How special types are represented

Since there is no direct NumPy dtype for variable-length strings, enums or references, h5py extends the dtype system slightly to let HDF5 know how to store these types. Each type is represented by a native NumPy dtype, with a small amount of metadata attached. NumPy routines ignore the metadata, but h5py can use it to determine how to store the data.

The metadata h5py attaches to dtypes is not part of the public API, so it may change between versions. Use the functions described below to create and check for these types.

## Variable-length strings

In HDF5, data in VL format is stored as arbitrary-length vectors of a base type. In particular, strings are stored C-style in null-terminated buffers. NumPy has no native mechanism to support this. Unfortunately, this is the de facto standard for representing strings in the HDF5 C API, and in many HDF5 applications.

Thankfully, NumPy has a generic pointer type in the form of the “object” (“O”) dtype. In h5py, variable-length strings are mapped to object arrays. A small amount of metadata attached to an “O” dtype tells h5py that its contents should be converted to VL strings when stored in the file.

Existing VL strings can be read and written to with no additional effort; Python strings and fixed-length NumPy strings can be auto-converted to VL data and stored.

Here’s an example showing how to create a VL array of strings:

```default
>>> f = h5py.File('foo.hdf5')
>>> dt = h5py.string_dtype(encoding='utf-8')
>>> ds = f.create_dataset('VLDS', (100,100), dtype=dt)
>>> ds.dtype.kind
'O'
>>> h5py.check_string_dtype(ds.dtype)
string_info(encoding='utf-8', length=None)
```

h5py.string\_dtype(*encoding\='utf-8'*, *length\=None*)[](https://docs.h5py.org/en/stable/#h5py.string_dtype "Link to this definition")

Make a numpy dtype for HDF5 strings

Parameters:

- **encoding** – `'utf-8'` or `'ascii'`.
- **length** – `None` for variable-length, or an integer for fixed-length string data, giving the length in bytes.

h5py.check\_string\_dtype(*dt*)[](https://docs.h5py.org/en/stable/#h5py.check_string_dtype "Link to this definition")

Check if `dt` is a string dtype. Returns a *string\_info* object if it is, or `None` if not.

*class* h5py.string\_info[](https://docs.h5py.org/en/stable/#h5py.string_info "Link to this definition")

A named tuple type holding string encoding and length.

encoding[](https://docs.h5py.org/en/stable/#h5py.string_info.encoding "Link to this definition")

The character encoding associated with the string dtype, which can be `'utf-8'` or `'ascii'`.

length[](https://docs.h5py.org/en/stable/#h5py.string_info.length "Link to this definition")

For fixed-length string dtypes, the length in bytes. `None` for variable-length strings.

## Arbitrary vlen data

Starting with h5py 2.3, variable-length types are not restricted to strings. For example, you can create a “ragged” array of integers:

```default
>>> dt = h5py.vlen_dtype(np.dtype('int32'))
>>> dset = f.create_dataset('vlen_int', (100,), dtype=dt)
>>> dset[0] = [1,2,3]
>>> dset[1] = [1,2,3,4,5]
```

Single elements are read as NumPy arrays:

```default
>>> dset[0]
array([1, 2, 3], dtype=int32)
```

Multidimensional selections produce an object array whose members are integer arrays:

```default
>>> dset[0:2]
array([array([1, 2, 3], dtype=int32), array([1, 2, 3, 4, 5], dtype=int32)], dtype=object)
```

Note

NumPy doesn’t support ragged arrays, and the ‘arrays of arrays’ h5py uses as a workaround are not as convenient or efficient as regular NumPy arrays. If you’re deciding how to store data, consider whether there’s a sensible way to do it without a variable-length type.

h5py.vlen\_dtype(*basetype*)[](https://docs.h5py.org/en/stable/#h5py.vlen_dtype "Link to this definition")

Make a numpy dtype for an HDF5 variable-length datatype.

Parameters:

**basetype** – The dtype of each element in the array.

h5py.check\_vlen\_dtype(*dt*)[](https://docs.h5py.org/en/stable/#h5py.check_vlen_dtype "Link to this definition")

Check if `dt` is a variable-length dtype. Returns the base type if it is, or `None` if not.

## Enumerated types

HDF5 has the concept of an *enumerated type*, which is an integer datatype with a restriction to certain named values. Since NumPy has no such datatype, HDF5 ENUM types are read and written as integers.

Here’s an example of creating an enumerated type:

```default
>>> dt = h5py.enum_dtype({"RED": 0, "GREEN": 1, "BLUE": 42}, basetype='i')
>>> h5py.check_enum_dtype(dt)
{'BLUE': 42, 'GREEN': 1, 'RED': 0}
>>> f = h5py.File('foo.hdf5','w')
>>> ds = f.create_dataset("EnumDS", (100,100), dtype=dt)
>>> ds.dtype.kind
'i'
>>> ds[0,:] = 42
>>> ds[0,0]
42
>>> ds[1,0]
0
```

h5py.enum\_dtype(*values\_dict*, *basetype\=np.uint8*)[](https://docs.h5py.org/en/stable/#h5py.enum_dtype "Link to this definition")

Create a NumPy representation of an HDF5 enumerated type

Parameters:

- **values\_dict** – Mapping of string names to integer values.
- **basetype** – An appropriate integer base dtype large enough to hold the possible options.

h5py.check\_enum\_dtype(*dt*)[](https://docs.h5py.org/en/stable/#h5py.check_enum_dtype "Link to this definition")

Check if `dt` represents an enumerated type. Returns the values dict if it is, or `None` if not.

## Object and region references

References have their [own section](https://docs.h5py.org/en/stable/refs.html#refs).

## Storing other types as opaque data

New in version 3.0.

Numpy datetime64 and timedelta64 dtypes have no equivalent in HDF5 (the HDF5 time type is broken and deprecated). h5py allows you to store such data with an HDF5 opaque type; it can be read back correctly by h5py, but won’t be interoperable with other tools.

Here’s an example of storing and reading a datetime array:

```default
>>> arr = np.array([np.datetime64('2019-09-22T17:38:30')])
>>> f['data'] = arr.astype(h5py.opaque_dtype(arr.dtype))
>>> print(f['data'][:])
['2019-09-22T17:38:30']
```

h5py.opaque\_dtype(*dt*)[](https://docs.h5py.org/en/stable/#h5py.opaque_dtype "Link to this definition")

Return a dtype like the input, tagged to be stored as HDF5 opaque type.

h5py.check\_opaque\_dtype(*dt*)[](https://docs.h5py.org/en/stable/#h5py.check_opaque_dtype "Link to this definition")

Return True if the dtype given is tagged to be stored as HDF5 opaque data.

Note

With some exceptions, you can use [`opaque_dtype()`](https://docs.h5py.org/en/stable/#h5py.opaque_dtype "h5py.opaque_dtype") with any numpy dtype. While this may seem like a convenient way to get arbitrary data into HDF5, remember that it’s not a standard format. It’s better to fit your data into HDF5’s native structures, or use a file format better suited to your data.

## Older API

Before h5py 2.10, a single pair of functions was used to create and check for all of these special dtypes. These are still available for backwards compatibility, but are deprecated in favour of the functions listed above.

h5py.special\_dtype(*\*\*kwds*)[](https://docs.h5py.org/en/stable/#h5py.special_dtype "Link to this definition")

Create a NumPy dtype object containing type hints. Only one keyword may be specified.

Parameters:

- **vlen** – Base type for HDF5 variable-length datatype.
- **enum** – 2-tuple `(basetype, values_dict)`. `basetype` must be an integer dtype; `values_dict` is a dictionary mapping string names to integer values.
- **ref** – Provide class `h5py.Reference` or `h5py.RegionReference` to create a type representing object or region references respectively.

h5py.check\_dtype(*\*\*kwds*)[](https://docs.h5py.org/en/stable/#h5py.check_dtype "Link to this definition")

Determine if the given dtype object is a special type. Example:

```default
>>> out = h5py.check_dtype(vlen=mydtype)
>>> if out is not None:
...     print("Vlen of type %s" % out)
str
```

Parameters:

- **vlen** – Check for an HDF5 variable-length type; returns base class
- **enum** – Check for an enumerated type; returns 2-tuple `(basetype, values_dict)`.
- **ref** – Check for an HDF5 object or region reference; returns either `h5py.Reference` or `h5py.RegionReference`.