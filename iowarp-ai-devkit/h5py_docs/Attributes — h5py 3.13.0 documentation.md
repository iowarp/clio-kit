---
title: "Attributes — h5py 3.13.0 documentation"
source: "https://docs.h5py.org/en/stable/high/attr.html"
author:
published:
created: 2025-02-23
description:
tags:
  - "clippings"
---
Attributes are a critical part of what makes HDF5 a “self-describing” format. They are small named pieces of data attached directly to [`Group`](https://docs.h5py.org/en/stable/high/group.html#h5py.Group "h5py.Group") and [`Dataset`](https://docs.h5py.org/en/stable/high/dataset.html#h5py.Dataset "h5py.Dataset") objects. This is the official way to store metadata in HDF5.

Each Group or Dataset has a small proxy object attached to it, at `<obj>.attrs`. Attributes have the following properties:

- They may be created from any scalar or NumPy array
- Each attribute should be small (generally < 64k)
- There is no partial I/O (i.e. slicing); the entire attribute must be read.

The `.attrs` proxy objects are of class [`AttributeManager`](https://docs.h5py.org/en/stable/high/#h5py.AttributeManager "h5py.AttributeManager"), below. This class supports a dictionary-style interface.

By default, attributes are iterated in alphanumeric order. However, if group or dataset is created with `track_order=True`, the attribute insertion order is remembered (tracked) in HDF5 file, and iteration uses that order. The latter is consistent with Python 3.7+ dictionaries.

The default `track_order` for all new groups and datasets can be specified globally with `h5.get_config().track_order`.

## Large attributes

HDF5 allows attributes to be larger than 64 KiB, but these need to be stored in a different way. As of March 2024, the way HDF5 documentation suggests you configure this does not work. Instead, enable order tracking when creating the object you want to attach attributes to:

```default
grp = f.create_group('g', track_order=True)
grp.attrs['large'] = np.arange(1_000_000, dtype=np.uint32)
```

## Reference

*class* h5py.AttributeManager(*parent*)[](https://docs.h5py.org/en/stable/high/#h5py.AttributeManager "Link to this definition")

AttributeManager objects are created directly by h5py. You should access instances by `group.attrs` or `dataset.attrs`, not by manually creating them.

\_\_iter\_\_()[](https://docs.h5py.org/en/stable/high/#h5py.AttributeManager.__iter__ "Link to this definition")

Get an iterator over attribute names.

\_\_contains\_\_(*name*)[](https://docs.h5py.org/en/stable/high/#h5py.AttributeManager.__contains__ "Link to this definition")

Determine if attribute name is attached to this object.

\_\_getitem\_\_(*name*)[](https://docs.h5py.org/en/stable/high/#h5py.AttributeManager.__getitem__ "Link to this definition")

Retrieve an attribute.

\_\_setitem\_\_(*name*, *val*)[](https://docs.h5py.org/en/stable/high/#h5py.AttributeManager.__setitem__ "Link to this definition")

Create an attribute, overwriting any existing attribute. The type and shape of the attribute are determined automatically by h5py.

\_\_delitem\_\_(*name*)[](https://docs.h5py.org/en/stable/high/#h5py.AttributeManager.__delitem__ "Link to this definition")

Delete an attribute. KeyError if it doesn’t exist.

keys()[](https://docs.h5py.org/en/stable/high/#h5py.AttributeManager.keys "Link to this definition")

Get the names of all attributes attached to this object.

Returns:

set-like object.

values()[](https://docs.h5py.org/en/stable/high/#h5py.AttributeManager.values "Link to this definition")

Get the values of all attributes attached to this object.

Returns:

collection or bag-like object.

items()[](https://docs.h5py.org/en/stable/high/#h5py.AttributeManager.items "Link to this definition")

Get `(name, value)` tuples for all attributes attached to this object.

Returns:

collection or set-like object.

get(*name*, *default\=None*)[](https://docs.h5py.org/en/stable/high/#h5py.AttributeManager.get "Link to this definition")

Retrieve name, or default if no such attribute exists.

get\_id(*name*)[](https://docs.h5py.org/en/stable/high/#h5py.AttributeManager.get_id "Link to this definition")

Get the low-level [`AttrID`](https://api.h5py.org/h5a.html#h5py.h5a.AttrID "(in Low-level API for h5py v3.13)") for the named attribute.

create(*name*, *data*, *shape\=None*, *dtype\=None*)[](https://docs.h5py.org/en/stable/high/#h5py.AttributeManager.create "Link to this definition")

Create a new attribute, with control over the shape and type. Any existing attribute will be overwritten.

Parameters:

- **name** (*String*) – Name of the new attribute
- **data** – Value of the attribute; will be put through `numpy.array(data)`.
- **shape** (*Tuple*) – Shape of the attribute. Overrides `data.shape` if both are given, in which case the total number of points must be unchanged.
- **dtype** (*NumPy dtype*) – Data type for the attribute. Overrides `data.dtype` if both are given.

modify(*name*, *value*)[](https://docs.h5py.org/en/stable/high/#h5py.AttributeManager.modify "Link to this definition")

Change the value of an attribute while preserving its type and shape. Unlike [`AttributeManager.__setitem__()`](https://docs.h5py.org/en/stable/high/#h5py.AttributeManager.__setitem__ "h5py.AttributeManager.__setitem__"), if the attribute already exists, only its value will be changed. This can be useful for interacting with externally generated files, where the type and shape must not be altered.

If the attribute doesn’t exist, it will be created with a default shape and type.

Parameters:

- **name** (*String*) – Name of attribute to modify.
- **value** – New value. Will be put through `numpy.array(value)`.