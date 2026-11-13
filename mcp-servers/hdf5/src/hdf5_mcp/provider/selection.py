"""Selection parsing and sizing shared by the provider and the data route."""

from __future__ import annotations

import math

import numpy as np
from h5grove.utils import parse_slice

Index = tuple[slice | int, ...]

# Variable-length elements have no fixed size; this is the per-element guess
# used when sizing a selection against a byte cap.
VLEN_ELEMENT_BYTES = 64


def parse_selection(selection: str | None, ndim: int) -> Index:
    """Parse an h5grove selection string (``"0:100"``, ``"5, ::2"``).

    Raises:
        ValueError: The string is malformed or has more members than `ndim`.
    """
    if selection is None or selection.strip() == "":
        return ()
    try:
        # h5grove's parser chokes on whitespace before an empty slice
        # member (" ::2"), so strip it first.
        index = parse_slice("".join(selection.split()))
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid selection {selection!r}: {e}") from e
    if len(index) > ndim:
        raise ValueError(
            f"Selection {selection!r} has {len(index)} members for a {ndim}-D dataset"
        )
    return index


def kept_axes(shape: tuple[int, ...], index: Index) -> list[tuple[int, int, int, int]]:
    """``(axis, start, step, count)`` for each axis a slice keeps in the result.

    Axes the index leaves out are kept whole; integer members drop their axis.
    """
    full = list(index) + [slice(None)] * (len(shape) - len(index))
    kept = []
    for axis, member in enumerate(full):
        if isinstance(member, slice):
            start, stop, step = member.indices(shape[axis])
            kept.append((axis, start, step, len(range(start, stop, step))))
    return kept


def element_bytes(dtype: np.dtype) -> int:
    return VLEN_ELEMENT_BYTES if dtype.kind == "O" else dtype.itemsize


def selection_bytes(shape: tuple[int, ...], dtype: np.dtype, index: Index) -> int:
    """Bytes the selection would read, using the vlen guess for object dtypes."""
    counts = [count for _, _, _, count in kept_axes(shape, index)]
    return math.prod(counts) * element_bytes(dtype)
