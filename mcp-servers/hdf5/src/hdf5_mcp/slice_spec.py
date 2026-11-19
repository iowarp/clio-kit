"""Parse a NumPy-style slice specification.

`hdf5_batch_read` takes a slice as text, such as ``"0:10, :, 5"``. This parser
accepts the selection syntax the tool documents and refuses everything else.
"""

from types import EllipsisType
from typing import Tuple, Union

SliceItem = Union[int, slice, EllipsisType]


def _parse_int(text: str, spec: str) -> int:
    """One integer bound, allowing a leading sign and surrounding spaces."""
    value = text.strip()
    body = value[1:] if value[:1] in "+-" else value
    if not body.isdigit():
        raise ValueError(f"Invalid slice specification: {spec!r}")
    return int(value)


def _parse_component(text: str, spec: str) -> SliceItem:
    """One comma-separated component: an index, a slice or ``...``."""
    item = text.strip()
    if item == "...":
        return Ellipsis
    if ":" not in item:
        return _parse_int(item, spec)
    parts = item.split(":")
    if len(parts) > 3:
        raise ValueError(f"Invalid slice specification: {spec!r}")
    bounds = [None if not p.strip() else _parse_int(p, spec) for p in parts]
    bounds += [None] * (3 - len(bounds))
    if bounds[2] == 0:
        raise ValueError(f"Slice step cannot be zero: {spec!r}")
    return slice(*bounds)


def parse_slice_spec(spec: str) -> Tuple[SliceItem, ...]:
    """Turn ``"0:10, :, 5, ..."`` into ``(slice(0, 10), slice(None), 5, ...)``.

    Accepted per comma-separated component: an integer (negative allowed),
    ``start:stop`` or ``start:stop:step`` with any part omitted, and ``...``
    at most once. Surrounding square brackets are allowed.

    Raises:
        ValueError: if the text is anything else.
    """
    text = spec.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text.strip():
        raise ValueError(f"Invalid slice specification: {spec!r}")
    items = tuple(_parse_component(part, spec) for part in text.split(","))
    if sum(1 for item in items if item is Ellipsis) > 1:
        raise ValueError(f"Only one '...' is allowed: {spec!r}")
    return items
