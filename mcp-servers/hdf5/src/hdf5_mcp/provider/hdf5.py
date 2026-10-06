"""HDF5 structured-file provider.

Implements the CLIO `StructuredFileProvider` protocol for HDF5 on top of
h5grove, so the agent-facing provider and the h5web viewer's data route resolve
links, parse selections and compute statistics the same way.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from h5grove.models import LinkResolution
from h5grove.utils import (
    NotFoundError,
    get_entity_from_file,
    get_filters,
    hdf_path_join,
    parse_slice,
)

from .protocol import Description, ErrorCode, Node, NodeKind, Payload, ProviderError

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"

# Variable-length elements have no fixed size; this is the per-element guess
# used when sizing a read against the payload cap.
VLEN_ELEMENT_BYTES = 64

_LAYOUTS = {
    h5py.h5d.COMPACT: "compact",
    h5py.h5d.CONTIGUOUS: "contiguous",
    h5py.h5d.CHUNKED: "chunked",
    h5py.h5d.VIRTUAL: "virtual",
}


class HDF5Provider:
    """Browse, describe and read HDF5 files for CLIO.

    Stateless: every call opens the file and closes it before returning.

    Args:
        preview_max_bytes: Cap on the JSON-serialized `describe()` preview.
        read_max_bytes: Cap on the data `read()` returns. Larger selections are
            trimmed and marked `truncated`.
        stats_max_bytes: Most dataset bytes `describe()` reads to compute
            min/mean/max. Larger datasets are summarized from a leading slab
            and the preview is marked `sampled`.
        preview_edge: Values shown from each end of a 1-D dataset.
        table_preview_rows: Rows shown from a compound (table) dataset.
        text_preview_chars: Characters shown from a string.
        attribute_max_elements: Attributes with more elements are reported by
            shape and dtype only.
    """

    id = "hdf5"
    media_types = frozenset({"application/x-hdf5"})

    def __init__(
        self,
        *,
        preview_max_bytes: int = 16_384,
        read_max_bytes: int = 16 * 1024 * 1024,
        stats_max_bytes: int = 64 * 1024 * 1024,
        preview_edge: int = 5,
        table_preview_rows: int = 5,
        text_preview_chars: int = 200,
        attribute_max_elements: int = 64,
    ):
        self.preview_max_bytes = preview_max_bytes
        self.read_max_bytes = read_max_bytes
        self.stats_max_bytes = stats_max_bytes
        self.preview_edge = preview_edge
        self.table_preview_rows = table_preview_rows
        self.text_preview_chars = text_preview_chars
        self.attribute_max_elements = attribute_max_elements

    # -- protocol ---------------------------------------------------------

    def detect(self, path: Path) -> bool:
        """True when the file carries the HDF5 signature.

        The superblock sits at offset 0, 512, 1024, 2048, ... (a user block
        shifts it), so each candidate offset is checked. The extension is
        never consulted.
        """
        try:
            with open(path, "rb") as f:
                size = os.fstat(f.fileno()).st_size
                offset = 0
                while offset + len(HDF5_SIGNATURE) <= size:
                    f.seek(offset)
                    if f.read(len(HDF5_SIGNATURE)) == HDF5_SIGNATURE:
                        return True
                    offset = 512 if offset == 0 else offset * 2
        except OSError:
            return False
        return False

    def children(self, path: Path, locator: str) -> list[Node]:
        """The direct children of a group. Reads metadata only, never data."""
        locator = self.canonicalize(locator)
        with self._open(path, locator) as f:
            entity = self._resolve(f, locator)
            if not isinstance(entity, h5py.Group):
                raise ProviderError("unsupported", f"{locator} is not a group", locator)
            nodes = []
            for name in entity.keys():
                # Resolve from the opened file, not entity.file: a group reached
                # through an external link lives in another file.
                child_locator = hdf_path_join(locator, name)
                child = self._resolve(f, child_locator)
                nodes.append(self._node(child, child_locator))
            return nodes

    def describe(self, path: Path, locator: str) -> Description:
        """Metadata, attributes, storage and a bounded preview of one object."""
        locator = self.canonicalize(locator)
        with self._open(path, locator) as f:
            entity = self._resolve(f, locator)
            node = self._node(entity, locator)
            attributes: dict[str, Any] = {}
            storage: dict[str, Any] = {}
            preview: Any = None

            if isinstance(entity, (h5py.Group, h5py.Dataset, h5py.Datatype)):
                attributes = self._attributes(entity)
            if isinstance(entity, h5py.Dataset):
                storage = self._storage(entity)
                preview = self._fit_preview(self._preview(entity))
            elif isinstance(entity, h5py.Group):
                preview = {"children": len(entity)}
            elif isinstance(entity, h5py.SoftLink):
                preview = {"target": entity.path, "resolved": False}
            elif isinstance(entity, h5py.ExternalLink):
                preview = {
                    "file": entity.filename,
                    "target": entity.path,
                    "resolved": False,
                }

            return {
                "node": node,
                "attributes": attributes,
                "preview": preview,
                "storage": storage,
                "render_hint": _render_hint(entity, node["kind"]),
                "convention": None,
            }

    def read(self, path: Path, locator: str, selection: str | None) -> Payload:
        """Read a dataset, or the part of it `selection` names.

        `selection` uses h5grove's NumPy-style grammar, for example
        ``"0:100"``, ``"5, ::2"`` or ``"1, 2:5, 0"``. A selection larger than
        `read_max_bytes` is trimmed along its leading axes and the payload is
        marked `truncated`.
        """
        locator = self.canonicalize(locator)
        with self._open(path, locator) as f:
            dataset = self._resolve(f, locator)
            if not isinstance(dataset, h5py.Dataset):
                raise ProviderError(
                    "unsupported", f"{locator} is not a dataset", locator
                )
            dtype = _dtype_label(dataset.dtype)
            if dataset.shape is None:
                return {"dtype": dtype, "shape": (), "data": None, "truncated": False}

            index = self._parse_selection(dataset, selection, locator)
            index, truncated = self._cap_selection(dataset, index, locator)
            try:
                data = _string_view(dataset)[index]
            except (ValueError, TypeError, IndexError) as e:
                raise ProviderError("unsupported", str(e), locator) from e

            if isinstance(data, (str, bytes)) and len(data) > self.read_max_bytes:
                data = data[: self.read_max_bytes]
                truncated = True
            return {
                "dtype": dtype,
                "shape": tuple(np.shape(data)),
                "data": data,
                "truncated": truncated,
            }

    def canonicalize(self, locator: str) -> str:
        """Normalize an object path: one leading slash, no empty or `.` parts.

        Dots inside names are kept; HDF5 names routinely contain them. The
        RFC's attribute (`@name`) and selection (`[...]`) suffixes are not
        parsed, because both characters are legal in HDF5 link names.
        """
        parts = [part for part in locator.split("/") if part not in ("", ".")]
        return "/" + "/".join(parts)

    def complete(self, path: Path, partial: str) -> list[str]:
        """Locators of the children of `partial`'s parent that start with its last part."""
        if partial == "" or partial.endswith("/"):
            parent, prefix = partial or "/", ""
        else:
            parent, _, prefix = partial.rpartition("/")
        try:
            nodes = self.children(path, parent or "/")
        except ProviderError:
            return []
        return [node["locator"] for node in nodes if node["name"].startswith(prefix)]

    # -- file access ------------------------------------------------------

    @contextmanager
    def _open(self, path: Path, locator: str) -> Iterator[h5py.File]:
        """Open `path` read-only, mapping h5py errors onto provider error codes."""
        try:
            f = h5py.File(path, "r")
        except FileNotFoundError as e:
            raise ProviderError("not_found", f"No such file: {path}", locator) from e
        except OSError as e:
            raise ProviderError(_open_error_code(e), str(e), locator) from e

        try:
            with f:
                yield f
        except OSError as e:
            raise ProviderError("io", str(e), locator) from e

    @staticmethod
    def _resolve(f: h5py.File, locator: str) -> Any:
        """The object at `locator`, following valid links and leaving broken ones."""
        try:
            return get_entity_from_file(f, locator, LinkResolution.ONLY_VALID)
        except NotFoundError as e:
            raise ProviderError("not_found", str(e), locator) from e

    # -- nodes ------------------------------------------------------------

    @staticmethod
    def _node(entity: Any, locator: str) -> Node:
        name = locator.rsplit("/", 1)[-1] or "/"
        kind: NodeKind
        shape: tuple[int, ...] | None = None
        dtype: str | None = None
        has_children = False

        if isinstance(entity, h5py.Group):
            kind = "container"
            has_children = len(entity) > 0
        elif isinstance(entity, h5py.Dataset):
            kind = _dataset_kind(entity)
            shape = entity.shape
            dtype = _dtype_label(entity.dtype)
        elif isinstance(entity, h5py.Datatype):
            kind = "blob"
            dtype = _dtype_label(entity.dtype)
        else:
            kind = "link"

        return {
            "locator": locator,
            "name": name,
            "kind": kind,
            "shape": shape,
            "dtype": dtype,
            "has_children": has_children,
        }

    # -- describe helpers -------------------------------------------------

    def _attributes(self, entity: Any) -> dict[str, Any]:
        attributes: dict[str, Any] = {}
        for name in sorted(entity.attrs.keys()):
            attr_id = entity.attrs.get_id(name)
            shape = attr_id.shape
            if shape is not None and math.prod(shape) > self.attribute_max_elements:
                attributes[name] = {
                    "shape": shape,
                    "dtype": _dtype_label(attr_id.dtype),
                    "omitted": True,
                }
                continue
            try:
                attributes[name] = _jsonable(entity.attrs[name])
            except (OSError, TypeError, ValueError) as e:
                attributes[name] = {"error": str(e)}
        return attributes

    @staticmethod
    def _storage(dataset: h5py.Dataset) -> dict[str, Any]:
        layout = dataset.id.get_create_plist().get_layout()
        logical = (
            0
            if dataset.shape is None
            else math.prod(dataset.shape) * dataset.dtype.itemsize
        )
        return {
            "layout": _LAYOUTS.get(layout, str(layout)),
            "chunks": dataset.chunks,
            "filters": _jsonable(get_filters(dataset)),
            "storage_bytes": dataset.id.get_storage_size(),
            "logical_bytes": logical,
        }

    def _preview(self, dataset: h5py.Dataset) -> Any:
        if dataset.shape is None:
            return None
        view = _string_view(dataset)
        if dataset.ndim == 0:
            return self._clip_text(_jsonable(view[()]))

        kind = _dataset_kind(dataset)
        if kind == "table":
            rows = view[: self.table_preview_rows]
            names = dataset.dtype.names
            return {
                "rows": [dict(zip(names, _jsonable(row))) for row in rows],
                "total_rows": dataset.shape[0],
            }

        if dataset.ndim == 1:
            n = dataset.shape[0]
            edge = self.preview_edge
            if n <= 2 * edge:
                return {"values": self._clip_text(_jsonable(view[:])), "count": n}
            return {
                "head": self._clip_text(_jsonable(view[:edge])),
                "tail": self._clip_text(_jsonable(view[n - edge :])),
                "count": n,
            }

        if _is_real_numeric(dataset.dtype):
            return self._stats(dataset)

        leading = (0,) * (dataset.ndim - 1) + (slice(0, self.preview_edge),)
        return {"first": self._clip_text(_jsonable(view[leading]))}

    def _stats(self, dataset: h5py.Dataset) -> dict[str, Any]:
        """min/mean/max over the dataset, or over a leading slab if it is large."""
        row_bytes = max(1, math.prod(dataset.shape[1:]) * dataset.dtype.itemsize)
        if row_bytes > self.stats_max_bytes:
            return {
                "omitted": True,
                "reason": f"one row is over the {self.stats_max_bytes}-byte stats budget",
            }
        rows = dataset.shape[0]
        sampled = rows * row_bytes > self.stats_max_bytes
        if sampled:
            rows = self.stats_max_bytes // row_bytes
        data = np.asarray(dataset[:rows])
        if np.issubdtype(data.dtype, np.floating):
            data = data[np.isfinite(data)]
        # Not h5grove's get_array_stats: it casts the mean of integer data to int.
        if data.size == 0:
            stats: dict[str, Any] = {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
            }
        else:
            stats = {
                "min": _jsonable(data.min()),
                "max": _jsonable(data.max()),
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
            }
        return {**stats, "sampled": sampled, "sampled_rows": rows if sampled else None}

    def _clip_text(self, value: Any) -> Any:
        if isinstance(value, str) and len(value) > self.text_preview_chars:
            return value[: self.text_preview_chars] + "…"
        if isinstance(value, list):
            return [self._clip_text(v) for v in value]
        return value

    def _fit_preview(self, preview: Any) -> Any:
        """Shrink lists and strings until the serialized preview fits the cap."""
        if _json_size(preview) <= self.preview_max_bytes:
            return preview
        shrunk = preview
        for _ in range(64):
            shrunk = _halve(shrunk)
            if _json_size(shrunk) <= self.preview_max_bytes:
                if isinstance(shrunk, dict):
                    shrunk["truncated"] = True
                return shrunk
        return {
            "omitted": True,
            "reason": f"preview exceeds {self.preview_max_bytes} bytes",
        }

    # -- read helpers -----------------------------------------------------

    @staticmethod
    def _parse_selection(
        dataset: h5py.Dataset, selection: str | None, locator: str
    ) -> tuple[slice | int, ...]:
        if selection is None or selection.strip() == "":
            return ()
        try:
            # h5grove's parser chokes on whitespace before an empty slice
            # member (" ::2"), so strip it first.
            index = parse_slice("".join(selection.split()))
        except (ValueError, TypeError) as e:
            raise ProviderError(
                "unsupported", f"Invalid selection {selection!r}: {e}", locator
            ) from e
        if len(index) > dataset.ndim:
            raise ProviderError(
                "unsupported",
                f"Selection {selection!r} has {len(index)} members "
                f"for a {dataset.ndim}-D dataset",
                locator,
            )
        return index

    def _cap_selection(
        self, dataset: h5py.Dataset, index: tuple[slice | int, ...], locator: str
    ) -> tuple[tuple[slice | int, ...], bool]:
        """Trim `index` along its leading axes so the result fits `read_max_bytes`."""
        if dataset.ndim == 0:
            return (), False

        full = list(index) + [slice(None)] * (dataset.ndim - len(index))
        item_bytes = (
            VLEN_ELEMENT_BYTES if dataset.dtype.kind == "O" else dataset.dtype.itemsize
        )
        if item_bytes > self.read_max_bytes:
            raise ProviderError(
                "too_large",
                f"One element is {item_bytes} bytes, over the "
                f"{self.read_max_bytes}-byte read cap",
                locator,
            )
        budget = self.read_max_bytes // item_bytes

        # (axis, start, step, count) for each axis kept in the result.
        kept: list[tuple[int, int, int, int]] = []
        for axis, member in enumerate(full):
            if isinstance(member, slice):
                start, stop, step = member.indices(dataset.shape[axis])
                kept.append((axis, start, step, len(range(start, stop, step))))

        counts = [count for _, _, _, count in kept]
        if math.prod(counts) <= budget:
            return tuple(index), False

        for i in range(len(counts)):
            rest = math.prod(counts[i + 1 :])
            counts[i] = max(1, min(counts[i], budget // max(1, rest)))
            if math.prod(counts) <= budget:
                break

        for (axis, start, step, _), count in zip(kept, counts):
            full[axis] = slice(start, start + count * step, step)
        return tuple(full), True


# -- module helpers ------------------------------------------------------


def _open_error_code(error: OSError) -> ErrorCode:
    message = str(error).lower()
    if "lock" in message:
        return "locked"
    if "signature" in message:
        return "unsupported"
    if "no such file" in message:
        return "not_found"
    return "io"


def _dataset_kind(dataset: h5py.Dataset) -> NodeKind:
    dtype = dataset.dtype
    if dataset.shape is None or dataset.ndim == 0:
        if h5py.check_string_dtype(dtype) is not None:
            return "text"
        return "scalar"
    if dtype.names is not None:
        return "table" if dataset.ndim == 1 else "array"
    if dtype.kind == "V" and dtype.subdtype is None:
        return "blob"
    return "array"


def _dtype_label(dtype: np.dtype) -> str:
    string = h5py.check_string_dtype(dtype)
    if string is not None:
        return f"string({string.encoding})"
    vlen = h5py.check_vlen_dtype(dtype)
    if vlen is not None:
        return f"vlen<{_dtype_label(vlen)}>"
    if h5py.check_ref_dtype(dtype) is not None:
        return "reference"
    enum = h5py.check_enum_dtype(dtype)
    if enum is not None:
        return f"enum<{dtype.name}>"
    if dtype.names is not None:
        return "compound"
    if dtype.subdtype is not None:
        base, shape = dtype.subdtype
        return f"{_dtype_label(base)}{list(shape)}"
    if dtype.kind == "V":
        return "opaque"
    return dtype.name


def _is_real_numeric(dtype: np.dtype) -> bool:
    return dtype.kind in "biuf" and h5py.check_enum_dtype(dtype) is None


def _render_hint(entity: Any, kind: NodeKind) -> str | None:
    if kind in ("scalar", "text", "table"):
        return kind
    if kind == "array" and isinstance(entity, h5py.Dataset):
        if entity.dtype.kind in "biufc":
            return "line" if entity.ndim == 1 else "heatmap"
        return "table" if entity.ndim == 1 else None
    return None


def _string_view(dataset: h5py.Dataset) -> Any:
    """Dataset view that decodes strings to `str` instead of `bytes`."""
    if h5py.check_string_dtype(dataset.dtype) is not None:
        return dataset.asstr(errors="replace")
    return dataset


def _jsonable(value: Any) -> Any:
    """Convert h5py/NumPy values to plain JSON-serializable Python.

    NaN and infinities become strings, complex numbers become [real, imag],
    and opaque bytes become hex.
    """
    if isinstance(value, h5py.Empty):
        return None
    if isinstance(value, (h5py.Reference, h5py.RegionReference)):
        return str(value)
    if isinstance(value, np.void):
        if value.dtype.names is not None:
            return [_jsonable(value[name]) for name in value.dtype.names]
        return value.tobytes().hex()
    if isinstance(value, np.ndarray):
        if value.dtype.names is not None or value.dtype.kind == "V":
            return [_jsonable(item) for item in value]
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, complex):
        return [_jsonable(value.real), _jsonable(value.imag)]
    return value


def _json_size(value: Any) -> int:
    return len(json.dumps(value, default=str))


def _halve(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _halve(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_halve(v) for v in value[: max(1, len(value) // 2)]]
    if isinstance(value, str) and len(value) > 8:
        return value[: len(value) // 2] + "…"
    return value
