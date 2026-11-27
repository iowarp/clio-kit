"""Shared parsing for ADIOS2 hyperslab selections given by an agent."""

import json
from math import prod
from typing import Any, List, Optional, Sequence


def parse_indices(value: Any, name: str) -> Optional[List[int]]:
    """Accept a list of ints or its JSON text such as "[0, 4]"; None stays None."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{name} must be a list of integers, got {value!r}"
            ) from exc
    if isinstance(value, int) and not isinstance(value, bool):
        value = [value]
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise ValueError(f"{name} must be a list of integers, got {value!r}")
    return value


def parse_shape(shape: Any) -> List[int]:
    """ADIOS reports Shape as text like "48, 48, 24"; a scalar has an empty shape."""
    if isinstance(shape, (list, tuple)):
        return [int(item) for item in shape]
    text = str(shape or "").strip()
    return [int(part) for part in text.split(",") if part.strip()]


def check_selection(
    shape: Sequence[int], start: Optional[List[int]], count: Optional[List[int]]
) -> List[int]:
    """Validate start/count against the global shape and return the selected shape."""
    if start is None and count is None:
        return list(shape)
    if not shape:
        raise ValueError("start/count apply only to array variables, not scalars")
    start = start if start is not None else [0] * len(shape)
    count = (
        count
        if count is not None
        else [dim - offset for dim, offset in zip(shape, start)]
    )
    if len(start) != len(shape) or len(count) != len(shape):
        raise ValueError(
            f"start and count need {len(shape)} values each for shape {list(shape)}"
        )
    for axis, (dim, offset, size) in enumerate(zip(shape, start, count)):
        if offset < 0 or size <= 0 or offset + size > dim:
            raise ValueError(
                f"selection on axis {axis} (start {offset}, count {size}) "
                f"is outside 0..{dim}"
            )
    return list(count)


def elements(shape: Sequence[int]) -> int:
    return prod(shape) if shape else 1
