"""Whether an ADIOS2 BP5 output that is still being written can be read yet.

The checks live in ``disk``, which runs where the data is; see its docstring.
"""

from __future__ import annotations

from .disk import (
    Readiness,
    check,
    is_bp,
    source_readiness,
    whole_steps,
    write_index,
)

__all__ = [
    "Readiness",
    "check",
    "is_bp",
    "source_readiness",
    "whole_steps",
    "write_index",
]
