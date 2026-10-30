"""CLIO structured-file provider for HDF5.

Plugin entry point: ``hdf5_mcp.provider:HDF5Provider``.
"""

from .hdf5 import HDF5Provider
from .protocol import (
    ERROR_CODES,
    Description,
    Node,
    NodeKind,
    Payload,
    ProviderError,
    StructuredFileProvider,
)

__all__ = [
    "ERROR_CODES",
    "Description",
    "HDF5Provider",
    "Node",
    "NodeKind",
    "Payload",
    "ProviderError",
    "StructuredFileProvider",
]
