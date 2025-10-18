"""HDF5 MCP Server - Advanced HDF5 operations for AI agents.

Features:
- 25+ tools for comprehensive HDF5 file operations
- Lazy loading and LRU caching for performance
- Parallel operations for batch processing
- Streaming support for large datasets
- Discovery and optimization tools
"""

from .server import HDF5Server
from .tools import HDF5Tools, ToolRegistry

__version__ = "2.0.0"
__all__ = ["HDF5Server", "HDF5Tools", "ToolRegistry"]
