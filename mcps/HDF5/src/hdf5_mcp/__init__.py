"""HDF5 FastMCP Server - Next-generation HDF5 operations for AI agents.

Features:
- 25+ tools with zero boilerplate (@mcp.tool)
- Resource URIs (hdf5:// scheme)
- Analysis workflow prompts
- LRU caching (100-1000x speedup)
- Parallel operations (4-8x faster batch processing)
- Streaming support (unlimited file sizes)
- Discovery and optimization tools
"""

from .server import mcp, main

__version__ = "3.0.0"
__all__ = ["mcp", "main"]
