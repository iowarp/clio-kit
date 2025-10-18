#!/usr/bin/env python3
# /// script
# dependencies = [
#   "mcp>=1.4.0",
#   "h5py>=3.9.0",
#   "numpy>=1.24.0,<2.0.0",
#   "pydantic>=2.4.2,<3.0.0",
#   "aiofiles>=23.2.1",
#   "aiohttp>=3.9.0",
#   "psutil>=5.9.0",
#   "dask>=2023.0.0",
#   "jinja2>=3.1.0",
#   "python-dotenv>=0.19.0"
# ]
# requires-python = ">=3.10"
# ///

"""
Enhanced HDF5 MCP Server with multi-transport support and performance optimizations.
"""
import asyncio
import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

# Import MCP components
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import JSONRPCMessage

# Import our modules
try:
    # Try relative imports first (when run as module)
    from .config import get_config, set_storage_path
    from .transports import TransportManager, StdioTransport, SSETransport
    from .transports.base import TransportConfig, TransportType
    from .tools import get_tools
    from .resources import get_resources
    from .prompts import get_available_prompts
except ImportError:
    # Fall back to absolute imports (when run as script)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from .config import get_config, set_storage_path
    from .transports import TransportManager, StdioTransport, SSETransport
    from .transports.base import TransportConfig, TransportType
    from .tools import get_tools
    from .resources import get_resources
    from .prompts import get_available_prompts

logger = logging.getLogger(__name__)

class EnhancedHDF5Server:
    """Enhanced HDF5 MCP Server with multi-transport support."""
    
    def __init__(self):
        self.config = get_config()
        self.transport_manager = TransportManager()
        self.mcp_server: Optional[Server] = None
        self.running = False
        
    async def start(self, data_dir: Optional[Path] = None) -> None:
        """Start the enhanced HDF5 MCP server."""
        try:
            # Set data directory if provided
            if data_dir:
                set_storage_path(data_dir)
                
            # Create MCP server instance
            self.mcp_server = Server("hdf5-mcp-server")
            
            # Register tools, resources, and prompts
            await self._register_handlers()
            
            # Setup transports
            await self._setup_transports()
            
            # Set message handler
            self.transport_manager.set_message_handler(self._handle_message)
            
            # Start all transports
            await self.transport_manager.start_all()
            
            self.running = True
            logger.info("Enhanced HDF5 MCP Server started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise
            
    async def stop(self) -> None:
        """Stop the server."""
        self.running = False
        
        # Stop all transports
        await self.transport_manager.stop_all()
        
        logger.info("Enhanced HDF5 MCP Server stopped")
        
    async def _setup_transports(self) -> None:
        """Setup transport instances based on configuration."""
        
        # Always setup STDIO transport (primary for Claude Code)
        if self.config.transport.enable_stdio:
            stdio_config = TransportConfig(
                transport_type=TransportType.STDIO,
                enable_batching=self.config.transport.enable_batching,
                batch_timeout=self.config.transport.batch_timeout,
                max_batch_size=self.config.transport.max_batch_size
            )
            stdio_transport = StdioTransport(stdio_config)
            self.transport_manager.add_transport("stdio", stdio_transport)
            
        # Setup SSE transport if enabled
        if self.config.transport.enable_sse:
            sse_config = TransportConfig(
                transport_type=TransportType.SSE,
                host=self.config.transport.sse_host,
                port=self.config.transport.sse_port,
                max_connections=self.config.transport.max_connections,
                enable_batching=self.config.transport.enable_batching,
                batch_timeout=self.config.transport.batch_timeout,
                max_batch_size=self.config.transport.max_batch_size
            )
            sse_transport = SSETransport(sse_config)
            self.transport_manager.add_transport("sse", sse_transport)
            
    async def _register_handlers(self) -> None:
        """Register MCP handlers for tools, resources, and prompts."""
        if not self.mcp_server:
            raise RuntimeError("MCP server not initialized")
            
        # Register tools
        tools = get_tools()
        for tool in tools:
            self.mcp_server.register_tool(tool.name, tool.description, tool.inputSchema)
            
        # Register resources  
        resources = get_resources()
        for resource in resources:
            # Resource registration will be handled by the resource manager
            pass
            
        # Register prompts
        prompts = get_available_prompts()
        for prompt in prompts:
            self.mcp_server.register_prompt(prompt.name, prompt.description, prompt.arguments)
            
    async def _handle_message(self, message: JSONRPCMessage, transport_name: str) -> None:
        """Handle incoming messages from transports."""
        try:
            # Process message through MCP server
            # This is a simplified version - in practice you'd need to properly
            # route the message through the MCP server's request handling
            logger.debug(f"Received message from {transport_name}: {message}")
            
            # For now, just acknowledge receipt
            # In a full implementation, this would process the request and send back a response
            
        except Exception as e:
            logger.error(f"Error handling message from {transport_name}: {e}")

async def run_stdio_mode(data_dir: Optional[Path] = None):
    """Run server in STDIO mode (for Claude Code)."""
    if data_dir:
        set_storage_path(data_dir)
        
    # Import the existing server implementation
    try:
        from .server import run_server
    except ImportError:
        from .server import run_server
    
    # Run the existing server implementation
    await run_server()

async def run_enhanced_mode(data_dir: Optional[Path] = None):
    """Run server in enhanced mode with multiple transports."""
    server = EnhancedHDF5Server()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(server.stop())
        
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda s, f: signal_handler())
        
    try:
        await server.start(data_dir)
        
        # Keep server running
        while server.running:
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"Server error: {e}")
        await server.stop()
        raise

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="HDF5 MCP Server")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing HDF5 files"
    )
    parser.add_argument(
        "--mode",
        choices=["stdio", "enhanced"],
        default="stdio",
        help="Server mode (stdio for Claude Code, enhanced for multi-transport)"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host for SSE transport (enhanced mode only)"
    )
    parser.add_argument(
        "--port", 
        type=int,
        default=8765,
        help="Port for SSE transport (enhanced mode only)"
    )
    parser.add_argument(
        "--enable-sse",
        action="store_true",
        help="Enable SSE transport (enhanced mode only)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set environment variables from command line args
    if args.data_dir:
        import os
        os.environ['HDF5_MCP_DATA_DIR'] = str(args.data_dir)
    if args.enable_sse:
        import os
        os.environ['HDF5_MCP_ENABLE_SSE'] = 'true'
        os.environ['HDF5_MCP_HOST'] = args.host  
        os.environ['HDF5_MCP_PORT'] = str(args.port)
    
    try:
        if args.mode == "stdio":
            asyncio.run(run_stdio_mode(args.data_dir))
        else:
            asyncio.run(run_enhanced_mode(args.data_dir))
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()