"""
HDF5 MCP server implementation with asynchronous processing and resource management.
"""
#!/usr/bin/env python3
# /// script
# dependencies = [
#   "mcp>=1.4.0",
#   "h5py>=3.9.0",
#   "numpy>=1.24.0,<2.0.0",
#   "pydantic>=2.4.2,<3.0.0",
#   "aiofiles>=23.2.1"
# ]
# requires-python = ">=3.10"
# ///

# =========================================================================
# Dependencies
# =========================================================================
import asyncio
import logging
import json
import os
import uuid
import time
import h5py
import numpy as np

from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from contextlib import asynccontextmanager
from importlib import metadata

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    GetPromptResult,
    Prompt,
    PromptMessage,
    PromptArgument
)

# Internal modules
from .config import Config, get_config
from .utils import HDF5Manager
from .tools import get_tools, HDF5Tools, create_tools
from .resources import ResourceManager, get_resources
from .prompts import get_available_prompts, generate_prompt
from .scanner import HDF5Scanner
from .protocol import (
    ServerState,
    ServerStatus,
    Message,
    MessageType,
    CommandRequest,
    CommandResponse,
    ErrorResponse,
    Notification,
    NotificationType,
    ErrorCodes
)
from .task_queue import AsyncTaskQueue

# =========================================================================
# Logging
# =========================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================================================================
# Server status tracking
# =========================================================================
from enum import Enum, auto
from dataclasses import dataclass
from datetime import datetime

class ServerState(Enum):
    INITIALIZING = auto()
    RUNNING = auto()
    SHUTDOWN = auto()
    ERROR = auto()

@dataclass
class ServerStatus:
    """Server status information."""
    state: ServerState = ServerState.INITIALIZING
    start_time: Optional[datetime] = None
    active_connections: int = 0
    requests_handled: int = 0
    errors: int = 0
    active_files: Set[str] = None
    
    def __post_init__(self):
        if self.active_files is None:
            self.active_files = set()
    
    @property
    def uptime_seconds(self) -> Optional[float]:
        """Calculate server uptime in seconds."""
        if self.start_time is None:
            return None
        return (datetime.now() - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "state": self.state.name,
            "active_connections": self.active_connections,
            "requests_handled": self.requests_handled,
            "errors": self.errors,
            "active_files": list(self.active_files),
        }
        
        if self.start_time:
            result["start_time"] = self.start_time.isoformat()
            result["uptime_seconds"] = self.uptime_seconds
            
        return result

# =========================================================================
# Server implementation
# =========================================================================

class HDF5Server:
    """HDF5 Model Context Protocol server implementation."""
    
    def __init__(self):
        self.config = get_config()
        self.server = Server(name="HDF5 MCP Server")
        self._handlers_registered = False
        self.status = ServerStatus()
        self.resource_manager = ResourceManager()
        self.tools = create_tools()
        self.task_queue = AsyncTaskQueue(
            max_workers=self.config.async_config.max_workers
        )
        self.cleanup_task = None
        
        # Active file tracking
        self._active_files = set()
        
        # Cleanup interval in seconds
        self._cleanup_interval = 60
        
        # Metrics
        self._metrics = {
            "requests_total": 0,
            "errors_total": 0,
            "operation_times": {}
        }
        
        self._register_handlers()
    
    @property
    def status(self) -> ServerStatus:
        """Get current server status."""
        return self._status
    
    @status.setter
    def status(self, status: ServerStatus):
        """Set server status."""
        self._status = status
    
    @asynccontextmanager
    async def file_access(self, file_path: str):
        """Context manager for tracking active file access."""
        file_path = str(file_path)
        self.status.active_files.add(file_path)
        try:
            yield
        finally:
            if file_path in self.status.active_files:
                self.status.active_files.remove(file_path)
    
    async def initialize(self):
        """Initialize the server."""
        try:
            logger.info("Initializing HDF5 MCP server...")
            
            # Ensure data directory exists
            data_dir = self.config.hdf5.data_dir
            if not data_dir.exists():
                logger.info(f"Creating data directory: {data_dir}")
                data_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize resource manager
            await self.resource_manager.initialize()
            
            # Ensure handlers are registered (server already created in __init__)
            if not self._handlers_registered:
                self._register_handlers()
            
            # Start the task queue
            await self.task_queue.start()
            
            # Set server as running
            self.status.state = ServerState.RUNNING
            self.status.start_time = datetime.now()
            
            # Start periodic cleanup
            self.cleanup_task = asyncio.create_task(self._periodic_cleanup())
            
            logger.info(f"Server initialized successfully. Data directory: {data_dir}")
        except Exception as e:
            logger.error(f"Error initializing server: {e}")
            self.status.state = ServerState.ERROR
            self.status.errors += 1
            raise
    
    def _register_handlers(self):
        """Register all MCP protocol handlers."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools handler."""
            logger.debug("Handling list_tools request")
            tools = get_tools()
            return tools
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any] | None) -> List[TextContent | ImageContent | EmbeddedResource]:
            """Tool call handler."""
            logger.debug(f"Handling call_tool request: {name}")
            self.status.requests_handled += 1
            
            # Forward to the tools implementation
            result = await self.tools.__getattribute__(name)(**arguments or {})
            return result
        
        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            """List available resources handler."""
            logger.debug("Handling list_resources request")
            resources = get_resources()
            return resources
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str):
            """Read resource handler."""
            logger.debug(f"Handling read_resource request: {uri}")
            self.status.requests_handled += 1
            return await self.resource_manager.read_resource(uri)
        
        @self.server.list_prompts()
        async def handle_list_prompts() -> list[Prompt]:
            """List available prompts handler."""
            logger.debug("Handling list_prompts request")
            prompts = get_available_prompts()
            return prompts
        
        @self.server.get_prompt()
        async def handle_get_prompt(name: str, arguments: dict[str, str] | None) -> GetPromptResult:
            """Get prompt handler."""
            logger.debug(f"Handling get_prompt request: {name}")
            self.status.requests_handled += 1
            
            # Process prompt
            try:
                result = await generate_prompt(name, **(arguments or {}))
                return result
            except Exception as e:
                logger.error(f"Error generating prompt {name}: {e}")
                self.status.errors += 1
                return GetPromptResult(
                    messages=[
                        PromptMessage(
                            role="assistant",
                            content=TextContent(text="Error generating prompt")
                        ),
                        PromptMessage(
                            role="user",
                            content=TextContent(text=f"An error occurred: {str(e)}")
                        )
                    ]
                )
        
        # Registration code for handlers...
        self._handlers_registered = True
    
    async def _periodic_cleanup(self):
        """Run periodic cleanup tasks."""
        try:
            while self.status.state == ServerState.RUNNING:
                try:
                    await asyncio.sleep(self._cleanup_interval)
                    
                    logger.debug("Running periodic cleanup")
                    
                    # Clean up any stale resources
                    await self.resource_manager.cleanup_stale_resources()
                    
                    # Log status
                    logger.debug(f"Server status: {self.status.to_dict()}")
                except asyncio.CancelledError:
                    # Re-raise CancelledError to be caught by the outer try-except
                    raise
                except Exception as e:
                    logger.error(f"Error in periodic cleanup: {e}")
                    # Continue the loop rather than exiting on error
        except asyncio.CancelledError:
            logger.debug("Cleanup task cancelled")
        except Exception as e:
            logger.error(f"Fatal error in cleanup task: {e}")
    
    async def start(self):
        """Start the server."""
        if not self.server:
            await self.initialize()

        logger.info("Starting HDF5 MCP server with stdio transport")

        # Get the distribution info for versioning
        try:
            dist = metadata.distribution("hdf5-mcp-server")
            version = dist.version
        except:
            version = "0.1.0"

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="HDF5 MCP Server",
                    server_version=version,
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    
    async def shutdown(self):
        """Shutdown the server gracefully."""
        logger.info("Shutting down HDF5 MCP server...")
        
        # Update status
        self.status.state = ServerState.SHUTDOWN
        
        # Cancel cleanup task
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            try:
                await asyncio.gather(self.cleanup_task, return_exceptions=True)
            except Exception as e:
                logger.error(f"Error cancelling cleanup task: {e}")
        
        # Stop task queue
        if self.task_queue:
            await self.task_queue.stop()
        
        # Close any active files
        if hasattr(self.tools, 'file') and self.tools.file:
            self.tools.file.close()
        
        # Clean up resource manager
        await self.resource_manager.shutdown()
        
        # Clear references
        self.cleanup_task = None
        
        logger.info("Server shutdown complete")

def create_server() -> HDF5Server:
    """
    Create a new HDF5 MCP server instance.
    
    Returns:
        A configured server instance ready to be started
    """
    return HDF5Server()

async def run_server():
    """Run the MCP server using stdin/stdout streams"""
    # Create the server
    server = create_server()
    await server.initialize()

    # Get the distribution info for versioning
    try:
        dist = metadata.distribution("hdf5-mcp-server")
        version = dist.version
    except:
        version = "0.1.0"

    # Use stdio transport for communication
    logger.info("Starting stdio server")
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="HDF5 MCP Server",
                    server_version=version,
                    capabilities=server.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    except Exception as e:
        logger.error(f"Error in stdio server: {e}")
        raise
    finally:
        logger.info("Server communication ended")


async def start_server():
    """
    Start the HDF5 MCP server.
    """
    # Create and start the server
    server = create_server()
    await server.initialize()

    try:
        await run_server()
    except KeyboardInterrupt:
        logger.info("Server interrupted. Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        # Always ensure server is properly shutdown
        try:
            await server.shutdown()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


if __name__ == "__main__":
    # When run directly, start the server
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levellevel)s - %(message)s"
    )

    # Run the server
    asyncio.run(start_server())
