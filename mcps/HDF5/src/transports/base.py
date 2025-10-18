#!/usr/bin/env python3
# /// script
# dependencies = [
#   "mcp>=1.4.0",
#   "pydantic>=2.4.2,<3.0.0",
#   "aiohttp>=3.9.0"
# ]
# requires-python = ">=3.10"
# ///

"""
Base transport abstraction for HDF5 MCP Server.
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum

from mcp.types import (
    JSONRPCMessage,
    JSONRPCRequest,
    JSONRPCResponse,
    JSONRPCNotification
)

logger = logging.getLogger(__name__)

class TransportType(Enum):
    """Available transport types."""
    STDIO = "stdio"
    SSE = "sse"
    HTTP = "http"

@dataclass
class TransportConfig:
    """Configuration for transport instances."""
    transport_type: TransportType
    host: Optional[str] = None
    port: Optional[int] = None
    max_connections: int = 100
    enable_batching: bool = True
    batch_timeout: float = 0.1  # seconds
    max_batch_size: int = 50

class BaseTransport(ABC):
    """Base class for all transport implementations."""
    
    def __init__(self, config: TransportConfig):
        self.config = config
        self.running = False
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'batches_sent': 0,
            'batches_received': 0,
            'errors': 0
        }
        self._batch_buffer: List[JSONRPCMessage] = []
        self._batch_timer: Optional[asyncio.Task] = None
        
    @abstractmethod
    async def start(self) -> None:
        """Start the transport."""
        pass
        
    @abstractmethod
    async def stop(self) -> None:
        """Stop the transport."""
        pass
        
    @abstractmethod
    async def send_message(self, message: JSONRPCMessage) -> None:
        """Send a single message."""
        pass
        
    @abstractmethod
    async def send_batch(self, messages: List[JSONRPCMessage]) -> None:
        """Send a batch of messages."""
        pass
        
    @abstractmethod
    async def receive_message(self) -> Optional[JSONRPCMessage]:
        """Receive a single message."""
        pass
        
    @abstractmethod
    async def receive_batch(self) -> List[JSONRPCMessage]:
        """Receive a batch of messages."""
        pass
        
    async def send_with_batching(self, message: JSONRPCMessage) -> None:
        """Send message with optional batching."""
        if not self.config.enable_batching:
            await self.send_message(message)
            return
            
        self._batch_buffer.append(message)
        
        # Start batch timer if not already running
        if self._batch_timer is None:
            self._batch_timer = asyncio.create_task(self._flush_batch_after_timeout())
            
        # Flush immediately if batch is full
        if len(self._batch_buffer) >= self.config.max_batch_size:
            await self._flush_batch()
            
    async def _flush_batch_after_timeout(self) -> None:
        """Flush batch after timeout."""
        try:
            await asyncio.sleep(self.config.batch_timeout)
            await self._flush_batch()
        except asyncio.CancelledError:
            pass
            
    async def _flush_batch(self) -> None:
        """Flush the current batch."""
        if not self._batch_buffer:
            return
            
        messages = self._batch_buffer.copy()
        self._batch_buffer.clear()
        
        if self._batch_timer:
            self._batch_timer.cancel()
            self._batch_timer = None
            
        try:
            if len(messages) == 1:
                await self.send_message(messages[0])
                self.stats['messages_sent'] += 1
            else:
                await self.send_batch(messages)
                self.stats['batches_sent'] += 1
                self.stats['messages_sent'] += len(messages)
        except Exception as e:
            logger.error(f"Failed to send batch: {e}")
            self.stats['errors'] += 1
            
    def get_stats(self) -> Dict[str, Any]:
        """Get transport statistics."""
        return {
            'type': self.config.transport_type.value,
            'running': self.running,
            **self.stats
        }

class TransportManager:
    """Manages multiple transport instances."""
    
    def __init__(self):
        self.transports: Dict[str, BaseTransport] = {}
        self.message_handler: Optional[Callable[[JSONRPCMessage, str], Awaitable[None]]] = None
        
    def add_transport(self, name: str, transport: BaseTransport) -> None:
        """Add a transport instance."""
        self.transports[name] = transport
        
    def remove_transport(self, name: str) -> None:
        """Remove a transport instance."""
        if name in self.transports:
            del self.transports[name]
            
    def set_message_handler(self, handler: Callable[[JSONRPCMessage, str], Awaitable[None]]) -> None:
        """Set the message handler for incoming messages."""
        self.message_handler = handler
        
    async def start_all(self) -> None:
        """Start all transports."""
        tasks = []
        for name, transport in self.transports.items():
            tasks.append(self._start_transport(name, transport))
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def stop_all(self) -> None:
        """Stop all transports."""
        tasks = []
        for transport in self.transports.values():
            tasks.append(transport.stop())
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _start_transport(self, name: str, transport: BaseTransport) -> None:
        """Start a single transport and listen for messages."""
        try:
            await transport.start()
            logger.info(f"Started transport: {name}")
            
            # Start message receiving loop
            asyncio.create_task(self._receive_loop(name, transport))
            
        except Exception as e:
            logger.error(f"Failed to start transport {name}: {e}")
            
    async def _receive_loop(self, name: str, transport: BaseTransport) -> None:
        """Message receiving loop for a transport."""
        while transport.running:
            try:
                # Try to receive batch first
                messages = await transport.receive_batch()
                if messages:
                    transport.stats['batches_received'] += 1
                    transport.stats['messages_received'] += len(messages)
                    
                    if self.message_handler:
                        for message in messages:
                            await self.message_handler(message, name)
                    continue
                    
                # Fall back to single message
                message = await transport.receive_message()
                if message:
                    transport.stats['messages_received'] += 1
                    
                    if self.message_handler:
                        await self.message_handler(message, name)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in receive loop for {name}: {e}")
                transport.stats['errors'] += 1
                await asyncio.sleep(0.1)  # Brief pause on error
                
    async def broadcast_message(self, message: JSONRPCMessage) -> None:
        """Broadcast message to all transports."""
        tasks = []
        for transport in self.transports.values():
            if transport.running:
                tasks.append(transport.send_with_batching(message))
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def send_to_transport(self, transport_name: str, message: JSONRPCMessage) -> None:
        """Send message to specific transport."""
        if transport_name in self.transports:
            transport = self.transports[transport_name]
            if transport.running:
                await transport.send_with_batching(message)
                
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all transports."""
        return {
            name: transport.get_stats()
            for name, transport in self.transports.items()
        }