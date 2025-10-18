#!/usr/bin/env python3
# /// script
# dependencies = [
#   "mcp>=1.4.0",
#   "pydantic>=2.4.2,<3.0.0"
# ]
# requires-python = ">=3.10"
# ///

"""
Enhanced STDIO transport with JSON-RPC batching support.
"""
import asyncio
import json
import sys
import logging
from typing import List, Optional, Union

from mcp.types import (
    JSONRPCMessage,
    JSONRPCRequest,
    JSONRPCResponse,
    JSONRPCNotification
)

from .base import BaseTransport, TransportConfig, TransportType

logger = logging.getLogger(__name__)

class StdioTransport(BaseTransport):
    """STDIO transport with batching support for MCP."""
    
    def __init__(self, config: TransportConfig = None):
        if config is None:
            config = TransportConfig(
                transport_type=TransportType.STDIO,
                enable_batching=True,
                batch_timeout=0.05,  # Faster batching for STDIO
                max_batch_size=20    # Smaller batches for STDIO
            )
        super().__init__(config)
        self._stdin_reader: Optional[asyncio.StreamReader] = None
        self._stdout_writer: Optional[asyncio.StreamWriter] = None
        self._receive_queue: asyncio.Queue = asyncio.Queue()
        
    async def start(self) -> None:
        """Start the STDIO transport."""
        try:
            # Create stdin reader
            self._stdin_reader = asyncio.StreamReader()
            stdin_protocol = asyncio.StreamReaderProtocol(self._stdin_reader)
            stdin_transport, _ = await asyncio.get_event_loop().connect_read_pipe(
                lambda: stdin_protocol, sys.stdin
            )
            
            # Create stdout writer  
            stdout_transport, stdout_protocol = await asyncio.get_event_loop().connect_write_pipe(
                asyncio.streams.FlowControlMixin, sys.stdout
            )
            self._stdout_writer = asyncio.StreamWriter(
                stdout_transport, stdout_protocol, None, asyncio.get_event_loop()
            )
            
            self.running = True
            
            # Start reading messages
            asyncio.create_task(self._read_messages())
            
            logger.info("STDIO transport started")
            
        except Exception as e:
            logger.error(f"Failed to start STDIO transport: {e}")
            raise
            
    async def stop(self) -> None:
        """Stop the STDIO transport."""
        self.running = False
        
        if self._stdout_writer:
            self._stdout_writer.close()
            await self._stdout_writer.wait_closed()
            
        logger.info("STDIO transport stopped")
        
    async def send_message(self, message: JSONRPCMessage) -> None:
        """Send a single message via STDIO."""
        if not self._stdout_writer:
            raise RuntimeError("STDIO transport not started")
            
        try:
            # Serialize message
            if hasattr(message, 'model_dump'):
                data = message.model_dump()
            elif hasattr(message, 'dict'):
                data = message.dict()
            else:
                data = message
                
            json_str = json.dumps(data, separators=(',', ':'))
            
            # Write to stdout with newline delimiter
            self._stdout_writer.write(f"{json_str}\n".encode('utf-8'))
            await self._stdout_writer.drain()
            
            logger.debug(f"Sent message: {json_str[:100]}...")
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.stats['errors'] += 1
            raise
            
    async def send_batch(self, messages: List[JSONRPCMessage]) -> None:
        """Send a batch of messages via STDIO."""
        if not messages:
            return
            
        if not self._stdout_writer:
            raise RuntimeError("STDIO transport not started")
            
        try:
            # Serialize batch
            batch_data = []
            for message in messages:
                if hasattr(message, 'model_dump'):
                    data = message.model_dump()
                elif hasattr(message, 'dict'):
                    data = message.dict()
                else:
                    data = message
                batch_data.append(data)
                
            json_str = json.dumps(batch_data, separators=(',', ':'))
            
            # Write batch to stdout
            self._stdout_writer.write(f"{json_str}\n".encode('utf-8'))
            await self._stdout_writer.drain()
            
            logger.debug(f"Sent batch of {len(messages)} messages")
            
        except Exception as e:
            logger.error(f"Failed to send batch: {e}")
            self.stats['errors'] += 1
            raise
            
    async def receive_message(self) -> Optional[JSONRPCMessage]:
        """Receive a single message from the queue."""
        try:
            # Try to get message without blocking
            return self._receive_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
            
    async def receive_batch(self) -> List[JSONRPCMessage]:
        """Receive a batch of messages from the queue."""
        messages = []
        
        # Get first message (blocking)
        try:
            first_message = await asyncio.wait_for(
                self._receive_queue.get(), timeout=0.01
            )
            messages.append(first_message)
        except asyncio.TimeoutError:
            return messages
            
        # Get additional messages (non-blocking)
        while len(messages) < self.config.max_batch_size:
            try:
                message = self._receive_queue.get_nowait()
                messages.append(message)
            except asyncio.QueueEmpty:
                break
                
        return messages
        
    async def _read_messages(self) -> None:
        """Read messages from stdin."""
        while self.running and self._stdin_reader:
            try:
                # Read line from stdin
                line = await self._stdin_reader.readline()
                if not line:
                    break
                    
                line_str = line.decode('utf-8').strip()
                if not line_str:
                    continue
                    
                # Parse JSON
                try:
                    data = json.loads(line_str)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
                    self.stats['errors'] += 1
                    continue
                    
                # Handle batch or single message
                if isinstance(data, list):
                    # Batch of messages
                    for item in data:
                        message = self._parse_message(item)
                        if message:
                            await self._receive_queue.put(message)
                else:
                    # Single message
                    message = self._parse_message(data)
                    if message:
                        await self._receive_queue.put(message)
                        
            except Exception as e:
                if self.running:
                    logger.error(f"Error reading message: {e}")
                    self.stats['errors'] += 1
                break
                
    def _parse_message(self, data: dict) -> Optional[JSONRPCMessage]:
        """Parse message data into appropriate MCP type."""
        try:
            # Determine message type
            if 'method' in data:
                if 'id' in data:
                    # Request
                    return JSONRPCRequest(**data)
                else:
                    # Notification
                    return JSONRPCNotification(**data)
            elif 'result' in data or 'error' in data:
                # Response
                return JSONRPCResponse(**data)
            else:
                logger.warning(f"Unknown message format: {data}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to parse message: {e}")
            self.stats['errors'] += 1
            return None