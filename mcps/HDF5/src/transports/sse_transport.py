#!/usr/bin/env python3
# /// script
# dependencies = [
#   "mcp>=1.4.0",
#   "aiohttp>=3.9.0",
#   "pydantic>=2.4.2,<3.0.0"
# ]
# requires-python = ">=3.10"
# ///

"""
Server-Sent Events (SSE) transport for streaming large datasets.
Implements the Streamable HTTP transport from MCP 2025-03-26 spec.
"""
import asyncio
import json
import logging
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass
from weakref import WeakSet

from aiohttp import web, WSMsgType
from aiohttp.web_request import Request
from aiohttp.web_response import Response, StreamResponse

from mcp.types import (
    JSONRPCMessage,
    JSONRPCRequest,
    JSONRPCResponse,
    JSONRPCNotification
)

from .base import BaseTransport, TransportConfig, TransportType

logger = logging.getLogger(__name__)

@dataclass
class SSEClient:
    """Represents an SSE client connection."""
    client_id: str
    request: Request
    response: StreamResponse
    queue: asyncio.Queue
    last_ping: float

class SSETransport(BaseTransport):
    """Server-Sent Events transport with JSON-RPC batching."""
    
    def __init__(self, config: TransportConfig = None):
        if config is None:
            config = TransportConfig(
                transport_type=TransportType.SSE,
                host="localhost",
                port=8765,
                max_connections=100,
                enable_batching=True,
                batch_timeout=0.1,
                max_batch_size=50
            )
        super().__init__(config)
        
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.clients: Dict[str, SSEClient] = {}
        self.client_counter = 0
        self._ping_task: Optional[asyncio.Task] = None
        
    async def start(self) -> None:
        """Start the SSE HTTP server."""
        try:
            # Create aiohttp application
            self.app = web.Application()
            
            # Add routes
            self.app.router.add_post('/mcp', self._handle_post)
            self.app.router.add_get('/mcp', self._handle_sse)
            self.app.router.add_get('/health', self._handle_health)
            self.app.router.add_get('/stats', self._handle_stats)
            
            # Start server
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            
            self.site = web.TCPSite(
                self.runner,
                self.config.host,
                self.config.port
            )
            await self.site.start()
            
            self.running = True
            
            # Start ping task to keep connections alive
            self._ping_task = asyncio.create_task(self._ping_clients())
            
            logger.info(f"SSE transport started on {self.config.host}:{self.config.port}")
            
        except Exception as e:
            logger.error(f"Failed to start SSE transport: {e}")
            raise
            
    async def stop(self) -> None:
        """Stop the SSE transport."""
        self.running = False
        
        # Cancel ping task
        if self._ping_task:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass
                
        # Close all client connections
        for client in list(self.clients.values()):
            await self._close_client(client)
            
        # Stop server
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
            
        logger.info("SSE transport stopped")
        
    async def send_message(self, message: JSONRPCMessage) -> None:
        """Send message to all connected clients."""
        if not self.clients:
            return
            
        # Serialize message
        try:
            if hasattr(message, 'model_dump'):
                data = message.model_dump()
            elif hasattr(message, 'dict'):
                data = message.dict()
            else:
                data = message
                
            json_str = json.dumps(data, separators=(',', ':'))
            
            # Send to all clients
            for client in list(self.clients.values()):
                try:
                    await client.queue.put(('message', json_str))
                except Exception as e:
                    logger.error(f"Failed to queue message for client {client.client_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to serialize message: {e}")
            self.stats['errors'] += 1
            
    async def send_batch(self, messages: List[JSONRPCMessage]) -> None:
        """Send batch of messages to all connected clients."""
        if not messages or not self.clients:
            return
            
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
            
            # Send to all clients
            for client in list(self.clients.values()):
                try:
                    await client.queue.put(('batch', json_str))
                except Exception as e:
                    logger.error(f"Failed to queue batch for client {client.client_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to serialize batch: {e}")
            self.stats['errors'] += 1
            
    async def receive_message(self) -> Optional[JSONRPCMessage]:
        """Not implemented for SSE (receive via POST)."""
        return None
        
    async def receive_batch(self) -> List[JSONRPCMessage]:
        """Not implemented for SSE (receive via POST)."""
        return []
        
    async def _handle_post(self, request: Request) -> Response:
        """Handle HTTP POST requests with JSON-RPC messages."""
        try:
            # Check content type
            content_type = request.headers.get('Content-Type', '')
            if 'application/json' not in content_type:
                return web.Response(
                    status=400,
                    text="Content-Type must be application/json"
                )
                
            # Parse request body
            try:
                data = await request.json()
            except Exception as e:
                return web.Response(
                    status=400,
                    text=f"Invalid JSON: {e}"
                )
                
            # Process message(s)
            messages = []
            if isinstance(data, list):
                # Batch request
                for item in data:
                    message = self._parse_message(item)
                    if message:
                        messages.append(message)
                self.stats['batches_received'] += 1
            else:
                # Single request
                message = self._parse_message(data)
                if message:
                    messages.append(message)
                    
            self.stats['messages_received'] += len(messages)
            
            # Check Accept header to determine response format
            accept = request.headers.get('Accept', '')
            
            if 'text/event-stream' in accept:
                # Client wants SSE stream - initiate SSE connection
                return await self._initiate_sse(request, messages)
            else:
                # Client wants JSON response
                # For now, return success - actual processing happens elsewhere
                return web.json_response({
                    'status': 'received',
                    'message_count': len(messages)
                })
                
        except Exception as e:
            logger.error(f"Error handling POST request: {e}")
            self.stats['errors'] += 1
            return web.Response(status=500, text=str(e))
            
    async def _handle_sse(self, request: Request) -> StreamResponse:
        """Handle SSE connection requests."""
        try:
            return await self._initiate_sse(request, [])
        except Exception as e:
            logger.error(f"Error handling SSE request: {e}")
            return web.Response(status=500, text=str(e))
            
    async def _initiate_sse(self, request: Request, initial_messages: List[JSONRPCMessage]) -> StreamResponse:
        """Initiate SSE connection."""
        # Create SSE response
        response = web.StreamResponse(
            status=200,
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
            }
        )
        
        await response.prepare(request)
        
        # Create client
        self.client_counter += 1
        client_id = f"client_{self.client_counter}"
        
        client = SSEClient(
            client_id=client_id,
            request=request,
            response=response,
            queue=asyncio.Queue(),
            last_ping=asyncio.get_event_loop().time()
        )
        
        self.clients[client_id] = client
        
        try:
            # Send initial messages if any
            for message in initial_messages:
                await self.send_message(message)
                
            # Send welcome message
            await self._send_sse_event(client, 'connected', {'client_id': client_id})
            
            # Process client queue
            while self.running and not response.transport.is_closing():
                try:
                    # Wait for message with timeout
                    event_type, data = await asyncio.wait_for(
                        client.queue.get(), timeout=30.0
                    )
                    
                    if event_type == 'ping':
                        await self._send_sse_event(client, 'ping', {'timestamp': data})
                        client.last_ping = asyncio.get_event_loop().time()
                    elif event_type in ('message', 'batch'):
                        await self._send_sse_event(client, event_type, data)
                    elif event_type == 'close':
                        break
                        
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    await client.queue.put(('ping', asyncio.get_event_loop().time()))
                    
        except Exception as e:
            logger.error(f"Error in SSE connection for {client_id}: {e}")
        finally:
            await self._close_client(client)
            
        return response
        
    async def _send_sse_event(self, client: SSEClient, event_type: str, data: Any) -> None:
        """Send SSE event to client."""
        try:
            if isinstance(data, str):
                data_str = data
            else:
                data_str = json.dumps(data, separators=(',', ':'))
                
            event = f"event: {event_type}\ndata: {data_str}\n\n"
            await client.response.write(event.encode('utf-8'))
            await client.response.drain()
            
        except Exception as e:
            logger.error(f"Failed to send SSE event to {client.client_id}: {e}")
            raise
            
    async def _close_client(self, client: SSEClient) -> None:
        """Close a client connection."""
        try:
            if client.client_id in self.clients:
                del self.clients[client.client_id]
                
            if not client.response.transport.is_closing():
                await client.response.write_eof()
                
        except Exception as e:
            logger.error(f"Error closing client {client.client_id}: {e}")
            
    async def _ping_clients(self) -> None:
        """Periodically ping clients to keep connections alive."""
        while self.running:
            try:
                await asyncio.sleep(30)  # Ping every 30 seconds
                
                current_time = asyncio.get_event_loop().time()
                stale_clients = []
                
                for client in self.clients.values():
                    if current_time - client.last_ping > 60:  # 60 second timeout
                        stale_clients.append(client)
                    else:
                        await client.queue.put(('ping', current_time))
                        
                # Close stale clients
                for client in stale_clients:
                    logger.info(f"Closing stale client: {client.client_id}")
                    await self._close_client(client)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ping task: {e}")
                
    async def _handle_health(self, request: Request) -> Response:
        """Health check endpoint."""
        return web.json_response({
            'status': 'healthy',
            'transport': 'sse',
            'running': self.running,
            'clients': len(self.clients)
        })
        
    async def _handle_stats(self, request: Request) -> Response:
        """Statistics endpoint."""
        stats = self.get_stats()
        stats['clients'] = {
            client_id: {
                'last_ping': client.last_ping,
                'queue_size': client.queue.qsize()
            }
            for client_id, client in self.clients.items()
        }
        return web.json_response(stats)
        
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