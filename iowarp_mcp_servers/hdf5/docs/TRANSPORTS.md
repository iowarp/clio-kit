# Transport Support

HDF5 MCP supports both MCP protocol transports for different use cases.

## stdio Transport (Default)

**Use case**: Local AI assistants (Claude Code, Cursor, VS Code)

### How it works
- Server launched as subprocess
- Communication via stdin/stdout
- Newline-delimited JSON-RPC
- Synchronous request-response

### Usage
```bash
# Default mode
uvx iowarp-mcps hdf5

# With data directory
uvx iowarp-mcps hdf5 --data-dir /path/to/data

# With logging
uvx iowarp-mcps hdf5 --log-level DEBUG
```

### Configuration
```json
{
  "mcpServers": {
    "hdf5-mcp": {
      "command": "uvx",
      "args": ["iowarp-mcps", "hdf5"]
    }
  }
}
```

### Characteristics
- ✅ Simple setup
- ✅ One client per server instance
- ✅ Automatic process lifecycle
- ✅ Works with all MCP clients
- ⚠️ Limited to memory-sized responses
- ⚠️ No streaming for multi-GB datasets

## SSE/HTTP Transport (Advanced)

**Use case**: Remote servers, streaming large datasets, multiple clients

### How it works
- Server runs as independent HTTP service
- Single `/mcp` endpoint (POST + GET)
- Server-Sent Events for streaming
- Session management for stateful connections
- Resumable streams with event IDs

### Usage
```bash
# Start SSE server
uvx iowarp-mcps hdf5 --transport sse --host localhost --port 8765

# Custom configuration
HDF5_TRANSPORT=sse \
HDF5_SSE_HOST=localhost \
HDF5_SSE_PORT=8765 \
uvx iowarp-mcps hdf5
```

### MCP Protocol Compliance (2025-06-18)

**Security features**:
- ✅ Origin validation (prevents DNS rebinding)
- ✅ Localhost-only binding by default
- ✅ Session management (Mcp-Session-Id header)
- ✅ Protocol version validation (MCP-Protocol-Version header)

**Streaming features**:
- ✅ SSE streams with event IDs
- ✅ Resumable streams (Last-Event-ID header)
- ✅ Event history (last 100 events per client)
- ✅ Multiple concurrent clients

**Endpoints**:
- `POST /mcp` - Send JSON-RPC messages
- `GET /mcp` - Open SSE stream for server→client messages
- `GET /health` - Health check
- `GET /stats` - Transport statistics

### Protocol Headers

**Client→Server**:
```http
POST /mcp HTTP/1.1
Content-Type: application/json
Accept: text/event-stream
MCP-Protocol-Version: 2025-06-18
Mcp-Session-Id: <uuid>

{json-rpc message}
```

**Server→Client**:
```http
HTTP/1.1 200 OK
Content-Type: text/event-stream
Mcp-Session-Id: <uuid>

id: client_1_42
event: message
data: {json-rpc response}

```

### Resumable Streams

Client disconnects can be resumed:

```http
GET /mcp HTTP/1.1
Mcp-Session-Id: <existing-session>
Last-Event-ID: client_1_42
```

Server replays events after `client_1_42`.

### Session Management

**Create session**:
```http
POST /mcp
Content-Type: application/json

{"jsonrpc":"2.0","id":1,"method":"initialize",...}

→ Response includes: Mcp-Session-Id: <uuid>
```

**Use session**:
```http
POST /mcp
Mcp-Session-Id: <uuid>
Content-Type: application/json

{subsequent requests}
```

**Terminate session**:
```http
DELETE /mcp
Mcp-Session-Id: <uuid>
```

### Characteristics
- ✅ Streams unlimited data sizes
- ✅ Multiple concurrent clients
- ✅ Server-initiated messages
- ✅ Resumable connections
- ✅ Session state management
- ⚠️ More complex setup
- ⚠️ Requires network configuration

## When to Use Which

### Use stdio (Default) when:
- Running locally with AI coding assistants
- One client per server instance
- Datasets < 1GB
- Simple request-response workflows

### Use SSE when:
- Streaming multi-GB datasets
- Multiple clients need access
- Server needs to push notifications
- Running as remote service
- Need resumable connections

## Configuration Reference

### Environment Variables
```bash
# Transport selection
HDF5_TRANSPORT=stdio              # stdio (default) or sse

# SSE transport options
HDF5_SSE_HOST=127.0.0.1           # Localhost for security
HDF5_SSE_PORT=8765                # Port number
HDF5_MAX_CONNECTIONS=100          # Max concurrent clients

# Performance
HDF5_CACHE_SIZE=1000              # LRU cache capacity
HDF5_NUM_WORKERS=4                # Parallel worker count
HDF5_ENABLE_BATCHING=true         # Message batching
```

### CLI Arguments
```bash
--transport {stdio,sse}           # Transport type
--host HOST                       # SSE host (default: 127.0.0.1)
--port PORT                       # SSE port (default: 8765)
--data-dir PATH                   # HDF5 files directory
--log-level {DEBUG,INFO,WARNING}  # Logging level
```

## Security Considerations

### stdio
- ✅ Inherently secure (local only)
- ✅ Process isolation
- ✅ No network exposure

### SSE
- ✅ Origin validation enforced
- ✅ Localhost-only by default
- ⚠️ Session IDs must remain secret
- ⚠️ No authentication (implement if needed)
- ⚠️ HTTPS recommended for remote use

## Examples

### stdio Example
```bash
# Start server (stdio mode)
uvx iowarp-mcps hdf5 --data-dir ./data

# Server communicates via stdin/stdout automatically
```

### SSE Example
```bash
# Start server (SSE mode)
uvx iowarp-mcps hdf5 --transport sse --port 8765

# In another terminal, test endpoints:
curl http://localhost:8765/health
curl http://localhost:8765/stats

# Open SSE stream
curl -N -H "Accept: text/event-stream" \
     -H "MCP-Protocol-Version: 2025-06-18" \
     http://localhost:8765/mcp

# Send message
curl -X POST http://localhost:8765/mcp \
     -H "Content-Type: application/json" \
     -H "Accept: text/event-stream" \
     -H "MCP-Protocol-Version: 2025-06-18" \
     -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

## Protocol Compliance

This implementation follows **MCP Protocol Specification 2025-06-18**.

### stdio
- ✅ Newline-delimited JSON-RPC
- ✅ UTF-8 encoding
- ✅ No embedded newlines
- ✅ stderr for logging only

### SSE/HTTP
- ✅ Single MCP endpoint (POST + GET)
- ✅ Session management (Mcp-Session-Id)
- ✅ Resumable streams (Last-Event-ID)
- ✅ Protocol version validation
- ✅ Origin validation
- ✅ Proper response codes (202 for notifications/responses)
- ✅ Event IDs on all SSE messages

## References

- [MCP Protocol Specification](https://modelcontextprotocol.io/specification)
- [stdio Transport](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#stdio)
- [Streamable HTTP Transport](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#streamable-http)
