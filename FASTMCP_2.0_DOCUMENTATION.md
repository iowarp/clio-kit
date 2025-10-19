# FastMCP 2.0 Comprehensive Documentation & Migration Guide

**Generated**: October 18, 2025
**Purpose**: Complete FastMCP 2.0 patterns and HDF5 MCP migration reference
**Sources**: gofastmcp.com, GitHub (jlowin/fastmcp), PyPI, Medium tutorials

---

## Table of Contents

1. [FastMCP 2.0 Overview](#fastmcp-20-overview)
2. [Core Decorator Patterns](#core-decorator-patterns)
3. [Server Initialization & Running](#server-initialization--running)
4. [Tool Definition Patterns](#tool-definition-patterns)
5. [Resource URI Patterns](#resource-uri-patterns)
6. [Prompt Template Patterns](#prompt-template-patterns)
7. [Transport Configuration](#transport-configuration)
8. [Pydantic Integration](#pydantic-integration)
9. [Async Patterns & Best Practices](#async-patterns--best-practices)
10. [Context Dependency Injection](#context-dependency-injection)
11. [Server Composition](#server-composition)
12. [Schema Generation](#schema-generation)
13. [HDF5 MCP Migration Strategy](#hdf5-mcp-migration-strategy)

---

## FastMCP 2.0 Overview

### What is FastMCP?

FastMCP 2.0 is the actively maintained, production-ready framework for building Model Context Protocol (MCP) servers and clients in Python. It provides:

- **Pythonic API**: Decorator-based tool/resource/prompt registration
- **Auto-schema Generation**: Type hints → JSON Schema automatically
- **Multi-transport**: stdio, HTTP, SSE, FastAPI, in-memory
- **Advanced Features**: Server composition, OAuth, OpenAPI integration
- **Production Ready**: Enterprise auth, testing utilities, deployment tools

### Installation

```bash
pip install fastmcp>=2.0.0
# or with uv
uv add fastmcp
```

### Latest Version

As of October 2025: **FastMCP 2.2.7** (actively maintained)

---

## Core Decorator Patterns

### 1. @mcp.tool() - Function → AI Tool

**Pattern**: Transform Python functions into AI-executable tools

```python
from fastmcp import FastMCP

mcp = FastMCP("ServerName")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
async def fetch_data(url: str) -> dict:
    """Fetch data from URL (async version)"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

**Key Points**:
- Function name becomes tool name
- Docstring becomes tool description
- Type hints generate parameter schema
- Both `def` and `async def` supported
- Return type defines output schema

### 2. @mcp.resource() - Dynamic Data Access

**Pattern**: Expose data through URI-based resources

```python
# Static resource
@mcp.resource("config://settings")
def get_settings() -> dict:
    """Server configuration"""
    return {"version": "1.0", "mode": "production"}

# Dynamic resource with URI parameters
@mcp.resource("user://{user_id}/profile")
def get_user_profile(user_id: str) -> dict:
    """Get user profile by ID"""
    return {"user_id": user_id, "name": "Alice"}

# Resource with query parameters
@mcp.resource("search://results{?query,limit}")
def search_results(query: str, limit: int = 10) -> list:
    """Search with optional limit"""
    return perform_search(query, limit)
```

**URI Template Rules**:
- Required parameters: `{param}` in path (no default value in function)
- Query parameters: `{?param}` syntax (must have default value)
- All URI parameters must exist as function parameters

### 3. @mcp.prompt() - Reusable Templates

**Pattern**: Create parameterized prompt templates

```python
from fastmcp import FastMCP
from fastmcp.prompts import UserMessage, AssistantMessage

# Simple string prompt
@mcp.prompt()
def explain_topic(topic: str) -> str:
    """Generate explanation prompt"""
    return f"Explain {topic} in simple terms for beginners"

# Multi-message prompt
@mcp.prompt()
def debug_error(error: str, code: str) -> list:
    """Debug assistance prompt"""
    return [
        UserMessage(f"I encountered this error:\n{error}"),
        UserMessage(f"In this code:\n```\n{code}\n```"),
        AssistantMessage("I'll help debug that. Let me analyze...")
    ]

# Complex typed parameters
@mcp.prompt()
def analyze_data(
    numbers: list[int],
    threshold: float,
    metadata: dict[str, str]
) -> str:
    """Analyze numerical data"""
    avg = sum(numbers) / len(numbers)
    return f"Data analysis: avg={avg}, threshold={threshold}, meta={metadata}"
```

---

## Server Initialization & Running

### Basic Pattern

```python
from fastmcp import FastMCP

# 1. Create server instance
mcp = FastMCP("My Server Name")

# 2. Define tools/resources/prompts with decorators
@mcp.tool()
def my_tool(param: str) -> dict:
    """Tool description"""
    return {"result": "success"}

# 3. Run server (typically in if __name__ == "__main__")
if __name__ == "__main__":
    mcp.run()  # Uses stdio transport by default
```

### Complete Server Template

```python
#!/usr/bin/env python3
"""My FastMCP Server"""

from fastmcp import FastMCP, Context
import asyncio
from pathlib import Path

# Initialize server
mcp = FastMCP(
    "MyServer",
    version="1.0.0"
)

# ============= Tools =============
@mcp.tool()
async def tool_one(filepath: str, ctx: Context) -> dict:
    """First tool with context"""
    await ctx.info(f"Processing {filepath}")
    return {"status": "success"}

@mcp.tool()
def tool_two(value: int, multiplier: float = 2.0) -> float:
    """Simple sync tool"""
    return value * multiplier

# ============= Resources =============
@mcp.resource("config://app")
def get_config() -> dict:
    """Application configuration"""
    return {"debug": True, "version": "1.0"}

@mcp.resource("file://{path}")
async def read_file(path: str) -> str:
    """Read file contents"""
    return Path(path).read_text()

# ============= Prompts =============
@mcp.prompt()
def analyze_code(code: str) -> str:
    """Code analysis prompt"""
    return f"Analyze this code:\n```\n{code}\n```"

# ============= Main Entry Point =============
def main():
    """Main entry point"""
    # Default: stdio transport
    mcp.run()

    # Or specify transport
    # mcp.run(transport="http", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
```

---

## Tool Definition Patterns

### 1. Simple Synchronous Tool

```python
@mcp.tool()
def calculate_sum(numbers: list[float]) -> float:
    """Calculate sum of numbers"""
    return sum(numbers)
```

### 2. Async Tool (I/O Operations)

```python
@mcp.tool()
async def read_file(filepath: str) -> str:
    """Read file asynchronously"""
    async with aiofiles.open(filepath, 'r') as f:
        return await f.read()
```

### 3. Tool with Optional Parameters

```python
@mcp.tool()
def search(query: str, limit: int = 10, offset: int = 0) -> list:
    """Search with pagination"""
    return perform_search(query, limit, offset)
```

### 4. Tool with Complex Return Types

```python
from pydantic import BaseModel

class SearchResult(BaseModel):
    title: str
    url: str
    score: float

@mcp.tool()
def search_advanced(query: str) -> list[SearchResult]:
    """Search returning Pydantic models"""
    results = perform_search(query)
    return [SearchResult(**r) for r in results]
```

### 5. Tool with Context (Logging, Progress)

```python
@mcp.tool()
async def long_task(items: list[str], ctx: Context) -> dict:
    """Process items with progress updates"""
    await ctx.info(f"Processing {len(items)} items")

    results = []
    for i, item in enumerate(items):
        result = await process_item(item)
        results.append(result)

        # Progress update
        progress = (i + 1) / len(items) * 100
        await ctx.info(f"Progress: {progress:.1f}%")

    return {"processed": len(results), "results": results}
```

### 6. Tool with Error Handling

```python
@mcp.tool()
async def safe_operation(data: str) -> dict:
    """Operation with proper error handling"""
    try:
        result = await process_data(data)
        return {
            "status": "success",
            "result": result,
            "_meta": {"tool": "safe_operation"}
        }
    except FileNotFoundError as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": "FileNotFoundError",
            "isError": True
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "isError": True
        }
```

---

## Resource URI Patterns

### Pattern Types

FastMCP uses URI templates following RFC 6570 for dynamic resources.

### 1. Static Resources

```python
@mcp.resource("config://server")
def server_config() -> dict:
    """Static server configuration"""
    return {"version": "1.0", "mode": "production"}
```

### 2. Path Parameters (Required)

```python
# Single parameter
@mcp.resource("user://{user_id}")
def get_user(user_id: str) -> dict:
    """Get user by ID"""
    return {"id": user_id, "name": users[user_id]}

# Multiple parameters
@mcp.resource("db://{database}/{table}")
def get_table(database: str, table: str) -> dict:
    """Access database table"""
    return fetch_table(database, table)

# Nested paths
@mcp.resource("file://{directory}/{filename}")
def read_file(directory: str, filename: str) -> str:
    """Read file from directory"""
    path = Path(directory) / filename
    return path.read_text()
```

### 3. Query Parameters (Optional)

```python
# Single query parameter
@mcp.resource("search://results{?query}")
def search(query: str = "") -> list:
    """Search with optional query"""
    return perform_search(query) if query else []

# Multiple query parameters
@mcp.resource("api://data{?limit,offset,sort}")
def get_data(limit: int = 10, offset: int = 0, sort: str = "asc") -> list:
    """Paginated data access"""
    return fetch_data(limit=limit, offset=offset, sort=sort)
```

### 4. Mixed Path and Query Parameters

```python
@mcp.resource("user://{user_id}/posts{?limit,tag}")
def user_posts(user_id: str, limit: int = 10, tag: str = None) -> list:
    """User posts with optional filtering"""
    posts = fetch_user_posts(user_id, limit)
    if tag:
        posts = [p for p in posts if tag in p.get("tags", [])]
    return posts
```

### 5. Multiple URI Templates for Same Function

```python
# Apply decorator multiple times for different access patterns
@mcp.resource("user://{user_id}/email")
@mcp.resource("email://{email_address}")
def get_user_email(user_id: str = None, email_address: str = None) -> str:
    """Access email by user ID or email address"""
    if user_id:
        return users[user_id]["email"]
    elif email_address:
        return email_address
```

### Type Conversion Rules

FastMCP automatically converts URI parameter strings to typed values:

- `int`: "123" → 123
- `float`: "3.14" → 3.14
- `bool`: "true" → True, "false" → False
- `str`: No conversion
- Complex types (list, dict, Pydantic models): JSON deserialization

---

## Prompt Template Patterns

### 1. Simple String Prompt

```python
@mcp.prompt()
def review_code(code: str) -> str:
    """Code review prompt"""
    return f"Please review this code:\n\n{code}"
```

### 2. Multi-Message Conversation

```python
from fastmcp.prompts import UserMessage, AssistantMessage

@mcp.prompt()
def interactive_help(topic: str) -> list:
    """Interactive help session"""
    return [
        UserMessage(f"I need help with {topic}"),
        AssistantMessage("I'll help you with that. What specifically would you like to know?"),
        UserMessage("Can you provide examples?")
    ]
```

### 3. Prompt with Complex Parameters

```python
@mcp.prompt()
def analyze_dataset(
    columns: list[str],
    row_count: int,
    metadata: dict[str, str]
) -> str:
    """Dataset analysis prompt"""
    cols = ", ".join(columns)
    meta_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
    return f"""Analyze this dataset:
- Columns: {cols}
- Rows: {row_count}
- Metadata: {meta_str}

What insights can you provide?"""
```

### 4. Prompt with Context Injection

```python
@mcp.prompt()
async def contextual_prompt(query: str, ctx: Context) -> str:
    """Prompt with server context"""
    # Access server resources
    config = await ctx.read_resource("config://app")

    return f"""Query: {query}
Server Mode: {config.get('mode', 'unknown')}

Please answer based on current server configuration."""
```

### 5. Template with Conditional Logic

```python
@mcp.prompt()
def adaptive_prompt(difficulty: str, topic: str) -> str:
    """Adapt prompt based on difficulty"""
    if difficulty == "beginner":
        return f"Explain {topic} in very simple terms for beginners"
    elif difficulty == "advanced":
        return f"Provide advanced technical details about {topic}"
    else:
        return f"Explain {topic} at an intermediate level"
```

---

## Transport Configuration

FastMCP supports multiple transport mechanisms for different deployment scenarios.

### 1. STDIO Transport (Default)

**Use Case**: Local AI assistants (Claude Desktop, Cursor, VS Code)

```python
# Simplest - uses stdio by default
mcp.run()

# Explicit
mcp.run(transport="stdio")
```

**Client Configuration** (Claude Desktop config):
```json
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["/path/to/server.py"]
    }
  }
}
```

### 2. HTTP Transport (Streamable HTTP)

**Use Case**: Remote servers, network access, multiple clients

```python
# Basic HTTP server
mcp.run(transport="http", host="127.0.0.1", port=8000)

# Public access
mcp.run(transport="streamable-http", host="0.0.0.0", port=8080)
```

**Client Connection**:
```python
from fastmcp.client.transports import StreamableHttpTransport

transport = StreamableHttpTransport(
    url="http://localhost:8000",
    headers={"Authorization": "Bearer token"}
)
client = Client(transport)
```

### 3. SSE Transport (Server-Sent Events)

**Use Case**: Streaming large datasets, real-time updates

```python
# SSE server
mcp.run(transport="sse", host="0.0.0.0", port=8765)
```

**Client Connection**:
```python
from fastmcp.client.transports import SSETransport

transport = SSETransport(
    url="http://localhost:8765/sse",
    headers={"Authorization": "Bearer token"}
)
client = Client(transport)
```

### 4. In-Memory Transport (Testing)

**Use Case**: Unit tests, development, same-process communication

```python
from fastmcp.client.transports import InMemoryTransport

# Direct server instance access (no network, no subprocess)
transport = InMemoryTransport(mcp)
client = Client(transport)
```

### 5. FastAPI Integration

**Use Case**: Integrate with existing FastAPI apps

```python
from fastapi import FastAPI
from fastmcp import FastMCP

app = FastAPI()
mcp = FastMCP("MyServer")

# Add MCP tools
@mcp.tool()
def my_tool() -> str:
    return "Hello"

# Mount to FastAPI
app.mount("/mcp", mcp.as_fastapi())

# Run with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 6. CLI Usage

FastMCP provides a CLI for running servers:

```bash
# Auto-detect and run
fastmcp run server.py

# Specify transport
fastmcp run server.py --transport http --port 8000

# Development mode with hot reload
fastmcp dev server.py --log-level DEBUG

# Production mode
fastmcp run server.py --transport sse --host 0.0.0.0 --port 8765
```

---

## Pydantic Integration

FastMCP has first-class Pydantic support for complex parameters and return types.

### 1. Pydantic Models as Tool Parameters

```python
from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(10, ge=1, le=100)
    include_archived: bool = False

@mcp.tool()
def search(request: SearchRequest) -> list[dict]:
    """Search with validated parameters"""
    return perform_search(
        request.query,
        request.max_results,
        request.include_archived
    )
```

### 2. Pydantic Models as Return Types

```python
class User(BaseModel):
    id: str
    name: str
    email: str
    created_at: datetime

@mcp.tool()
def get_user(user_id: str) -> User:
    """Get user with typed response"""
    user_data = fetch_user(user_id)
    return User(**user_data)
```

### 3. Nested Pydantic Models

```python
class Address(BaseModel):
    street: str
    city: str
    country: str

class UserProfile(BaseModel):
    user: User
    address: Address
    preferences: dict[str, Any]

@mcp.tool()
def get_full_profile(user_id: str) -> UserProfile:
    """Get complete user profile"""
    return fetch_profile(user_id)
```

### 4. Annotated Fields with Validation

```python
from typing import Annotated

@mcp.tool()
def process_data(
    value: Annotated[int, Field(ge=0, le=100, description="Value between 0-100")],
    name: Annotated[str, Field(min_length=3, max_length=50)],
    tags: Annotated[list[str], Field(max_items=10)]
) -> dict:
    """Process data with validation"""
    return {"value": value, "name": name, "tags": tags}
```

### 5. Automatic JSON Serialization

Pydantic models and complex types are automatically serialized:

```python
@mcp.resource("stats://summary")
def get_stats() -> dict:
    """Returns dict - auto-serialized to JSON"""
    return {"count": 100, "average": 45.2}

@mcp.resource("users://all")
def get_users() -> list[User]:
    """Returns Pydantic models - auto-serialized"""
    return [User(id="1", name="Alice", email="alice@example.com")]
```

**MIME Types**:
- `dict`, `list`, `Pydantic models` → `application/json`
- Strings → `text/plain`
- Custom types can override serialization

---

## Async Patterns & Best Practices

### When to Use Async

**Use `async def` for**:
- File I/O operations
- Database queries
- HTTP/API calls
- Any network communication
- Long-running operations

**Use `def` for**:
- Pure computation (CPU-bound)
- Quick synchronous operations
- No I/O involved

### 1. Basic Async Tool

```python
@mcp.tool()
async def read_file(path: str) -> str:
    """Async file reading"""
    async with aiofiles.open(path, 'r') as f:
        return await f.read()
```

### 2. Concurrent Operations with asyncio.gather

```python
@mcp.tool()
async def batch_fetch(urls: list[str]) -> list[dict]:
    """Fetch multiple URLs concurrently"""
    async def fetch_one(url: str) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.json()

    # Run all fetches concurrently
    results = await asyncio.gather(*[fetch_one(url) for url in urls])
    return results
```

### 3. Async with Context for Progress

```python
@mcp.tool()
async def process_items(items: list[str], ctx: Context) -> dict:
    """Process items with progress tracking"""
    await ctx.info(f"Starting processing of {len(items)} items")

    results = []
    for i, item in enumerate(items):
        result = await process_single_item(item)
        results.append(result)

        # Progress updates
        if (i + 1) % 10 == 0:
            await ctx.info(f"Processed {i + 1}/{len(items)}")

    await ctx.info("Processing complete")
    return {"count": len(results), "results": results}
```

### 4. Handling Blocking Operations

For CPU-intensive or blocking sync operations, use `anyio` (bundled with FastMCP):

```python
import anyio

@mcp.tool()
async def cpu_intensive_task(data: list[int]) -> float:
    """CPU-intensive calculation wrapped as async"""
    def blocking_calculation(numbers):
        # Expensive computation
        return sum(x ** 2 for x in numbers) / len(numbers)

    # Run in thread pool to not block event loop
    result = await anyio.to_thread.run_sync(blocking_calculation, data)
    return result
```

### 5. Resource Cleanup with Context Managers

```python
@mcp.tool()
async def process_database_query(query: str) -> list[dict]:
    """Database query with proper cleanup"""
    async with get_db_connection() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(query)
            results = await cursor.fetchall()
            return [dict(row) for row in results]
```

### Best Practices Summary

1. **Always use async for I/O** - File, network, database operations
2. **Use asyncio.gather for concurrency** - Parallel I/O operations
3. **Wrap blocking code with anyio.to_thread** - CPU-intensive sync code
4. **Use context managers for cleanup** - `async with` for resources
5. **Report progress for long tasks** - Use `ctx.info()` for user feedback
6. **Handle errors gracefully** - Try/except with meaningful error responses
7. **Avoid blocking the event loop** - Never use `time.sleep()`, use `await asyncio.sleep()`

---

## Context Dependency Injection

### Overview

The `Context` object provides access to server capabilities and is automatically injected into tools, resources, and prompts.

### Basic Usage

```python
from fastmcp import FastMCP, Context

mcp = FastMCP("MyServer")

@mcp.tool()
async def my_tool(data: str, ctx: Context) -> dict:
    """Tool with context access"""
    # Context is automatically injected
    await ctx.info("Processing data...")
    return {"status": "success"}
```

**Key Rules**:
- Parameter name doesn't matter (e.g., `ctx`, `context`, `c`)
- Only the type hint `Context` matters
- Context parameter is **not exposed** to MCP clients
- Can be placed anywhere in function signature

### Context Methods

#### 1. Logging

```python
@mcp.tool()
async def process(data: str, ctx: Context) -> dict:
    """Logging examples"""
    await ctx.info("Processing started")
    await ctx.debug(f"Data length: {len(data)}")
    await ctx.warning("Large dataset detected")
    await ctx.error("Validation failed")
    return {}
```

#### 2. LLM Sampling (Request LLM Completions)

```python
@mcp.tool()
async def generate_summary(text: str, ctx: Context) -> str:
    """Use client's LLM for completion"""
    prompt = f"Summarize this text:\n\n{text}"

    response = await ctx.sample(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )

    return response.content
```

#### 3. Resource Access

```python
@mcp.tool()
async def process_with_config(data: str, ctx: Context) -> dict:
    """Access server resources from tool"""
    # Read a resource
    config = await ctx.read_resource("config://app")

    # Use config in processing
    mode = config.get("mode", "default")
    return process_data(data, mode=mode)
```

#### 4. HTTP Requests

```python
@mcp.tool()
async def fetch_external_data(api_url: str, ctx: Context) -> dict:
    """Make HTTP requests via context"""
    response = await ctx.http_request(
        method="GET",
        url=api_url,
        headers={"Authorization": "Bearer token"}
    )
    return response.json()
```

#### 5. Progress Reporting

```python
@mcp.tool()
async def long_task(items: list[str], ctx: Context) -> dict:
    """Report progress for long-running tasks"""
    total = len(items)

    for i, item in enumerate(items):
        await process_item(item)

        # Report progress
        await ctx.report_progress(
            progress=i + 1,
            total=total,
            message=f"Processed {i+1}/{total}"
        )

    return {"processed": total}
```

### Advanced: get_context() Function

Access context from nested functions without passing it explicitly:

```python
from fastmcp.server.dependencies import get_context

async def helper_function(data: str):
    """Helper can access context without explicit parameter"""
    ctx = get_context()
    await ctx.info(f"Helper processing {len(data)} bytes")
    return process(data)

@mcp.tool()
async def main_tool(data: str, ctx: Context) -> dict:
    """Main tool delegates to helper"""
    # Helper can access ctx via get_context()
    result = await helper_function(data)
    return {"result": result}
```

### Context in Resources

```python
@mcp.resource("data://{dataset_id}")
async def get_dataset(dataset_id: str, ctx: Context) -> dict:
    """Resource with context for logging"""
    await ctx.info(f"Fetching dataset: {dataset_id}")
    data = fetch_dataset(dataset_id)
    return data
```

### Context in Prompts

```python
@mcp.prompt()
async def contextual_prompt(query: str, ctx: Context) -> str:
    """Prompt that uses server state"""
    # Access current server configuration
    config = await ctx.read_resource("config://app")
    version = config.get("version", "unknown")

    return f"""Query: {query}
Server Version: {version}

Please answer based on current server configuration."""
```

---

## Server Composition

FastMCP allows combining multiple servers into larger applications.

### Two Composition Methods

1. **`import_server()`** - Static copy (one-time)
2. **`mount()`** - Live link (dynamic delegation)

### 1. Import Server (Static Composition)

Copies all components from one server to another:

```python
from fastmcp import FastMCP

# Create specialized servers
hdf5_server = FastMCP("HDF5")
parquet_server = FastMCP("Parquet")
pandas_server = FastMCP("Pandas")

# Define tools in each
@hdf5_server.tool()
def read_hdf5(path: str) -> dict:
    """Read HDF5 file"""
    return h5py_read(path)

@parquet_server.tool()
def read_parquet(path: str) -> dict:
    """Read Parquet file"""
    return pq_read(path)

# Combine into main server
main_server = FastMCP("DataServer")
main_server.import_server(hdf5_server, prefix="hdf5_")
main_server.import_server(parquet_server, prefix="parquet_")

# Run combined server
if __name__ == "__main__":
    main_server.run()
```

**Result**: Tools become `hdf5_read_hdf5`, `parquet_read_parquet`

### 2. Mount Server (Dynamic Composition)

Creates live link where requests are delegated:

```python
# Create sub-servers
auth_server = FastMCP("AuthServer")
data_server = FastMCP("DataServer")

@auth_server.tool()
def login(username: str, password: str) -> dict:
    """Authenticate user"""
    return authenticate(username, password)

@data_server.tool()
def fetch_data(user_id: str) -> dict:
    """Fetch user data"""
    return get_data(user_id)

# Mount sub-servers to main server
main_server = FastMCP("MainServer")
main_server.mount("/auth", auth_server)
main_server.mount("/data", data_server)

# Run main server
if __name__ == "__main__":
    main_server.run()
```

### Mount Modes

#### Direct Mounting (Default)

Parent server directly accesses mounted server's objects:

```python
main_server.mount("/sub", sub_server)  # Direct access
```

#### Proxy Mounting

Parent treats mounted server as separate entity:

```python
main_server.mount("/sub", sub_server, mode="proxy")  # Client interface
```

### Use Cases

- **Modularity**: Break large apps into focused servers
- **Reusability**: Common utility servers mounted everywhere
- **Team Collaboration**: Different teams work on separate servers
- **Organization**: Group related functionality logically

### Example: Multi-Domain Server

```python
# Domain-specific servers
file_ops = FastMCP("FileOps")
db_ops = FastMCP("DatabaseOps")
api_ops = FastMCP("APIOperations")

# Define tools for each domain
@file_ops.tool()
def read_file(path: str) -> str:
    return Path(path).read_text()

@db_ops.tool()
async def query_db(sql: str) -> list:
    return await execute_query(sql)

@api_ops.tool()
async def call_api(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

# Unified server
unified = FastMCP("UnifiedServer")
unified.import_server(file_ops, prefix="file_")
unified.import_server(db_ops, prefix="db_")
unified.import_server(api_ops, prefix="api_")

# Now exposes: file_read_file, db_query_db, api_call_api
```

---

## Schema Generation

FastMCP automatically generates JSON Schema from Python type hints.

### Automatic Type Conversion

FastMCP inspects function signatures and converts type hints to JSON Schema:

```python
@mcp.tool()
def example(
    name: str,
    age: int,
    score: float,
    active: bool,
    tags: list[str],
    metadata: dict[str, Any]
) -> dict:
    """All common types handled automatically"""
    return {}
```

**Generated Schema**:
```json
{
  "name": {"type": "string"},
  "age": {"type": "integer"},
  "score": {"type": "number"},
  "active": {"type": "boolean"},
  "tags": {"type": "array", "items": {"type": "string"}},
  "metadata": {"type": "object"}
}
```

### Optional vs Required Parameters

```python
@mcp.tool()
def search(
    query: str,              # Required (no default)
    limit: int = 10,         # Optional (has default)
    offset: int = 0          # Optional (has default)
) -> list:
    """Schema marks query as required, limit/offset as optional"""
    return []
```

### Pydantic Field Annotations

Add descriptions and validation:

```python
from pydantic import Field
from typing import Annotated

@mcp.tool()
def create_user(
    name: Annotated[str, Field(min_length=3, max_length=50, description="User's full name")],
    age: Annotated[int, Field(ge=0, le=150, description="User's age in years")],
    email: Annotated[str, Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$", description="Valid email address")]
) -> dict:
    """Creates user with validated inputs"""
    return {}
```

### Output Schema Generation

Return type hints generate output schemas:

```python
class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    created_at: datetime

@mcp.tool()
def get_user(user_id: str) -> UserResponse:
    """Output schema generated from UserResponse model"""
    return UserResponse(
        id=user_id,
        name="Alice",
        email="alice@example.com",
        created_at=datetime.now()
    )
```

### Validation Behavior

**Default**: Pydantic's flexible validation (coercion)
- `"10"` → `10` (string to int)
- `"3.14"` → `3.14` (string to float)
- `"true"` → `True` (string to bool)

**Strict Mode** (if needed):
```python
from pydantic import Field, ValidationError

@mcp.tool()
def strict_tool(value: Annotated[int, Field(strict=True)]) -> int:
    """Only accepts actual integers, no coercion"""
    return value * 2
```

### Complex Type Support

```python
from typing import Union, Literal, Optional

@mcp.tool()
def complex_types(
    mode: Literal["fast", "accurate", "balanced"],
    data: Union[str, list[int]],
    config: Optional[dict] = None
) -> dict:
    """Complex types generate appropriate JSON Schema"""
    return {}
```

---

## HDF5 MCP Migration Strategy

### Current Architecture Analysis

**Current State**: HDF5 MCP v2.0.2 using standard MCP SDK
- Server class: `HDF5Server` with manual handler registration
- Tool registry: Custom `ToolRegistry` with decorator-based registration
- Resources: `ResourceManager` for file handles and caching
- Prompts: Template-based with Jinja2
- Transports: Custom stdio and SSE implementations
- 25+ tools with advanced features

**Estimated Codebase**:
- Server: ~470 lines
- Tools: ~200+ lines per tool with registry
- Resources: ~150 lines
- Prompts: ~100 lines
- Total: ~1500+ lines

### Migration Benefits

**Expected Outcomes**:
- **60-70% code reduction** - Eliminate boilerplate
- **Simplified transports** - Built-in stdio/HTTP/SSE
- **Better maintainability** - Standard FastMCP patterns
- **Future-proof** - Access to FastMCP 2.0 features

### Migration Approach

#### Option A: Full Migration (Recommended for Future)

**Scope**: Complete rewrite to FastMCP
**Effort**: 12-16 hours
**Risk**: Medium (requires extensive testing)

**Not Recommended Now**: HDF5 v2.0.2 is production-ready and stable

#### Option B: Incremental Hybrid (Best for HDF5)

**Scope**: Keep current implementation, use FastMCP for new features
**Effort**: 2-4 hours
**Risk**: Very Low

**Recommended**: Defer full migration to Q1 2026

### Code Comparison: Before/After

#### Server Initialization

**Before (Current - Standard SDK)**:
```python
class HDF5Server:
    def __init__(self):
        self.config = get_config()
        self.server = Server(name="HDF5 MCP Server")
        self._handlers_registered = False
        self.status = ServerStatus()
        self.resource_manager = ResourceManager()
        self.tools = create_tools()
        self.task_queue = AsyncTaskQueue(...)
        self._register_handlers()

    def _register_handlers(self):
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            return get_tools()

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict) -> List[TextContent]:
            result = await self.tools.__getattribute__(name)(**arguments or {})
            return result

        # ... more handlers
```

**After (FastMCP)**:
```python
from fastmcp import FastMCP, Context

mcp = FastMCP("HDF5")

# All tools defined directly with @mcp.tool()
# No manual handler registration needed
```

#### Tool Definition

**Before (Current)**:
```python
class ToolRegistry:
    @classmethod
    def register(cls, category: str = "general", ...):
        def decorator(func: Callable) -> Callable:
            # Extract parameter info from type hints
            sig = inspect.signature(func)
            param_info = {}
            type_hints = get_type_hints(func)

            for param_name, param in sig.parameters.items():
                if param_name == 'self' or param_name == 'cls':
                    continue
                param_type = type_hints.get(param_name, Any).__name__
                param_info[param_name] = {
                    "type": param_type,
                    "required": param.default == inspect.Parameter.empty
                }

            # Register tool manually
            cls._tools[tool_name] = {...}
            return func
        return decorator

class HDF5Tools:
    @ToolRegistry.register(category="file", description="Open HDF5 file")
    async def open_file(self, filename: str, mode: str = "r") -> List[TextContent]:
        # Implementation
        return [TextContent(text=json.dumps(result))]
```

**After (FastMCP)**:
```python
@mcp.tool()
async def open_file(filename: str, mode: str = "r") -> dict:
    """Open HDF5 file"""
    # Implementation
    return result  # Auto-serialized
```

**Reduction**: ~50 lines → ~5 lines per tool

#### Resource Definition

**Before (Current)**:
```python
class ResourceManager:
    def __init__(self):
        self.file_handles = {}
        self.cache = LRUCache(maxsize=100)

    async def read_resource(self, uri: str):
        # Parse URI manually
        # Handle caching manually
        # Return TextContent manually
        pass

def get_resources() -> List[Resource]:
    return [
        Resource(
            uri="hdf5://file/{filename}",
            name="HDF5 File",
            description="..."
        )
    ]
```

**After (FastMCP)**:
```python
@mcp.resource("hdf5://file/{filename}")
async def get_hdf5_file(filename: str) -> dict:
    """HDF5 file resource"""
    # URI parsing automatic
    # Caching can be added with decorator
    return file_info  # Auto-serialized
```

#### Transport Configuration

**Before (Current)**:
```python
# Custom stdio transport
async with stdio_server() as (read_stream, write_stream):
    await self.server.run(read_stream, write_stream, InitializationOptions(...))

# Custom SSE transport (separate implementation)
# mcps/HDF5/src/hdf5_mcp/transports/sse_transport.py (~200 lines)
```

**After (FastMCP)**:
```python
# Stdio
mcp.run()

# SSE
mcp.run(transport="sse", port=8765)

# HTTP
mcp.run(transport="http", port=8000)
```

### Example: Migrating One Tool

**Before (read_full_dataset)**:
```python
class HDF5Tools:
    @ToolRegistry.register(
        category="dataset",
        description="Read complete dataset from HDF5 file"
    )
    async def read_full_dataset(
        self,
        path: str,
        filename: Optional[str] = None
    ) -> List[TextContent]:
        """Read complete dataset from HDF5 file."""
        try:
            file = self._ensure_file_open(filename)

            if path not in file:
                return [TextContent(
                    text=json.dumps({
                        "error": f"Dataset {path} not found",
                        "isError": True
                    })
                )]

            dataset = file[path]
            data = dataset[()]

            result = {
                "path": path,
                "shape": dataset.shape,
                "dtype": str(dataset.dtype),
                "data": data.tolist() if isinstance(data, np.ndarray) else data,
                "_meta": {
                    "tool": "read_full_dataset",
                    "file": self.current_file
                }
            }

            return [TextContent(text=json.dumps(result))]

        except Exception as e:
            return [TextContent(
                text=json.dumps({
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "isError": True
                })
            )]
```

**After (FastMCP)**:
```python
@mcp.tool()
async def read_full_dataset(
    path: str,
    filename: Optional[str] = None,
    ctx: Context = None
) -> dict:
    """Read complete dataset from HDF5 file"""
    try:
        if ctx:
            await ctx.info(f"Reading dataset: {path}")

        file = await ensure_file_open(filename)

        if path not in file:
            return {
                "error": f"Dataset {path} not found",
                "isError": True
            }

        dataset = file[path]
        data = dataset[()]

        return {
            "path": path,
            "shape": list(dataset.shape),
            "dtype": str(dataset.dtype),
            "data": data.tolist() if isinstance(data, np.ndarray) else data,
            "_meta": {
                "tool": "read_full_dataset",
                "file": str(filename)
            }
        }
    except Exception as e:
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "isError": True
        }
```

**Changes**:
- Remove `List[TextContent]` return type → `dict`
- Remove manual JSON serialization (auto-handled)
- Add optional `Context` for logging
- Keep error handling structure (compatible)
- Eliminate class method → standalone function

### Migration Checklist (If Proceeding)

#### Phase 1: Preparation (1 hour)
- [ ] Create migration branch
- [ ] Install FastMCP: `uv add fastmcp`
- [ ] Review all 25 tools and their signatures
- [ ] Identify shared state/resources (file handles, cache)
- [ ] Document current behavior

#### Phase 2: Core Server (2 hours)
- [ ] Create new `server_fastmcp.py` in parallel
- [ ] Initialize FastMCP server
- [ ] Set up configuration (keep existing Config class)
- [ ] Handle server lifecycle (startup/shutdown)

#### Phase 3: Tools Migration (6-8 hours)
- [ ] Migrate file operations (open, close, get_filename, etc.)
- [ ] Migrate dataset operations (read_full, read_partial, etc.)
- [ ] Migrate attribute operations
- [ ] Migrate performance operations (parallel_scan, batch_read, etc.)
- [ ] Migrate discovery operations (analyze_structure, find_similar, etc.)
- [ ] Update error handling to return dicts
- [ ] Test each tool individually

#### Phase 4: Resources & Prompts (2 hours)
- [ ] Convert resources to @mcp.resource()
- [ ] Convert prompts to @mcp.prompt()
- [ ] Test resource URIs
- [ ] Test prompt generation

#### Phase 5: Advanced Features (2 hours)
- [ ] Implement file handle management (Context or global state)
- [ ] Implement LRU caching (decorator or manual)
- [ ] Implement parallel processing
- [ ] Implement streaming

#### Phase 6: Testing (3 hours)
- [ ] Unit tests for all tools
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] Client testing (Cursor, Claude Code)

#### Phase 7: Documentation (1 hour)
- [ ] Update README.md
- [ ] Update TOOLS.md
- [ ] Update examples
- [ ] Update client configurations

### Handling Advanced Features

#### File Handle Management

**Current**: Class-level state in `HDF5Tools`

**FastMCP Options**:

1. **Global State** (Simple):
```python
_file_handles: dict[str, h5py.File] = {}

@mcp.tool()
async def open_file(filename: str, mode: str = "r") -> dict:
    """Open HDF5 file"""
    _file_handles[filename] = h5py.File(filename, mode)
    return {"filename": filename, "mode": mode}

@mcp.tool()
async def read_dataset(path: str, filename: str) -> dict:
    """Read dataset from open file"""
    file = _file_handles.get(filename)
    if not file:
        return {"error": "File not open", "isError": True}
    # ...
```

2. **Context-based** (Advanced):
```python
from fastmcp import FastMCP, Context

class FileManager:
    def __init__(self):
        self.handles = {}

    def get_file(self, filename: str) -> h5py.File:
        return self.handles.get(filename)

file_manager = FileManager()

@mcp.tool()
async def read_dataset(path: str, filename: str, ctx: Context) -> dict:
    """Read dataset with context"""
    file = file_manager.get_file(filename)
    await ctx.info(f"Reading {path} from {filename}")
    # ...
```

#### LRU Caching

**Current**: Manual `LRUCache` implementation

**FastMCP Options**:

1. **functools.lru_cache** (Built-in):
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_dataset_metadata(filename: str, path: str) -> dict:
    """Cached metadata lookup"""
    with h5py.File(filename, 'r') as f:
        dataset = f[path]
        return {
            "shape": list(dataset.shape),
            "dtype": str(dataset.dtype)
        }

@mcp.tool()
async def read_dataset(path: str, filename: str) -> dict:
    """Uses cached metadata"""
    metadata = get_dataset_metadata(filename, path)
    # ...
```

2. **Custom Decorator**:
```python
from functools import wraps

def cached_tool(func):
    cache = {}

    @wraps(func)
    async def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = await func(*args, **kwargs)
        return cache[key]

    return wrapper

@mcp.tool()
@cached_tool
async def expensive_operation(data: str) -> dict:
    """Cached tool"""
    # Expensive processing
    return result
```

#### Parallel Processing

**Current**: ThreadPoolExecutor in `parallel_ops.py`

**FastMCP**: Same approach works

```python
from concurrent.futures import ThreadPoolExecutor
import asyncio

executor = ThreadPoolExecutor(max_workers=4)

@mcp.tool()
async def batch_read_datasets(paths: list[str], filename: str) -> list[dict]:
    """Read multiple datasets in parallel"""
    def read_one(path: str) -> dict:
        with h5py.File(filename, 'r') as f:
            dataset = f[path]
            return {
                "path": path,
                "data": dataset[()].tolist()
            }

    # Run in parallel
    loop = asyncio.get_event_loop()
    results = await asyncio.gather(
        *[loop.run_in_executor(executor, read_one, path) for path in paths]
    )

    return results
```

### Recommendation for HDF5

**DO NOT MIGRATE NOW** - HDF5 v2.0.2 is production-ready and stable

**Instead**:
1. Use FastMCP for **new** MCPs in the ecosystem
2. Gain experience with FastMCP on simpler MCPs first
3. Revisit HDF5 migration in Q1 2026 if desired
4. Focus migration effort on the 12 standard SDK MCPs

**HDF5-Specific Complexity**:
- 25+ tools (most complex MCP)
- Advanced features: caching, streaming, parallel processing
- Custom transport implementations
- Already highly optimized
- Low maintenance burden

**Risk vs Reward**: Low reward (already stable) vs Medium risk (extensive testing needed)

---

## Quick Reference

### FastMCP Essentials

```python
from fastmcp import FastMCP, Context

# Create server
mcp = FastMCP("ServerName")

# Tool
@mcp.tool()
async def my_tool(param: str, ctx: Context) -> dict:
    await ctx.info("Processing...")
    return {"result": "success"}

# Resource
@mcp.resource("data://{id}")
def get_data(id: str) -> dict:
    return {"id": id, "data": "..."}

# Prompt
@mcp.prompt()
def my_prompt(topic: str) -> str:
    return f"Explain {topic}"

# Run
if __name__ == "__main__":
    mcp.run()  # stdio
    # mcp.run(transport="http", port=8000)
```

### Key Differences from Standard SDK

| Aspect | Standard SDK | FastMCP |
|--------|-------------|---------|
| Server Init | `Server(name="...")` + handler registration | `FastMCP("name")` |
| Tool Registration | Manual `@server.list_tools()` + `@server.call_tool()` | `@mcp.tool()` |
| Parameter Schema | Manual JSON schema | Auto-generated from type hints |
| Return Values | `List[TextContent]` | Native Python types (auto-serialized) |
| Resources | Manual `@server.read_resource()` | `@mcp.resource("uri")` |
| Prompts | Manual `@server.get_prompt()` | `@mcp.prompt()` |
| Transports | Custom implementations | Built-in (stdio/http/sse) |
| Context Access | Manual parameter passing | `Context` dependency injection |
| Server Composition | Not supported | `import_server()` / `mount()` |

### Common Patterns

**Async I/O Tool**:
```python
@mcp.tool()
async def read_file(path: str, ctx: Context) -> str:
    await ctx.info(f"Reading {path}")
    async with aiofiles.open(path) as f:
        return await f.read()
```

**Pydantic Validation**:
```python
from pydantic import BaseModel, Field

class Request(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(10, ge=1, le=100)

@mcp.tool()
def search(request: Request) -> list:
    return perform_search(request.query, request.limit)
```

**Error Handling**:
```python
@mcp.tool()
async def safe_operation(data: str) -> dict:
    try:
        result = await process(data)
        return {"status": "success", "result": result}
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "isError": True
        }
```

**Progress Reporting**:
```python
@mcp.tool()
async def batch_process(items: list[str], ctx: Context) -> dict:
    for i, item in enumerate(items):
        await process(item)
        await ctx.report_progress(i + 1, len(items), f"Processed {i+1}/{len(items)}")
    return {"count": len(items)}
```

---

## Resources

### Official Documentation
- **FastMCP Docs**: https://gofastmcp.com/
- **GitHub**: https://github.com/jlowin/fastmcp
- **PyPI**: https://pypi.org/project/fastmcp/

### Tutorials
- **Quickstart**: https://gofastmcp.com/getting-started/quickstart
- **Tools Guide**: https://gofastmcp.com/servers/tools
- **Resources Guide**: https://gofastmcp.com/servers/resources
- **Prompts Guide**: https://gofastmcp.com/servers/prompts
- **Context Guide**: https://gofastmcp.com/servers/context
- **Server Composition**: https://gofastmcp.com/servers/composition
- **Running Server**: https://gofastmcp.com/deployment/running-server

### Version Info
- **Current Version**: 2.2.7 (October 2025)
- **Python Support**: >= 3.10
- **MCP Protocol**: 2025-06-18

---

**Document Version**: 1.0
**Last Updated**: October 18, 2025
**Prepared For**: HDF5 MCP Migration Planning
