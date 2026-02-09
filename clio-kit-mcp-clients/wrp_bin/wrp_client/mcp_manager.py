import json
import asyncio
import sys
import os
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from pathlib import Path
from typing import List, Optional

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from .providers.base import BaseLLM, ToolDef

def find_server_py(server_name: str) -> str:
    """
    Locates the main server.py script for a given MCP server name.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    for search_path in [repo_root / "mcps" / server_name / "src", repo_root / server_name]:
        if search_path.exists():
            matches = list(search_path.rglob("server.py"))
            if matches:
                return str(matches[0])
    raise FileNotFoundError(f"Could not find server.py for {server_name}")


class MCPManager(ABC):
    """
    Abstract base class for managing MCP interactions.
    """
    def __init__(self, llm_adapter: BaseLLM, verbose: bool = False):
        self.llm = llm_adapter
        self.verbose = verbose

    @abstractmethod
    async def connect(self, server_script: str):
        """Connect to MCP server(s)."""
        pass

    @abstractmethod
    async def process_query(self, query: str) -> str:
        """Process a user query and return a response."""
        pass

    async def chat_loop(self):
        """Interactive chat loop"""
        print(f"\nMCP Client Started! (type 'quit' to exit)")
        try:
            while True:
                q = input("\nQuery: ").strip()
                if q.lower() in ('quit', 'exit'):
                    break
                response_text = await self.process_query(q)
                print(f"\n{response_text}")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
        finally:
            await self.cleanup()

    @abstractmethod
    async def cleanup(self):
        """Clean up resources."""
        pass


class WarpMCPManager(MCPManager):
    """
    Manages the connection and interaction with a single MCP server.
    This is the original MCPManager functionality.
    """
    def __init__(self, llm_adapter: BaseLLM, verbose: bool = False):
        super().__init__(llm_adapter, verbose)
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.tools: List[ToolDef] = []
        self.conversation_history: List[dict] = []

    async def connect(self, server_script: str):
        print(f"Starting server with stdio: {server_script}" if self.verbose else "")
        params = StdioServerParameters(command=sys.executable, args=[server_script], env=os.environ)
        reader, writer = await self.exit_stack.enter_async_context(stdio_client(params))
        self.session = await self.exit_stack.enter_async_context(ClientSession(reader, writer))
        
        await self.session.initialize()
        tool_list = await self.session.list_tools()
        
        self.tools = [
            ToolDef(
                name=t.name,
                description=t.description,
                input_schema=t.inputSchema,
            )
            for t in tool_list.tools
        ]
        print("\nConnected. Tools available:\n")
        for tool in self.tools:
            print(f" * {tool.name}: {tool.description}")

    async def process_query(self, query: str) -> str:
        # Add user query to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        llm_reply = None

        try:
            llm_reply = await self.llm.chat(self.conversation_history, self.tools)
        
        except Exception as e:
            return f"Error during initial LLM processing: {e}"

        if llm_reply.tool_calls:
            tool_results = []
            verbose_parts = []
            
            for tool_call in llm_reply.tool_calls:
                name, args = tool_call["name"], tool_call["args"]
                if self.verbose:
                    verbose_parts.append(f"[Calling tool {name} with args {args}]")
                
                try:
                    tr = await self.session.call_tool(name, args)
                    raw_txt = tr.content[0].text if tr.content else "No content returned"

                    is_error = False
                    try:
                        parsed = json.loads(raw_txt)
                        if isinstance(parsed, dict) and (parsed.get("isError") or "error" in parsed):
                            is_error = True
                    except Exception:
                        pass

                    if self.verbose:
                        verbose_parts.append(f"[Called {name}: {raw_txt}]")
                    
                    if is_error:
                        tool_results.append("Error: Incorrect filepath or argument passed.")
                    else:
                        tool_results.append(raw_txt)

                except Exception as e:
                    error_msg = f"Error calling {name}: {e}"
                    tool_results.append(error_msg)
                    if self.verbose:
                        verbose_parts.append(f"[{error_msg}]")

            # Combine all tool results for LLM processing
            all_tool_results = "\n".join(tool_results)
            
            # Add tool call and results to conversation history
            self.conversation_history.append({
                "role": "assistant", 
                "content": llm_reply.text,
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": all_tool_results
            })
            
            # Create a new message with the original query and tool results
            followup_messages = [
                {"role": "user", "content": f"Original query: {query}\n\nTool results: {all_tool_results}\n\nPlease provide a clear, natural language response to the original query based on these tool results."}
            ]

            final_reply = None

            try:
                # Get LLM to process the results into natural language (no tools needed)
                final_reply = await self.llm.chat(followup_messages, [])
            except Exception as e:
                return f"Error during post-tool LLM processing: {e}"

            # Add final response to conversation history
            self.conversation_history.append({"role": "assistant", "content": final_reply.text})
            
            if self.verbose:
                # Show verbose info first, then the natural language output
                verbose_output = "\n".join(verbose_parts)
                return f"{verbose_output}\nOutput: {final_reply.text}"
            else:
                return f"Output: {final_reply.text}"
        else:
            # No tool calls - add assistant response to history
            self.conversation_history.append({"role": "assistant", "content": llm_reply.text})
            return llm_reply.text

    async def cleanup(self):
        await self.exit_stack.aclose()


class EmptyMCPManager(MCPManager):
    """
    Simplified MCP manager that assumes the LLM handles tool calls internally.
    Used when MCPs are managed by an external agent framework.
    """
    def __init__(self, llm_adapter: BaseLLM, verbose: bool = False):
        super().__init__(llm_adapter, verbose)

    async def connect(self, server_script: str):
        """No-op connection since MCPs are externally managed."""
        if self.verbose:
            print(f"EmptyMCPManager: Skipping connection to {server_script} (externally managed)")

    async def process_query(self, query: str) -> str:
        """Simple query processing that delegates to LLM's internal tool handling."""
        messages = [{"role": "user", "content": query}]
        
        try:
            # Assume the LLM handles tool calls internally
            llm_reply = await self.llm.chat(messages, [])
            return llm_reply.text
        except Exception as e:
            return f"Error during LLM processing: {e}"

    async def cleanup(self):
        """No cleanup needed for empty manager."""
        pass 