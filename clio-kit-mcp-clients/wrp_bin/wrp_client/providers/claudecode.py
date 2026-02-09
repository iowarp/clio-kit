import os
import sys
import subprocess
import asyncio

from ..helpers import extract_mcp_tools
from .base import BaseLLM, LLMReply, ToolDef
from typing import List, Dict
from pathlib import Path


class ClaudeCodeLLM(BaseLLM):
    """Adapter for ClaudeCode agent wrapper"""
    def __init__(self, model_name: str = "claude-3-haiku-20240307", api_key: str = None, **kwargs):
        if "MCP" not in kwargs:
            raise ValueError("ClaudeCode backend requires MCP specifications")

        # TBD: Pass model name to local config during initialization
        self.model_name = model_name
        self.externally_managed_mcps = True
        self.subproc_env = os.environ.copy()

        # Set up listed MCPs within Claude Code
        for mcp in kwargs["MCP"]:
            try:
                dir = mcp
                executable = mcp.lower() + "-mcp"

                # TODO - remove old MCPs before this
                result = subprocess.run(["claude", "mcp", "add", mcp, "--",
                                        "uv", f"--directory={os.path.dirname(os.path.abspath(__file__))}/../../../mcps/" + dir, "run", executable])
            except subprocess.CalledProcessError as e:
                print(f"Failed to add MCP {mcp} with exit code {e.returncode}.\
                      \nError output: {e.stderr}")

        print("Retrieving Claude Code tools...")
        result = subprocess.run(
            ["claude", "-p", "List all tools available to you, in an ESCAPED CODE BLOCK with the format\n# tool header 1\ntool1 - desc1\n...\n# tool header 2\ntoolN - desc2\n..."],
            env=self.subproc_env,
            capture_output=True,
            text=True,
            timeout=60
        )

        all_tools = result.stdout

        print(all_tools)
        self.mcp_tools = extract_mcp_tools(all_tools, kwargs["MCP"])
        print(f"\nConnected to Claude Code agent with model {self.model_name}. Available tools:\n{self.mcp_tools}")


    async def chat(
        self, messages: List[Dict[str, str]], tools: List[ToolDef]
    ) -> LLMReply:
        user_messages = [m['content'] for m in messages if m["role"] != "system"]        

        # Continuous opencode session handles remembering previous msgs in interaction
        # Just send the new msg
        result = subprocess.run(
            ["claude", "-p", user_messages[-1], "-c"],
            env=self.subproc_env,
            capture_output=True,
            text=True,
            timeout=60
        )

        agent_stdout = result.stdout
        agent_stderr = result.stderr

        return LLMReply(text=agent_stdout, tool_calls=None)
