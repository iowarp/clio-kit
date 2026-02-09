import os
import sys
import subprocess
import asyncio

from ..helpers import extract_mcp_tools
from .base import BaseLLM, LLMReply, ToolDef
from typing import List, Dict
from pathlib import Path

class OpenCodeLLM(BaseLLM):
    """Adapter for OpenCode agent wrapper"""
    def __init__(self, model_name: str = "claude-3-haiku-20240307", api_key: str = None, **kwargs):
        if "Provider_Config_Path" not in kwargs:
            raise ValueError("OpenCodeLLM requires Provider_Config_Path in YAML to point to opencode config JSON")

        self.model_name = model_name
        self.externally_managed_mcps = True
        self.subproc_env = os.environ.copy()
        self.subproc_env["OPENCODE_CONFIG"] = kwargs["Provider_Config_Path"]

        result = subprocess.run(
            ["opencode", "run",
             "List all tools available to you, in a code block with the format\n# tool header 1\ntool1 - desc1\n...\n# tool header 2\ntoolN - desc2\n...",
             "--model=" + self.model_name],
            env=self.subproc_env,
            capture_output=True,
            text=True,
            timeout=60
        )

        all_tools = result.stdout

        self.mcp_tools = extract_mcp_tools(all_tools, kwargs["MCP"])
        print(f"\nConnected to opencode agent framework with model {self.model_name}. Available tools:\n{self.mcp_tools}")

    async def chat(
        self, messages: List[Dict[str, str]], tools: List[ToolDef]
    ) -> LLMReply:
        user_messages = [m['content'] for m in messages if m["role"] != "system"]        

        # Continuous opencode session handles remembering previous msgs in interaction
        # Just send the new msg
        result = subprocess.run(
            ["opencode", "run", user_messages[-1], "-c"],
            env=self.subproc_env,
            capture_output=True,
            text=True,
            timeout=60
        )

        agent_stdout = result.stdout
        agent_stderr = result.stderr

        return LLMReply(text=agent_stdout, tool_calls=None)
