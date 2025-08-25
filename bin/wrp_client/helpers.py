
def extract_mcp_tools(text, mcps):
    """
    Extract only MCP-specific tools from opencode tool listing text.

    Removes built-in opencode tools and keeps only tools under novel MCP headers

    Args:
        text (str): The full tool listing text from opencode

    Returns:
        str: Filtered text containing only MCP-specific tools
    """
    lines = text.split('\n')

    # Built-in tool names to remove
    builtin_tools = {
        "read", "write", "edit", "list", "glob", "grep",
        "bash", "task",
        "todowrite", "todoread",
        "webfetch"
    }

    filtered_lines = []
    skip_section = False

    for line in lines:
        # Check if this is a section header
        if line.startswith('# '):
            # TBD - More robust detection would be preferable

            # Determine whether this section pertains MCP tools
            skip_section = True

            for mcp in mcps:
                skip_section = (mcp.lower() not in line.strip().lower() and
                                "mcp" not in line.strip().lower())

            if skip_section:
                continue

        elif skip_section:
            continue
        elif ' - ' in line.strip():
            # Extract tool name by going from start of line to ' - '
            tool_name = line.strip().split(' - ')[0].strip()
            # If tool name is in builtin tools, continue (skip this line)
            if tool_name in builtin_tools:
                continue
        else:
            # Skip other lines by default (e.g. other semantic LLM output)
            continue

        filtered_lines.append(line)

    return '\n'.join(filtered_lines).strip()
