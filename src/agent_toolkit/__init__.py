#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path
import click

# Determine if we're running from development or installed package
MODULE_DIR = Path(__file__).parent

def get_servers_path():
    """Get the path to servers directory (dev or installed)"""
    # First try development path (../../agent-toolkit-mcp-servers from module)
    dev_path = MODULE_DIR.parent.parent / "agent-toolkit-mcp-servers"
    if dev_path.exists():
        return dev_path

    # Try to find shared data in the installed package
    # When installed via wheel, shared data goes to site-packages/
    # Look for agent-toolkit-mcp-servers directory in various possible locations
    possible_paths = [
        # Standard site-packages installation
        MODULE_DIR.parent / "agent-toolkit-mcp-servers",  # ../agent-toolkit-mcp-servers from module
        # Alternative installation paths
        MODULE_DIR / "agent-toolkit-mcp-servers",  # ./agent-toolkit-mcp-servers from module (if included directly)
        # System-wide data directory
        Path(sys.prefix) / "share" / "agent-toolkit" / "agent-toolkit-mcp-servers",
        # Local data directory
        Path.home() / ".local" / "share" / "agent-toolkit" / "agent-toolkit-mcp-servers",
    ]

    # Try each possible path
    for path in possible_paths:
        if path.exists() and path.is_dir():
            return path

    # If none found, check if we're in an isolated environment (like uvx)
    # and try to find the data directory relative to the Python executable
    python_path = Path(sys.executable)
    isolated_paths = [
        # uvx style isolated environment - agent-toolkit-mcp-servers is at the root level
        python_path.parent.parent / "agent-toolkit-mcp-servers",
        python_path.parent.parent / "share" / "agent-toolkit-mcp-servers",
        python_path.parent.parent / "purelib" / "agent-toolkit-mcp-servers",
        python_path.parent.parent / "data" / "agent-toolkit-mcp-servers",
    ]

    for path in isolated_paths:
        if path.exists() and path.is_dir():
            return path

    # Last resort: return the dev path even if it doesn't exist
    # so the caller can handle the missing directory appropriately
    return dev_path

def get_prompts_path():
    """Get the path to prompts directory (dev or installed)"""
    # First try development path (../../prompts from module)
    dev_path = MODULE_DIR.parent.parent / "prompts"
    if dev_path.exists():
        return dev_path

    # Try to find shared data in the installed package
    possible_paths = [
        # Standard site-packages installation
        MODULE_DIR.parent / "prompts",  # ../prompts from module
        # Alternative installation paths
        MODULE_DIR / "prompts",  # ./prompts from module
        # System-wide data directory
        Path(sys.prefix) / "share" / "agent-toolkit" / "prompts",
        # Local data directory
        Path.home() / ".local" / "share" / "agent-toolkit" / "prompts",
    ]

    # Try each possible path
    for path in possible_paths:
        if path.exists() and path.is_dir():
            return path

    # If none found, check if we're in an isolated environment (like uvx)
    python_path = Path(sys.executable)
    isolated_paths = [
        # uvx style isolated environment
        python_path.parent.parent / "prompts",
        python_path.parent.parent / "share" / "prompts",
        python_path.parent.parent / "purelib" / "prompts",
        python_path.parent.parent / "data" / "prompts",
    ]

    for path in isolated_paths:
        if path.exists() and path.is_dir():
            return path

    # Last resort: return the dev path
    return dev_path

def auto_discover_mcps():
    """Auto-discover MCP servers from the agent-toolkit-mcp-servers directory"""
    servers_path = get_servers_path()
    if not servers_path.exists():
        return {}, {}

    server_command_map = {}
    dir_name_map = {}

    # Scan for directories containing pyproject.toml
    for item in servers_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            pyproject_file = item / "pyproject.toml"
            if pyproject_file.exists():
                # Read pyproject.toml to extract entry point
                try:
                    with open(pyproject_file, 'r') as f:
                        content = f.read()

                    # Simple parsing to find the entry point
                    # Look for lines like: server-name-mcp = "module:main"
                    entry_point = None
                    for line in content.split('\n'):
                        line = line.strip()
                        if '-mcp =' in line and '=' in line:
                            entry_point = line.split('=')[0].strip().strip('"\'')
                            break

                    if entry_point:
                        # Create server name by removing -mcp suffix
                        server_name = entry_point.replace('-mcp', '').lower()
                        # Handle special cases for naming
                        if server_name == 'node-hardware':
                            server_name = 'node-hardware'
                        elif server_name == 'parallel-sort':
                            server_name = 'parallel-sort'

                        server_command_map[server_name] = entry_point
                        dir_name_map[server_name] = item.name

                except Exception as e:
                    # Skip directories that can't be processed
                    continue

    return server_command_map, dir_name_map

def auto_discover_prompts():
    """Auto-discover prompts from the prompts directory (recursively)"""
    prompts_path = get_prompts_path()
    if not prompts_path.exists():
        return {}

    prompt_map = {}

    # Recursively scan for .md files
    for md_file in prompts_path.rglob("*.md"):
        # Get relative path from prompts directory
        relative_path = md_file.relative_to(prompts_path)

        # Create prompt name from relative path without extension
        # e.g., "code-coverage-prompt.md" -> "code-coverage-prompt"
        # e.g., "testing/foo.md" -> "testing/foo"
        prompt_name = str(relative_path.with_suffix(''))

        # Also support underscore version
        # "code-coverage-prompt" -> also accessible as "code_coverage_prompt"
        prompt_map[prompt_name] = md_file
        prompt_map[prompt_name.replace('-', '_')] = md_file

    return prompt_map

def list_available_servers():
    """List all available servers"""
    server_command_map, _ = auto_discover_mcps()
    return sorted(server_command_map.keys())

def list_available_prompts():
    """List all available prompts"""
    prompt_map = auto_discover_prompts()
    # Remove duplicates (dash vs underscore versions)
    unique_prompts = set()
    for name in prompt_map.keys():
        # Normalize to dash version for display
        unique_prompts.add(name.replace('_', '-'))
    return sorted(unique_prompts)

@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """iowarp-agent-toolkit: Unified launcher for MCP servers and AI prompts"""
    if ctx.invoked_subcommand is None:
        click.echo("iowarp-agent-toolkit: Unified launcher for MCP servers and AI prompts")
        click.echo("\nAvailable commands:")
        click.echo("  mcp-server   Run an MCP server")
        click.echo("  mcp-servers  List all available MCP servers")
        click.echo("  prompt       Print a prompt to stdout")
        click.echo("  prompts      List all available prompts")
        click.echo("\nUsage:")
        click.echo("  uvx iowarp-agent-toolkit mcp-server <server-name>")
        click.echo("  uvx iowarp-agent-toolkit prompt <prompt-name>")
        click.echo("\nFor more help: uvx iowarp-agent-toolkit <command> --help")

@main.command("mcp-server")
@click.argument('server', required=False)
@click.option('-b', '--branch', help='Git branch to use (for development)')
@click.argument('args', nargs=-1, type=click.UNPROCESSED)
def mcp_server(server, branch, args):
    """Run an MCP server. List all if no server specified."""

    server_command_map, dir_name_map = auto_discover_mcps()

    if not server:
        click.echo("Available MCP servers:")
        for s in sorted(server_command_map.keys()):
            click.echo(f"  - {s}")
        click.echo("\nUsage: uvx iowarp-agent-toolkit mcp-server <server-name>")
        click.echo("   or: agent-toolkit mcp-server <server-name> (if installed)")
        return

    # Normalize server name to lowercase
    server_lower = server.lower()

    if server_lower not in server_command_map:
        click.echo(f"Error: Unknown server '{server}'")
        click.echo(f"Available servers: {', '.join(sorted(server_command_map.keys()))}")
        sys.exit(1)

    # Get the entry point command and directory name
    entry_command = server_command_map[server_lower]
    actual_dir = dir_name_map[server_lower]

    # Build uvx command
    if branch:
        # Run from git branch
        cmd = [
            "uvx",
            "--from",
            f"git+https://github.com/iowarp/agent-toolkit.git@{branch}#subdirectory=agent-toolkit-mcp-servers/{actual_dir}",
            entry_command
        ]
    else:
        # Run from local path in development mode
        servers_path = get_servers_path()
        server_path = servers_path / actual_dir

        if server_path.exists():
            # Development mode - run from local path
            cmd = [
                "uvx",
                "--from",
                str(server_path),
                entry_command
            ]
        else:
            # Not in development, try to run the command directly (if installed)
            cmd = [entry_command]

    # Add any additional arguments
    cmd.extend(args)

    # Execute the command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except FileNotFoundError:
        if cmd[0] == "uvx":
            click.echo("Error: uvx not found. Please install uv: https://github.com/astral-sh/uv")
        else:
            click.echo(f"Error: {entry_command} not found. Please install the server package.")
        sys.exit(1)

@main.command("mcp-servers")
def list_mcp_servers():
    """List all available MCP servers"""
    servers = list_available_servers()
    if servers:
        click.echo("Available MCP servers:")
        for s in servers:
            click.echo(f"  - {s}")
    else:
        click.echo("No MCP servers found.")

@main.command("prompt")
@click.argument('prompt_name', required=False)
def prompt(prompt_name):
    """Print a prompt to stdout. List all if no name specified."""

    prompt_map = auto_discover_prompts()

    if not prompt_name:
        # List all prompts
        prompts = list_available_prompts()
        if prompts:
            click.echo("Available prompts:")
            for p in prompts:
                click.echo(f"  - {p}")
        else:
            click.echo("No prompts found.")
        click.echo("\nUsage: uvx iowarp-agent-toolkit prompt <prompt-name>")
        click.echo("   or: agent-toolkit prompt <prompt-name> (if installed)")
        return

    # Normalize prompt name (support both dash and underscore)
    prompt_lower = prompt_name.lower()

    if prompt_lower not in prompt_map:
        click.echo(f"Error: Unknown prompt '{prompt_name}'")
        click.echo(f"Available prompts: {', '.join(list_available_prompts())}")
        sys.exit(1)

    # Read and print the prompt file
    prompt_file = prompt_map[prompt_lower]
    try:
        with open(prompt_file, 'r') as f:
            content = f.read()
        click.echo(content)
    except Exception as e:
        click.echo(f"Error reading prompt file: {e}")
        sys.exit(1)

@main.command("prompts")
def list_prompts_cmd():
    """List all available prompts"""
    prompts = list_available_prompts()
    if prompts:
        click.echo("Available prompts:")
        for p in prompts:
            click.echo(f"  - {p}")
    else:
        click.echo("No prompts found.")

def cli():
    """Entry point for the CLI"""
    main()

if __name__ == "__main__":
    main()
