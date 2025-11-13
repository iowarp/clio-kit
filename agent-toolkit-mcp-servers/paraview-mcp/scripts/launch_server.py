#!/usr/bin/env python3
"""
ParaView server launcher script - detects multiple installation types
"""
import sys
import subprocess
import os
import argparse
import shutil
from pathlib import Path


def find_pvserver_binary():
    """Find ParaView server binary from multiple installation types"""
    project_root = Path(__file__).parent.parent.resolve()
    
    # Priority 1: Locally built ParaView (best ADIOS2 support)
    local_pvserver = project_root / ".paraview" / "bin" / "pvserver"
    if local_pvserver.exists():
        return str(local_pvserver), "locally built (full ADIOS2 support)"
    
    # Priority 2: Conda environment ParaView
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        conda_pvserver = Path(conda_prefix) / "bin" / "pvserver"
        if conda_pvserver.exists():
            return str(conda_pvserver), "conda environment"
    
    # Priority 3: System ParaView in PATH
    system_pvserver = shutil.which("pvserver")
    if system_pvserver:
        return system_pvserver, "system installation"
    
    return None, None


def main():
    """Launch ParaView server"""
    parser = argparse.ArgumentParser(description="Launch ParaView server")
    parser.add_argument("--port", type=int, default=11111,
                       help="Server port (default: 11111)")
    parser.add_argument("--multi-clients", action="store_true", default=True,
                       help="Allow multiple client connections")
    parser.add_argument("--server-url", default="localhost", 
                       help="Server URL (default: localhost)")
    
    args, unknown_args = parser.parse_known_args()
    
    project_root = Path(__file__).parent.parent.resolve()
    
    print("🖥️  Starting ParaView Server...")
    print(f"Project root: {project_root}")
    
    # Find ParaView server installation
    pvserver_bin, install_type = find_pvserver_binary()
    
    if not pvserver_bin:
        print("❌ ParaView server not found in any location:")
        print("   • No local build (.paraview directory)")
        print("   • No conda environment with ParaView")
        print("   • No system pvserver in PATH")
        print()
        print("💡 Installation options:")
        print("   uv run automate-setup          # Build with full ADIOS2 support")
        print("   conda install -c conda-forge paraview adios2  # Quick conda install")
        print("   sudo apt install paraview      # System package (Ubuntu)")
        return 1
    
    print(f"🚀 Found ParaView server: {install_type}")
    print(f"   Binary: {pvserver_bin}")
    
    # Build server command
    server_cmd = [pvserver_bin]
    
    if args.multi_clients:
        server_cmd.append("--multi-clients")
    
    if args.port != 11111:
        server_cmd.extend(["--server-port", str(args.port)])
    
    # Add any additional unknown arguments
    server_cmd.extend(unknown_args)
    
    print(f"🚀 Server command: {' '.join(server_cmd)}")
    print(f"📡 Server will run on: {args.server_url}:{args.port}")
    print(f"👥 Multi-client mode: {'Enabled' if args.multi_clients else 'Disabled'}")
    print()
    print("📋 Connection Instructions:")
    print("   1. Start ParaView GUI: uv run paraview-gui")
    print("   2. In GUI: File → Connect")
    print(f"   3. Add server: {args.server_url}:{args.port}")
    print("   4. Click Connect")
    print()
    print("Press Ctrl+C to stop server")
    print("-" * 50)
    
    try:
        # Change to project directory
        os.chdir(project_root)
        
        # Launch ParaView server
        subprocess.run(server_cmd)
        return 0
        
    except KeyboardInterrupt:
        print("\n👋 ParaView server stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Failed to launch ParaView server: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())