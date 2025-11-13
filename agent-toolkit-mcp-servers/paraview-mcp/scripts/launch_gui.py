#!/usr/bin/env python3
"""
ParaView GUI launcher script - detects multiple installation types
"""
import sys
import subprocess
import os
import shutil
from pathlib import Path


def find_paraview_binary():
    """Find ParaView binary from multiple installation types"""
    project_root = Path(__file__).parent.parent.resolve()
    
    # Priority 1: Locally built ParaView (best ADIOS2 support)
    local_paraview = project_root / ".paraview" / "bin" / "paraview"
    if local_paraview.exists():
        return str(local_paraview), "locally built (full ADIOS2 support)"
    
    # Priority 2: Conda environment ParaView
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        conda_paraview = Path(conda_prefix) / "bin" / "paraview"
        if conda_paraview.exists():
            return str(conda_paraview), "conda environment"
    
    # Priority 3: System ParaView in PATH
    system_paraview = shutil.which("paraview")
    if system_paraview:
        return system_paraview, "system installation"
    
    return None, None


def main():
    """Launch ParaView GUI"""
    project_root = Path(__file__).parent.parent.resolve()
    
    print("🎨 Starting ParaView GUI...")
    print(f"Project root: {project_root}")
    
    # Find ParaView installation
    paraview_bin, install_type = find_paraview_binary()
    
    if not paraview_bin:
        print("❌ ParaView not found in any location:")
        print("   • No local build (.paraview directory)")
        print("   • No conda environment with ParaView")
        print("   • No system ParaView in PATH")
        print()
        print("💡 Installation options:")
        print("   uv run automate-setup          # Build with full ADIOS2 support")
        print("   conda install -c conda-forge paraview adios2  # Quick conda install")
        print("   sudo apt install paraview      # System package (Ubuntu)")
        return 1
    
    print(f"🚀 Found ParaView: {install_type}")
    print(f"   Binary: {paraview_bin}")
    
    try:
        # Change to project directory for proper data access
        os.chdir(project_root)
        
        # Launch ParaView GUI with any additional arguments
        subprocess.run([paraview_bin] + sys.argv[1:])
        return 0
        
    except KeyboardInterrupt:
        print("\n👋 ParaView GUI closed by user")
        return 0
    except Exception as e:
        print(f"❌ Failed to launch ParaView GUI: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())