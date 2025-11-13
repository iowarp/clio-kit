#!/usr/bin/env python3
"""
Automated ParaView build script with ADIOS2 support
This script automates the entire ParaView compilation process
"""

import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path
import platform
import multiprocessing


def run_command(cmd, cwd=None, check=True):
    """Run a shell command with proper error handling"""
    print(f"Running: {cmd}")
    if isinstance(cmd, str):
        cmd = cmd.split()
    
    result = subprocess.run(cmd, cwd=cwd, capture_output=False, text=True)
    if check and result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        sys.exit(1)
    return result


def check_system_dependencies():
    """Check if required system dependencies are installed"""
    print("Checking system dependencies...")
    
    # Check for required commands
    required_commands = ['git', 'cmake', 'ninja']
    missing = []
    
    for cmd in required_commands:
        if shutil.which(cmd) is None:
            missing.append(cmd)
    
    if missing:
        print(f"Missing required commands: {', '.join(missing)}")
        print("Please install system dependencies first:")
        
        if platform.system() == "Linux":
            print("sudo apt update && sudo apt install git cmake build-essential ninja-build \\")
            print("    libgl1-mesa-dev libxt-dev python3-dev python3-numpy \\")
            print("    libopenmpi-dev libtbb-dev qtbase5-dev \\")
            print("    libadios2-mpi-core-dev libadios2-mpi-c++11-dev \\")
            print("    libadios2-serial-core-dev libadios2-serial-c++11-dev")
        
        return False
    
    return True


def get_python_executable():
    """Get the current Python executable path"""
    return sys.executable


def build_paraview(project_root, force_rebuild=False, parallel_jobs=None):
    """Build ParaView with ADIOS2 support"""
    
    if not parallel_jobs:
        parallel_jobs = min(multiprocessing.cpu_count(), 8)  # Limit to 8 to avoid memory issues
    
    project_root = Path(project_root).absolute()
    parent_dir = project_root.parent
    paraview_src = parent_dir / "paraview"
    paraview_build = parent_dir / "paraview_build"
    paraview_install = project_root / ".paraview"
    
    print(f"Project root: {project_root}")
    print(f"ParaView source: {paraview_src}")
    print(f"ParaView build: {paraview_build}")
    print(f"ParaView install: {paraview_install}")
    
    # Check if ParaView is already built
    if paraview_install.exists() and not force_rebuild:
        print("ParaView installation already exists. Use --force to rebuild.")
        return True
    
    # Clone ParaView if not exists
    if not paraview_src.exists():
        print("Cloning ParaView repository...")
        run_command(f"git clone --recursive https://gitlab.kitware.com/paraview/paraview.git {paraview_src}")
        run_command("git checkout v5.13.1", cwd=paraview_src)
        run_command("git submodule update --init --recursive", cwd=paraview_src)
    else:
        print("ParaView source already exists, skipping clone.")
    
    # Clean build directory if force rebuild
    if force_rebuild and paraview_build.exists():
        print("Removing existing build directory...")
        shutil.rmtree(paraview_build)
    
    # Create build directory
    paraview_build.mkdir(exist_ok=True)
    
    # Configure CMake
    python_executable = get_python_executable()
    cmake_args = [
        "cmake", "-GNinja",
        "-DPARAVIEW_USE_PYTHON=ON",
        "-DPARAVIEW_USE_MPI=ON",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DPython3_EXECUTABLE={python_executable}",
        f"-DCMAKE_INSTALL_PREFIX={paraview_install}",
        "-DPARAVIEW_ENABLE_ADIOS2=ON",
        "-DVTK_MODULE_ENABLE_VTK_IOADIOS2:STRING=DEFAULT",
        str(paraview_src)
    ]
    
    print("Configuring ParaView build...")
    run_command(cmake_args, cwd=paraview_build)
    
    # Build ParaView
    print(f"Building ParaView with {parallel_jobs} parallel jobs...")
    print("This may take 1-3 hours depending on your system...")
    run_command(f"ninja -j{parallel_jobs}", cwd=paraview_build)
    
    # Install ParaView
    print("Installing ParaView...")
    run_command("ninja install", cwd=paraview_build)
    
    return True


def setup_python_path(project_root):
    """Setup Python path for ParaView"""
    project_root = Path(project_root).absolute()
    venv_site_packages = project_root / ".venv" / "lib" / "python3.10" / "site-packages"
    paraview_site_packages = project_root / ".paraview" / "lib" / "python3.10" / "site-packages"
    
    # Create the site-packages directory if it doesn't exist
    venv_site_packages.mkdir(parents=True, exist_ok=True)
    
    # Create paraview.pth file
    pth_file = venv_site_packages / "paraview.pth"
    with open(pth_file, "w") as f:
        f.write(str(paraview_site_packages) + "\n")
    
    print(f"Created Python path file: {pth_file}")


def verify_installation(project_root):
    """Verify that ParaView and ADIOS2 are properly installed"""
    print("Verifying installation...")
    
    try:
        # Test ParaView import
        result = subprocess.run([
            sys.executable, "-c", 
            "import paraview.simple; print('✓ ParaView available')"
        ], capture_output=True, text=True, cwd=project_root)
        
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print("❌ ParaView import failed")
            print(result.stderr)
            return False
        
        # Test ADIOS2 support
        result = subprocess.run([
            sys.executable, "-c",
            "import vtk; reader = vtk.vtkADIOS2CoreImageReader(); print('✓ ADIOS2 support enabled')"
        ], capture_output=True, text=True, cwd=project_root)
        
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print("❌ ADIOS2 support verification failed")
            print(result.stderr)
            return False
        
        print("✅ Installation verification successful!")
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Build ParaView with ADIOS2 support")
    parser.add_argument("--force", action="store_true", 
                       help="Force rebuild even if ParaView is already installed")
    parser.add_argument("--jobs", "-j", type=int, 
                       help="Number of parallel build jobs")
    parser.add_argument("--skip-deps-check", action="store_true",
                       help="Skip system dependencies check")
    parser.add_argument("--project-root", 
                       default=os.getcwd(),
                       help="Project root directory")
    
    args = parser.parse_args()
    
    print("🚀 ParaView Build Automation Script")
    print("=" * 50)
    
    # Check system dependencies
    if not args.skip_deps_check:
        if not check_system_dependencies():
            sys.exit(1)
    
    project_root = Path(args.project_root).absolute()
    
    try:
        # Build ParaView
        success = build_paraview(project_root, args.force, args.jobs)
        if not success:
            print("❌ ParaView build failed")
            sys.exit(1)
        
        # Setup Python path
        setup_python_path(project_root)
        
        # Verify installation
        if verify_installation(project_root):
            print("\n🎉 ParaView installation completed successfully!")
            print("\nNext steps:")
            print("1. Start ParaView GUI: ./.paraview/bin/paraview")
            print("2. Start ParaView server: ./.paraview/bin/pvserver --multi-clients")
            print("3. Start MCP server: uv run paraview-mcp")
        else:
            print("❌ Installation verification failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n❌ Build interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Build failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
