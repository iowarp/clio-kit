#!/usr/bin/env python3
"""
Complete automated setup for ParaView MCP after virtual environment creation
This script handles everything from dependency installation to final verification
"""

import os
import sys
import subprocess
import shutil
import argparse
import time
import multiprocessing
from pathlib import Path

# Import existing helper modules  
from .install_deps import check_dependencies, install_dependencies_interactive


class SetupAutomator:
    def __init__(self, project_root, verbose=False, clean=False, jobs=None):
        self.project_root = Path(project_root).resolve()
        self.verbose = verbose
        self.clean = clean
        self.jobs = jobs or multiprocessing.cpu_count()
        
        # Define paths
        self.parent_dir = self.project_root.parent
        self.paraview_src = self.parent_dir / "paraview"
        self.paraview_build = self.parent_dir / "paraview_build"
        self.paraview_install = self.project_root / ".paraview"
        self.venv_python = self.project_root / ".venv" / "bin" / "python"
        
    def log(self, message, level="INFO"):
        """Log messages with timestamp"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        prefix = "🔧" if level == "INFO" else "❌" if level == "ERROR" else "✅"
        print(f"[{timestamp}] {prefix} {message}")
        if self.verbose and level == "DEBUG":
            print(f"    {message}")
            
    def run_command(self, cmd, cwd=None, check=True, shell=False, show_progress=False):
        """Run a command with proper error handling"""
        if isinstance(cmd, str) and not shell:
            cmd = cmd.split()
            
        self.log(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        
        try:
            if show_progress or self.verbose:
                # Show live output for long-running commands
                process = subprocess.Popen(
                    cmd,
                    cwd=cwd or self.project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    shell=shell
                )
                
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        print(f"   {output.strip()}")
                
                return process.returncode == 0
            else:
                result = subprocess.run(
                    cmd, 
                    cwd=cwd or self.project_root,
                    check=check,
                    capture_output=not self.verbose,
                    text=True,
                    shell=shell
                )
                return result.returncode == 0
                
        except subprocess.CalledProcessError as e:
            self.log(f"Command failed with return code {e.returncode}", "ERROR")
            if hasattr(e, 'stderr') and e.stderr:
                self.log(f"Error: {e.stderr}", "ERROR")
            return False
        except Exception as e:
            self.log(f"Command execution failed: {e}", "ERROR")
            return False
    
    def check_prerequisites(self):
        """Check if prerequisites are met"""
        self.log("Checking prerequisites...")
        
        # Check if we're in a virtual environment
        if not self.venv_python.exists():
            self.log("Virtual environment not found. Please run 'uv venv --python 3.10' first.", "ERROR")
            return False
            
        # Check if project dependencies are installed
        try:
            result = subprocess.run([str(self.venv_python), "-c", "import fastmcp"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                self.log("Project dependencies not installed. Please run 'uv pip install -e .' first.", "ERROR")
                return False
        except Exception as e:
            self.log(f"Failed to check project dependencies: {e}", "ERROR")
            return False
            
        self.log("Prerequisites check passed ✓")
        return True
    
    def install_system_dependencies(self):
        """Install system dependencies"""
        self.log("Installing system dependencies...")
        
        if check_dependencies():
            self.log("System dependencies already satisfied ✓")
            return True
            
        if not install_dependencies_interactive():
            self.log("Failed to install system dependencies", "ERROR")
            return False
            
        # Verify installation
        if not check_dependencies():
            self.log("System dependency installation verification failed", "ERROR")
            return False
            
        self.log("System dependencies installed successfully ✓")
        return True
    
    def setup_paraview_source(self):
        """Clone and setup ParaView source code"""
        self.log("Setting up ParaView source code...")
        
        if self.clean and self.paraview_src.exists():
            self.log("Cleaning existing ParaView source...")
            shutil.rmtree(self.paraview_src)
            
        if not self.paraview_src.exists():
            self.log("Cloning ParaView repository (this may take 5-15 minutes)...")
            self.log("Repository size: ~2GB, please be patient...")
            cmd = [
                "git", "clone", "--recursive", "--progress",
                "https://gitlab.kitware.com/paraview/paraview.git",
                str(self.paraview_src)
            ]
            if not self.run_command(cmd, cwd=self.parent_dir, show_progress=True):
                return False
                
        # Navigate to ParaView source
        os.chdir(self.paraview_src)
        
        # Checkout specific version
        self.log("Checking out ParaView v5.13.1...")
        if not self.run_command(["git", "checkout", "v5.13.1"], cwd=self.paraview_src):
            return False
            
        # Update submodules
        self.log("Updating git submodules (may take additional time)...")
        if not self.run_command(["git", "submodule", "update", "--init", "--recursive"], 
                               cwd=self.paraview_src, show_progress=True):
            return False
            
        self.log("ParaView source setup completed ✓")
        return True
    
    def configure_paraview_build(self):
        """Configure ParaView build with CMake"""
        self.log("Configuring ParaView build...")
        
        if self.clean and self.paraview_build.exists():
            self.log("Cleaning existing build directory...")
            shutil.rmtree(self.paraview_build)
            
        # Create build directory
        self.paraview_build.mkdir(exist_ok=True)
        
        # Prepare CMake command with all options from README
        python_executable = str(self.venv_python)
        install_prefix = str(self.paraview_install)
        
        # Check if ADIOS2 MPI libraries are available
        adios2_mpi_available = self.check_adios2_mpi_support()
        
        cmake_args = [
            "cmake", "-GNinja",
            f"-DPython3_EXECUTABLE={python_executable}",
            f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
            "-DPARAVIEW_USE_PYTHON=ON",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DPARAVIEW_ENABLE_ADIOS2=ON",
            "-DVTK_MODULE_ENABLE_VTK_IOADIOS2:STRING=DEFAULT",
            "-Wno-dev",  # Suppress developer warnings
        ]
        
        # Add MPI configuration based on ADIOS2 MPI availability
        if adios2_mpi_available:
            self.log("ADIOS2 MPI support detected - enabling ParaView MPI")
            cmake_args.extend([
                "-DPARAVIEW_USE_MPI=ON",
                "-DADIOS2_USE_MPI=ON"
            ])
        else:
            self.log("ADIOS2 MPI support not detected - building without MPI")
            cmake_args.extend([
                "-DPARAVIEW_USE_MPI=OFF",
                "-DADIOS2_USE_MPI=OFF"
            ])
            
        cmake_args.append(str(self.paraview_src))
        
        self.log(f"CMake configuration command: {' '.join(cmake_args)}")
        
        if not self.run_command(cmake_args, cwd=self.paraview_build):
            self.log("CMake configuration failed", "ERROR")
            return False
            
        self.log("ParaView build configuration completed ✓")
        return True
    
    def build_paraview(self):
        """Build and install ParaView"""
        self.log(f"Building ParaView with {self.jobs} parallel jobs...")
        self.log("This may take 1-3 hours depending on your system...")
        
        start_time = time.time()
        
        # Build ParaView
        build_cmd = ["ninja", f"-j{self.jobs}"]
        if not self.run_command(build_cmd, cwd=self.paraview_build):
            self.log("ParaView build failed", "ERROR")
            return False
            
        # Install ParaView
        self.log("Installing ParaView...")
        install_cmd = ["ninja", "install"]
        if not self.run_command(install_cmd, cwd=self.paraview_build):
            self.log("ParaView installation failed", "ERROR")
            return False
            
        elapsed = time.time() - start_time
        self.log(f"ParaView build and installation completed in {elapsed/60:.1f} minutes ✓")
        return True
    
    def check_adios2_mpi_support(self):
        """Check if ADIOS2 MPI libraries are available"""
        try:
            # Check for ADIOS2 MPI development libraries
            result = subprocess.run(
                ["pkg-config", "--exists", "adios2-mpi"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return True
                
            # Alternative check for ADIOS2 MPI libraries
            import glob
            mpi_libs = glob.glob("/usr/lib*/libadios2*mpi*") + \
                      glob.glob("/usr/local/lib*/libadios2*mpi*")
            return len(mpi_libs) > 0
            
        except Exception as e:
            self.log(f"ADIOS2 MPI check failed: {e}", "DEBUG")
            return False
    
    def setup_python_integration(self):
        """Setup Python path for ParaView integration"""
        self.log("Setting up Python integration...")
        
        # Create paraview.pth file
        paraview_site_packages = self.paraview_install / "lib" / "python3.10" / "site-packages"
        venv_site_packages = self.project_root / ".venv" / "lib" / "python3.10" / "site-packages"
        pth_file = venv_site_packages / "paraview.pth"
        
        if not paraview_site_packages.exists():
            self.log(f"ParaView Python packages not found at {paraview_site_packages}", "ERROR")
            return False
            
        # Write the path to .pth file
        with open(pth_file, 'w') as f:
            f.write(str(paraview_site_packages) + '\n')
            
        self.log(f"Created paraview.pth: {pth_file}")
        
        # Install numpy in virtual environment if not present
        self.log("Installing numpy in virtual environment...")
        numpy_cmd = ["uv", "add", "numpy"]
        if not self.run_command(numpy_cmd):
            self.log("Failed to install numpy", "ERROR")
            return False
            
        self.log("Python integration setup completed ✓")
        return True
    
    def verify_installation(self):
        """Verify the complete installation"""
        self.log("Verifying installation...")
        
        # Test ParaView import
        test_commands = [
            ("ParaView Simple", "import paraview.simple; print('✓ ParaView available')"),
            ("ADIOS2 Support", """
try:
    import vtk
    reader = vtk.vtkADIOS2CoreImageReader()
    print('✓ Native ADIOS2 support available')
except Exception as e:
    print(f'⚠ ADIOS2 support issue: {e}')
"""),
            ("MCP Server", "import sys; sys.path.insert(0, '.'); from src.server import main; print('✓ MCP server importable')")
        ]
        
        all_passed = True
        for test_name, test_code in test_commands:
            self.log(f"Testing {test_name}...")
            result = subprocess.run([str(self.venv_python), "-c", test_code],
                                  capture_output=True, text=True, cwd=self.project_root)
            if result.returncode == 0:
                self.log(f"  {result.stdout.strip()}")
            else:
                self.log(f"  ❌ {test_name} failed: {result.stderr.strip()}", "ERROR")
                all_passed = False
                
        if all_passed:
            self.log("All verification tests passed ✓")
        else:
            self.log("Some verification tests failed", "ERROR")
            
        return all_passed
    
    def run_complete_setup(self, steps=None):
        """Run the complete setup process"""
        default_steps = [
            ("check_prerequisites", "Checking prerequisites"),
            ("install_system_dependencies", "Installing system dependencies"),
            ("setup_paraview_source", "Setting up ParaView source"),
            ("configure_paraview_build", "Configuring ParaView build"),
            ("build_paraview", "Building ParaView"),
            ("setup_python_integration", "Setting up Python integration"),
            ("verify_installation", "Verifying installation")
        ]
        
        if steps is None:
            steps = default_steps
            
        self.log("🚀 Starting ParaView MCP automated setup")
        self.log(f"Project root: {self.project_root}")
        self.log(f"Build jobs: {self.jobs}")
        
        start_time = time.time()
        
        for step_func, step_name in steps:
            self.log(f"Step: {step_name}")
            if not getattr(self, step_func)():
                self.log(f"Setup failed at step: {step_name}", "ERROR")
                return False
                
        total_time = time.time() - start_time
        self.log(f"🎉 Complete setup finished successfully in {total_time/60:.1f} minutes!")
        
        # Print usage instructions
        self.print_usage_instructions()
        return True
    
    def print_usage_instructions(self):
        """Print usage instructions after successful setup"""
        print("\n" + "="*60)
        print("🎯 Setup Complete! Usage Instructions:")
        print("="*60)
        print("\n🚀 Quick Commands:")
        print("   uv run paraview-gui      # Start ParaView GUI")
        print("   uv run paraview-server   # Start ParaView server")
        print("   uv run paraview-mcp      # Start MCP server")
        print("\n📁 Direct paths:")
        print(f"   GUI: {self.paraview_install}/bin/paraview")
        print(f"   Server: {self.paraview_install}/bin/pvserver --multi-clients")
        print("\n🔧 Configuration:")
        print(f"   Project path: {self.project_root}")
        print("   Add to Claude Desktop config for MCP integration")
        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Automated ParaView MCP setup")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--clean", action="store_true",
                       help="Clean existing build and source directories")
    parser.add_argument("--jobs", "-j", type=int,
                       help="Number of parallel build jobs (default: CPU count)")
    parser.add_argument("--deps-only", action="store_true",
                       help="Only install system dependencies")
    parser.add_argument("--build-only", action="store_true",
                       help="Only build ParaView (skip dependency installation)")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify existing installation")
    
    args = parser.parse_args()
    
    # Determine project root
    project_root = Path(__file__).parent.parent
    
    # Create setup automator
    automator = SetupAutomator(
        project_root=project_root,
        verbose=args.verbose,
        clean=args.clean,
        jobs=args.jobs
    )
    
    # Determine which steps to run
    if args.deps_only:
        steps = [
            ("check_prerequisites", "Checking prerequisites"),
            ("install_system_dependencies", "Installing system dependencies")
        ]
    elif args.build_only:
        steps = [
            ("check_prerequisites", "Checking prerequisites"),
            ("setup_paraview_source", "Setting up ParaView source"),
            ("configure_paraview_build", "Configuring ParaView build"), 
            ("build_paraview", "Building ParaView"),
            ("setup_python_integration", "Setting up Python integration"),
            ("verify_installation", "Verifying installation")
        ]
    elif args.verify_only:
        steps = [
            ("verify_installation", "Verifying installation")
        ]
    else:
        steps = None  # Run all steps
    
    # Run the setup
    success = automator.run_complete_setup(steps)
    
    if not success:
        sys.exit(1)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())