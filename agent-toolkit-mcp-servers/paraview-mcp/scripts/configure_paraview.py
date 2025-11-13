#!/usr/bin/env python3
"""
Detect and configure existing ParaView installations
Enhanced post-build configuration and verification
"""
import subprocess
import sys
import os
from pathlib import Path

def find_system_paraview():
    """Find system ParaView installation"""
    try:
        # Try to import paraview from system Python
        result = subprocess.run([
            'python3', '-c', 'import paraview; print(paraview.__file__)'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            paraview_path = result.stdout.strip()
            paraview_dir = str(Path(paraview_path).parent)
            return paraview_dir
    except:
        pass
    
    return None

def find_conda_paraview():
    """Find conda ParaView installation"""
    try:
        # Check if we're in a conda environment
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            paraview_path = Path(conda_prefix) / 'lib' / 'python3.10' / 'site-packages' / 'paraview'
            if paraview_path.exists():
                return str(paraview_path)
    except:
        pass
    
    return None

def find_built_paraview(project_root):
    """Find locally built ParaView installation"""
    project_root = Path(project_root).resolve()
    paraview_install = project_root / ".paraview"
    paraview_site_packages = paraview_install / "lib" / "python3.10" / "site-packages"
    
    if paraview_site_packages.exists():
        return str(paraview_site_packages)
    
    return None

def verify_adios2_support(verbose=False):
    """Verify ADIOS2 support in ParaView"""
    test_commands = [
        ("ADIOS2 Import", "import vtk; print('VTK available')"),
        ("ADIOS2 Reader", "import vtk; reader = vtk.vtkADIOS2CoreImageReader(); print('✓ ADIOS2 reader available')"),
        ("ADIOS2 Writer", "import vtk; writer = vtk.vtkADIOS2VTXWriter(); print('✓ ADIOS2 writer available')")
    ]
    
    results = {}
    for test_name, test_code in test_commands:
        try:
            result = subprocess.run([
                sys.executable, "-c", test_code
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                results[test_name] = True
                if verbose:
                    print(f"✅ {test_name}: {result.stdout.strip()}")
            else:
                results[test_name] = False
                if verbose:
                    print(f"❌ {test_name}: {result.stderr.strip()}")
        except Exception as e:
            results[test_name] = False
            if verbose:
                print(f"❌ {test_name}: {e}")
    
    return results

def setup_desktop_integration(project_root):
    """Setup desktop integration for ParaView"""
    project_root = Path(project_root).resolve()
    paraview_bin = project_root / ".paraview" / "bin" / "paraview"
    
    if not paraview_bin.exists():
        return False
        
    # Create convenience scripts
    scripts_dir = project_root / "scripts"
    
    # ParaView launcher script
    paraview_launcher = scripts_dir / "launch_paraview.sh"
    with open(paraview_launcher, 'w') as f:
        f.write(f"""#!/bin/bash
# ParaView GUI launcher
cd "{project_root}"
"{paraview_bin}" "$@"
""")
    paraview_launcher.chmod(0o755)
    
    # ParaView server launcher script
    pvserver_launcher = scripts_dir / "launch_pvserver.sh"
    pvserver_bin = project_root / ".paraview" / "bin" / "pvserver"
    with open(pvserver_launcher, 'w') as f:
        f.write(f"""#!/bin/bash
# ParaView server launcher
cd "{project_root}"
echo "Starting ParaView server on localhost:11111"
echo "Connect from ParaView GUI: File -> Connect -> localhost:11111"
"{pvserver_bin}" --multi-clients "$@"
""")
    pvserver_launcher.chmod(0o755)
    
    print("✅ Created launcher scripts:")
    print(f"   ParaView GUI: {paraview_launcher}")
    print(f"   ParaView Server: {pvserver_launcher}")
    
    return True

def setup_paraview_path(paraview_dir, venv_path):
    """Setup paraview.pth file"""
    if not paraview_dir:
        return False
    
    pth_file = Path(venv_path) / 'lib' / 'python3.10' / 'site-packages' / 'paraview.pth'
    
    try:
        with open(pth_file, 'w') as f:
            f.write(f"{paraview_dir}\n")
            # Also add parent directory for vtk modules
            f.write(f"{Path(paraview_dir).parent}\n")
        
        print(f"✅ Created paraview.pth: {pth_file}")
        print(f"   Added path: {paraview_dir}")
        return True
    except Exception as e:
        print(f"❌ Failed to create paraview.pth: {e}")
        return False

def test_paraview_import():
    """Test if ParaView can be imported"""
    try:
        import paraview.simple
        print("✅ ParaView import successful")
        return True
    except ImportError as e:
        print(f"❌ ParaView import failed: {e}")
        return False

def main():
    """Main configuration function"""
    print("🔍 ParaView Installation Detector & Configurator")
    print("=" * 50)
    
    project_root = Path('.').resolve()
    
    # Check for existing virtual environment
    venv_path = project_root / '.venv'
    if not venv_path.exists():
        print("❌ No virtual environment found. Run 'uv venv --python 3.10' first.")
        return 1
    
    print(f"📁 Project root: {project_root}")
    print(f"📁 Using virtual environment: {venv_path}")
    
    # First check if ParaView is already configured
    print("\n🔍 Testing current ParaView configuration...")
    if test_paraview_import():
        print("✅ ParaView is already working!")
        
        # Test ADIOS2 support
        print("\n🔍 Testing ADIOS2 support...")
        adios2_results = verify_adios2_support(verbose=True)
        
        all_adios2_working = all(adios2_results.values())
        if all_adios2_working:
            print("✅ ADIOS2 support fully functional!")
        else:
            print("⚠️  Some ADIOS2 features may be limited")
            
        # Setup desktop integration if built locally
        built_paraview = find_built_paraview(project_root)
        if built_paraview:
            setup_desktop_integration(project_root)
            
        print("🎉 ParaView configuration complete!")
        return 0
    
    print("❌ ParaView not available, searching for installations...")
    print("\n🔍 Searching for ParaView installations...")
    
    # Look for locally built ParaView first
    built_paraview = find_built_paraview(project_root)
    if built_paraview:
        print(f"🔨 Found locally built ParaView: {built_paraview}")
        if setup_paraview_path(built_paraview, venv_path):
            if test_paraview_import():
                print("✅ Successfully configured locally built ParaView!")
                
                # Test ADIOS2 and setup integration
                adios2_results = verify_adios2_support(verbose=True)
                setup_desktop_integration(project_root)
                
                print("🎉 Local ParaView build configuration complete!")
                return 0
    
    # Look for conda installation
    conda_paraview = find_conda_paraview()
    if conda_paraview:
        print(f"📦 Found conda ParaView: {conda_paraview}")
        if setup_paraview_path(conda_paraview, venv_path):
            if test_paraview_import():
                print("✅ Successfully configured conda ParaView!")
                verify_adios2_support(verbose=True)
                return 0
    
    # Look for system installation
    system_paraview = find_system_paraview()
    if system_paraview:
        print(f"🖥️ Found system ParaView: {system_paraview}")
        if setup_paraview_path(system_paraview, venv_path):
            if test_paraview_import():
                print("✅ Successfully configured system ParaView!")
                verify_adios2_support(verbose=True)
                return 0
    
    # No ParaView found
    print("\n❌ No ParaView installation found.")
    print("\nRecommended installation methods (in order of preference):")
    print("1. 🔨 Build with full ADIOS2 support: uv run automate-setup")
    print("2. 📦 Conda with ADIOS2: conda install -c conda-forge paraview adios2")
    print("3. 🖥️  System package: sudo apt install paraview python3-paraview")
    print("4. 🔧 Manual build: uv run build-paraview")
    
    print("\n💡 For best ADIOS2/BP5 support, use option 1 (automated build).")
    
    return 1

if __name__ == "__main__":
    sys.exit(main())
