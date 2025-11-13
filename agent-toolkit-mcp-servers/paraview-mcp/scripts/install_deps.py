#!/usr/bin/env python3
"""
System dependency checker and installer helper for ParaView MCP
"""

import sys
import subprocess
import platform
import shutil
import argparse


def run_command(cmd, check=False):
    """Run a shell command"""
    print(f"Running: {cmd}")
    if isinstance(cmd, str):
        cmd = cmd.split()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Command failed: {result.stderr}")
        return False
    return result.returncode == 0


def check_command_exists(command):
    """Check if a command exists in PATH"""
    return shutil.which(command) is not None


def get_linux_distribution():
    """Get Linux distribution information"""
    try:
        with open('/etc/os-release', 'r') as f:
            lines = f.readlines()
        
        info = {}
        for line in lines:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                info[key] = value.strip('"')
        
        return info.get('ID', 'unknown').lower()
    except:
        return 'unknown'


def install_ubuntu_dependencies():
    """Install dependencies on Ubuntu/Debian systems"""
    print("Installing Ubuntu/Debian dependencies...")
    
    packages = [
        "git", "cmake", "build-essential", "ninja-build",
        "libgl1-mesa-dev", "libxt-dev", "python3-dev", "python3-numpy",
        "libopenmpi-dev", "libtbb-dev", "qtbase5-dev",
        "libadios2-mpi-core-dev", "libadios2-mpi-c++11-dev",
        "libadios2-serial-core-dev", "libadios2-serial-c++11-dev"
    ]
    
    # Update package list
    if not run_command("sudo apt update", check=True):
        return False
    
    # Install packages
    cmd = ["sudo", "apt", "install", "-y"] + packages
    return run_command(cmd, check=True)


def install_fedora_dependencies():
    """Install dependencies on Fedora/RHEL systems"""
    print("Installing Fedora/RHEL dependencies...")
    
    packages = [
        "git", "cmake", "gcc-c++", "ninja-build",
        "mesa-libGL-devel", "libXt-devel", "python3-devel", "python3-numpy",
        "openmpi-devel", "tbb-devel", "qt5-qtbase-devel",
        "adios2-devel", "adios2-openmpi-devel"
    ]
    
    cmd = ["sudo", "dnf", "install", "-y"] + packages
    return run_command(cmd, check=True)


def install_arch_dependencies():
    """Install dependencies on Arch Linux"""
    print("Installing Arch Linux dependencies...")
    
    packages = [
        "git", "cmake", "gcc", "ninja",
        "mesa", "libxt", "python", "python-numpy",
        "openmpi", "tbb", "qt5-base",
        "adios2"
    ]
    
    cmd = ["sudo", "pacman", "-S", "--noconfirm"] + packages
    return run_command(cmd, check=True)


def check_dependencies():
    """Check if all required dependencies are available"""
    print("Checking system dependencies...")
    
    required_commands = {
        'git': 'Git version control',
        'cmake': 'CMake build system',
        'ninja': 'Ninja build tool',
        'gcc': 'GCC compiler (or equivalent)',
        'g++': 'G++ compiler (or equivalent)'
    }
    
    missing = []
    for cmd, desc in required_commands.items():
        if not check_command_exists(cmd):
            missing.append(f"{cmd} ({desc})")
    
    # Check for pkg-config to verify libraries
    libraries_to_check = [
        ('gl', 'OpenGL development libraries'),
        ('xt', 'X11 Toolkit libraries'),
        ('ompi', 'OpenMPI development libraries'),
        ('tbb', 'Threading Building Blocks'),
        ('Qt5Core', 'Qt5 development libraries')
    ]
    
    missing_libs = []
    if check_command_exists('pkg-config'):
        for lib, desc in libraries_to_check:
            result = subprocess.run(['pkg-config', '--exists', lib], 
                                  capture_output=True)
            if result.returncode != 0:
                missing_libs.append(f"{lib} ({desc})")
    
    if missing or missing_libs:
        print("\n❌ Missing dependencies:")
        for item in missing:
            print(f"  - {item}")
        for item in missing_libs:
            print(f"  - {item}")
        return False
    else:
        print("✅ All required dependencies are available!")
        return True


def install_dependencies_interactive():
    """Interactive dependency installation"""
    system = platform.system().lower()
    
    if system != 'linux':
        print("❌ Automatic dependency installation is only supported on Linux.")
        print(f"Please install dependencies manually for {system}.")
        return False
    
    distro = get_linux_distribution()
    print(f"Detected Linux distribution: {distro}")
    
    if distro in ['ubuntu', 'debian', 'pop', 'mint']:
        return install_ubuntu_dependencies()
    elif distro in ['fedora', 'rhel', 'centos', 'rocky', 'almalinux']:
        return install_fedora_dependencies()
    elif distro in ['arch', 'manjaro', 'endeavouros']:
        return install_arch_dependencies()
    else:
        print(f"❌ Unsupported distribution: {distro}")
        print("Please install dependencies manually:")
        print_manual_instructions()
        return False


def print_manual_instructions():
    """Print manual installation instructions"""
    print("\nManual installation instructions:")
    print("\nUbuntu/Debian:")
    print("sudo apt update && sudo apt install git cmake build-essential ninja-build \\")
    print("    libgl1-mesa-dev libxt-dev python3-dev python3-numpy \\")
    print("    libopenmpi-dev libtbb-dev qtbase5-dev \\")
    print("    libadios2-mpi-core-dev libadios2-mpi-c++11-dev \\")
    print("    libadios2-serial-core-dev libadios2-serial-c++11-dev")
    
    print("\nFedora/RHEL:")
    print("sudo dnf install git cmake gcc-c++ ninja-build \\")
    print("    mesa-libGL-devel libXt-devel python3-devel python3-numpy \\")
    print("    openmpi-devel tbb-devel qt5-qtbase-devel \\")
    print("    adios2-devel adios2-openmpi-devel")
    
    print("\nArch Linux:")
    print("sudo pacman -S git cmake gcc ninja mesa libxt python python-numpy \\")
    print("    openmpi tbb qt5-base adios2")


def main():
    parser = argparse.ArgumentParser(description="Check and install system dependencies for ParaView MCP")
    parser.add_argument("--install", action="store_true",
                       help="Attempt to install missing dependencies automatically")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check dependencies, don't install")
    
    args = parser.parse_args()
    
    print("🔍 ParaView MCP System Dependencies Checker")
    print("=" * 50)
    
    # Check current dependencies
    deps_ok = check_dependencies()
    
    if deps_ok:
        print("\n✅ All dependencies are satisfied!")
        return 0
    
    if args.check_only:
        print("\n❌ Some dependencies are missing. Run with --install to attempt automatic installation.")
        return 1
    
    if args.install:
        print("\n🔧 Attempting to install missing dependencies...")
        if install_dependencies_interactive():
            print("\n✅ Dependencies installed successfully!")
            print("Please run the dependency check again to verify.")
            return 0
        else:
            print("\n❌ Failed to install dependencies automatically.")
            print_manual_instructions()
            return 1
    else:
        print("\n💡 To install dependencies automatically, run:")
        print("    uv run install-deps --install")
        print("\nOr install manually using the instructions above.")
        print_manual_instructions()
        return 1


if __name__ == "__main__":
    sys.exit(main())
