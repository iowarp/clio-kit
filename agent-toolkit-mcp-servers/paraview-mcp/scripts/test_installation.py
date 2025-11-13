#!/usr/bin/env python3
"""
Comprehensive test script to validate ParaView MCP installation
Enhanced with detailed diagnostics and BP5 file testing
"""
import subprocess
import sys
import time
from pathlib import Path

class InstallationTester:
    def __init__(self, project_root):
        self.project_root = Path(project_root).resolve()
        self.passed = 0
        self.total = 0
        self.results = {}
        
    def run_test(self, command, description, category="General", timeout=30, allow_fail=False):
        """Run a test command and report results"""
        print(f"\n🧪 Testing: {description}")
        print(f"   Command: {command}")
        
        self.total += 1
        
        try:
            if isinstance(command, str):
                result = subprocess.run(command, shell=True, capture_output=True, 
                                      text=True, timeout=timeout, cwd=self.project_root)
            else:
                result = subprocess.run(command, capture_output=True, text=True, 
                                      timeout=timeout, cwd=self.project_root)
                                      
            if result.returncode == 0:
                print(f"   ✅ PASS: {description}")
                if result.stdout.strip():
                    print(f"   Output: {result.stdout.strip()}")
                self.passed += 1
                self.results[description] = {"status": "PASS", "category": category}
                return True
            else:
                status = "EXPECTED_FAIL" if allow_fail else "FAIL"
                print(f"   ❌ {status}: {description}")
                if result.stderr.strip():
                    print(f"   Error: {result.stderr.strip()}")
                if result.stdout.strip():
                    print(f"   Output: {result.stdout.strip()}")
                    
                if not allow_fail:
                    self.results[description] = {"status": "FAIL", "category": category, 
                                               "error": result.stderr.strip()}
                else:
                    self.results[description] = {"status": "EXPECTED_FAIL", "category": category}
                    
                return allow_fail  # Return True for expected failures
                
        except subprocess.TimeoutExpired:
            print(f"   ⏱️ TIMEOUT: {description}")
            self.results[description] = {"status": "TIMEOUT", "category": category}
            return False
        except Exception as e:
            print(f"   ❌ ERROR: {description} - {e}")
            self.results[description] = {"status": "ERROR", "category": category, "error": str(e)}
            return False
    
    def test_basic_environment(self):
        """Test basic environment and dependencies"""
        print("\n" + "="*60)
        print("🔧 BASIC ENVIRONMENT TESTS")
        print("="*60)
        
        # UV package manager
        self.run_test("uv --version", "UV package manager", "Environment")
        
        # Python virtual environment
        self.run_test("python --version", "Python in virtual environment", "Environment")
        
        # Project dependencies
        self.run_test("python -c 'import fastmcp; print(f\"FastMCP v{fastmcp.__version__}\")'", 
                     "FastMCP dependency", "Environment")
        
        # System dependencies check
        self.run_test("uv run install-deps --check-only", "System dependencies", "Environment")
    
    def test_paraview_core(self):
        """Test ParaView core functionality"""
        print("\n" + "="*60)
        print("🎨 PARAVIEW CORE TESTS")
        print("="*60)
        
        # Basic ParaView import
        self.run_test("python -c 'import paraview; print(f\"ParaView available at: {paraview.__file__}\")'",
                     "ParaView Python package", "ParaView")
        
        # ParaView Simple module
        self.run_test("python -c 'import paraview.simple as pv; print(\"ParaView Simple module loaded\")'",
                     "ParaView Simple API", "ParaView")
        
        # ParaView version
        self.run_test("python -c 'import paraview; print(f\"ParaView version: {paraview.version}\")'",
                     "ParaView version info", "ParaView", allow_fail=True)
    
    def test_vtk_adios2(self):
        """Test VTK and ADIOS2 integration"""
        print("\n" + "="*60)
        print("🔬 VTK & ADIOS2 INTEGRATION TESTS")
        print("="*60)
        
        # VTK import
        self.run_test("python -c 'import vtk; print(f\"VTK version: {vtk.vtkVersion.GetVTKVersion()}\")'",
                     "VTK library", "VTK")
        
        # ADIOS2 Core Reader
        self.run_test("python -c 'import vtk; reader = vtk.vtkADIOS2CoreImageReader(); print(\"✓ ADIOS2 Core Reader available\")'",
                     "ADIOS2 Core Image Reader", "ADIOS2")
        
        # ADIOS2 VTX Writer (if available)
        self.run_test("python -c 'import vtk; writer = vtk.vtkADIOS2VTXWriter(); print(\"✓ ADIOS2 VTX Writer available\")'",
                     "ADIOS2 VTX Writer", "ADIOS2", allow_fail=True)
        
        # Test more VTK ADIOS2 classes
        adios2_classes = [
            ("vtkADIOS2CoreImageReader", "Core Image Reader"),
            ("vtkADIOS2VTXReader", "VTX Reader"),
        ]
        
        for class_name, desc in adios2_classes:
            self.run_test(f"python -c 'import vtk; getattr(vtk, \"{class_name}\"); print(\"✓ {class_name} available\")'",
                         f"ADIOS2 {desc}", "ADIOS2", allow_fail=True)
    
    def test_bp5_file_handling(self):
        """Test BP5 file handling capabilities"""
        print("\n" + "="*60)
        print("📁 BP5 FILE HANDLING TESTS")
        print("="*60)
        
        # Check for example BP5 files
        bp5_dataset_dir = self.project_root / "bp5-dataset-collection"
        if bp5_dataset_dir.exists():
            print(f"   📂 Found BP5 dataset collection: {bp5_dataset_dir}")
            
            # Find first BP5 file for testing
            bp5_files = list(bp5_dataset_dir.rglob("*.bp5"))
            if bp5_files:
                test_file = bp5_files[0]
                print(f"   📄 Testing with file: {test_file}")
                
                # Test file reading capability
                test_code = f"""
import vtk
reader = vtk.vtkADIOS2CoreImageReader()
reader.SetFileName('{test_file}')
reader.Update()
print(f'✓ Successfully read BP5 file: {test_file.name}')
output = reader.GetOutput()
print(f'  Points: {{output.GetNumberOfPoints()}}')
print(f'  Cells: {{output.GetNumberOfCells()}}')
"""
                self.run_test(f'python -c "{test_code}"', "BP5 file reading", "BP5", timeout=60)
            else:
                print("   ⚠️  No BP5 files found in dataset collection")
        else:
            print("   ⚠️  No BP5 dataset collection found")
    
    def test_mcp_server(self):
        """Test MCP server functionality"""
        print("\n" + "="*60)
        print("🌐 MCP SERVER TESTS")
        print("="*60)
        
        # Test MCP server help
        self.run_test("uv run paraview-mcp --help", "MCP server help", "MCP", allow_fail=True)
        
        # Test server import (without starting)
        self.run_test("python -c 'import sys; sys.path.insert(0, \"src\"); import server; print(\"✓ MCP server module importable\")'",
                     "MCP server module import", "MCP", allow_fail=True)
    
    def test_paraview_binaries(self):
        """Test ParaView binary executables"""
        print("\n" + "="*60)
        print("🖥️  PARAVIEW BINARY TESTS")  
        print("="*60)
        
        paraview_install = self.project_root / ".paraview"
        if paraview_install.exists():
            # Test ParaView GUI binary
            paraview_bin = paraview_install / "bin" / "paraview"
            if paraview_bin.exists():
                self.run_test(f"{paraview_bin} --version", "ParaView GUI binary", "Binaries", timeout=10, allow_fail=True)
            
            # Test ParaView server binary  
            pvserver_bin = paraview_install / "bin" / "pvserver"
            if pvserver_bin.exists():
                self.run_test(f"{pvserver_bin} --version", "ParaView server binary", "Binaries", timeout=10, allow_fail=True)
        else:
            print("   ⚠️  No local ParaView installation found (.paraview directory missing)")
    
    def run_all_tests(self):
        """Run all test suites"""
        start_time = time.time()
        
        print("🔬 ParaView MCP Comprehensive Installation Validation")
        print("="*70)
        print(f"Project root: {self.project_root}")
        
        # Run test suites
        self.test_basic_environment()
        self.test_paraview_core() 
        self.test_vtk_adios2()
        self.test_bp5_file_handling()
        self.test_mcp_server()
        self.test_paraview_binaries()
        
        # Final results
        elapsed = time.time() - start_time
        print("\n" + "="*70)
        print("📊 FINAL TEST RESULTS")
        print("="*70)
        print(f"⏱️  Total time: {elapsed:.1f} seconds")
        print(f"📈 Tests passed: {self.passed}/{self.total}")
        print(f"📉 Success rate: {(self.passed/self.total)*100:.1f}%")
        
        # Categorized results
        categories = {}
        for test, result in self.results.items():
            cat = result["category"]
            if cat not in categories:
                categories[cat] = {"passed": 0, "total": 0}
            categories[cat]["total"] += 1
            if result["status"] in ["PASS", "EXPECTED_FAIL"]:
                categories[cat]["passed"] += 1
        
        print("\n📂 Results by category:")
        for category, stats in categories.items():
            success_rate = (stats["passed"] / stats["total"]) * 100
            status = "✅" if success_rate >= 80 else "⚠️" if success_rate >= 50 else "❌"
            print(f"   {status} {category}: {stats['passed']}/{stats['total']} ({success_rate:.0f}%)")
        
        # Critical failures
        critical_failures = [test for test, result in self.results.items() 
                           if result["status"] == "FAIL" and result["category"] in ["Environment", "ParaView", "VTK"]]
        
        if critical_failures:
            print("\n❌ Critical failures found:")
            for failure in critical_failures:
                print(f"   • {failure}")
        
        # Overall status
        critical_success_rate = (self.passed / self.total) * 100
        if critical_success_rate >= 80:
            print("\n🎉 Installation appears to be working well!")
            if critical_success_rate < 100:
                print("   Minor issues detected but core functionality should work.")
        elif critical_success_rate >= 50:
            print("\n⚠️  Installation has significant issues that should be addressed.")
            print("   Some functionality may not work correctly.")
        else:
            print("\n❌ Installation has major problems and likely won't work.")
            print("   Please check the installation process and dependencies.")
        
        return 0 if critical_success_rate >= 80 else 1

def main():
    """Main test function"""
    project_root = Path(__file__).parent.parent
    tester = InstallationTester(project_root)
    return tester.run_all_tests()

if __name__ == "__main__":
    sys.exit(main())
