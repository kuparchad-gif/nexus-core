import os
import sys
import json
import importlib
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("diagnostic")

class SystemDiagnostic:
    """Diagnostic tool for the Nexus system"""
    
    def __init__(self):
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.results = {
            "file_checks": {},
            "import_checks": {},
            "memory_checks": {},
            "environment_checks": {}
        }
    
    def run_diagnostics(self):
        """Run all diagnostics"""
        print("\n===== NEXUS SYSTEM DIAGNOSTICS =====\n")
        
        self.check_required_files()
        self.check_imports()
        self.check_memory_files()
        self.check_environment()
        
        print("\n===== DIAGNOSTIC SUMMARY =====\n")
        self.print_summary()
        
        return self.results
    
    def check_required_files(self):
        """Check if required files exist"""
        print("Checking required files...")
        
        required_files = [
            "standardized_pod.py",
            "quantum_translator.py",
            "emotional_processor.py",
            "frequency_protocol.py",
            "caas_interface.py",
            "pod_manager.py",
            "run_lillith.py",
            "run_system.py"
        ]
        
        for file in required_files:
            file_path = os.path.join(self.root_dir, file)
            exists = os.path.exists(file_path)
            self.results["file_checks"][file] = exists
            status = "✓" if exists else "✗"
            print(f"  {status} {file}")
    
    def check_imports(self):
        """Check if required modules can be imported"""
        print("\nChecking imports...")
        
        # Check custom modules
        custom_modules = [
            "standardized_pod",
            "quantum_translator",
            "emotional_processor",
            "frequency_protocol",
            "caas_interface",
            "pod_manager"
        ]
        
        for module in custom_modules:
            try:
                # Add the current directory to sys.path if not already there
                if self.root_dir not in sys.path:
                    sys.path.append(self.root_dir)
                
                # Try to import the module
                importlib.import_module(module)
                self.results["import_checks"][module] = True
                print(f"  ✓ {module}")
            except ImportError as e:
                self.results["import_checks"][module] = False
                print(f"  ✗ {module}: {str(e)}")
        
        # Check required Python packages
        required_packages = [
            "numpy",
            "requests"
        ]
        
        for package in required_packages:
            try:
                importlib.import_module(package)
                self.results["import_checks"][package] = True
                print(f"  ✓ {package}")
            except ImportError:
                self.results["import_checks"][package] = False
                print(f"  ✗ {package}")
    
    def check_memory_files(self):
        """Check if memory files exist and are valid JSON"""
        print("\nChecking memory files...")
        
        memory_dir = os.path.join(self.root_dir, "memory")
        if not os.path.exists(memory_dir):
            print(f"  ✗ Memory directory not found: {memory_dir}")
            self.results["memory_checks"]["directory"] = False
            return
        
        self.results["memory_checks"]["directory"] = True
        print(f"  ✓ Memory directory found")
        
        # Check nova_memory.json
        nova_memory_path = os.path.join(memory_dir, "nova_memory.json")
        if os.path.exists(nova_memory_path):
            try:
                with open(nova_memory_path, 'r') as f:
                    json.load(f)
                self.results["memory_checks"]["nova_memory.json"] = True
                print(f"  ✓ nova_memory.json (valid JSON)")
            except json.JSONDecodeError:
                self.results["memory_checks"]["nova_memory.json"] = False
                print(f"  ✗ nova_memory.json (invalid JSON)")
        else:
            self.results["memory_checks"]["nova_memory.json"] = False
            print(f"  ✗ nova_memory.json (not found)")
        
        # Check bootstrap/genesis directory
        genesis_dir = os.path.join(memory_dir, "bootstrap", "genesis")
        if os.path.exists(genesis_dir):
            self.results["memory_checks"]["genesis_directory"] = True
            print(f"  ✓ bootstrap/genesis directory found")
            
            # Check birth records
            for record in ["nova_birth_record.json", "lillith_birth_record.json", "system_manifest.json"]:
                record_path = os.path.join(genesis_dir, record)
                if os.path.exists(record_path):
                    try:
                        with open(record_path, 'r') as f:
                            json.load(f)
                        self.results["memory_checks"][record] = True
                        print(f"  ✓ {record} (valid JSON)")
                    except json.JSONDecodeError:
                        self.results["memory_checks"][record] = False
                        print(f"  ✗ {record} (invalid JSON)")
                else:
                    self.results["memory_checks"][record] = False
                    print(f"  ✗ {record} (not found)")
        else:
            self.results["memory_checks"]["genesis_directory"] = False
            print(f"  ✗ bootstrap/genesis directory not found")
    
    def check_environment(self):
        """Check environment details"""
        print("\nChecking environment...")
        
        # Check Python version
        python_version = sys.version
        self.results["environment_checks"]["python_version"] = python_version
        print(f"  Python version: {python_version}")
        
        # Check OS
        import platform
        os_name = platform.system()
        self.results["environment_checks"]["os"] = os_name
        print(f"  Operating system: {os_name}")
        
        # Check available memory
        try:
            import psutil
            available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
            self.results["environment_checks"]["available_memory"] = f"{available_memory:.2f} MB"
            print(f"  Available memory: {available_memory:.2f} MB")
        except ImportError:
            self.results["environment_checks"]["available_memory"] = "Unknown (psutil not installed)"
            print(f"  Available memory: Unknown (psutil not installed)")
        
        # Check if GPU is available
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            self.results["environment_checks"]["gpu_available"] = gpu_available
            print(f"  GPU available: {gpu_available}")
        except ImportError:
            self.results["environment_checks"]["gpu_available"] = "Unknown (torch not installed)"
            print(f"  GPU available: Unknown (torch not installed)")
    
    def print_summary(self):
        """Print diagnostic summary"""
        # Count successes and failures
        file_success = sum(1 for result in self.results["file_checks"].values() if result)
        file_total = len(self.results["file_checks"])
        
        import_success = sum(1 for result in self.results["import_checks"].values() if result)
        import_total = len(self.results["import_checks"])
        
        memory_success = sum(1 for result in self.results["memory_checks"].values() if result)
        memory_total = len(self.results["memory_checks"])
        
        # Print summary
        print(f"Required files: {file_success}/{file_total} found")
        print(f"Required imports: {import_success}/{import_total} successful")
        print(f"Memory files: {memory_success}/{memory_total} valid")
        
        # Overall status
        if file_success == file_total and import_success == import_total and memory_success == memory_total:
            print("\nDIAGNOSTIC RESULT: SYSTEM READY")
        else:
            print("\nDIAGNOSTIC RESULT: ISSUES DETECTED")
            print("Please fix the issues marked with ✗ above before running the system.")

if __name__ == "__main__":
    diagnostic = SystemDiagnostic()
    diagnostic.run_diagnostics()