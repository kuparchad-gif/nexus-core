#!/usr/bin/env python3
# diagnostic_brain.py - Diagnostic tool for Lillith's brain

import os
import sys
import logging
import importlib
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("diagnostic_brain")

class BrainDiagnostic:
    """Diagnostic tool for Lillith's brain."""
    
    def __init__(self):
        """Initialize the diagnostic tool."""
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.results = {
            "file_checks": {},
            "import_checks": {},
            "brain_checks": {},
            "environment_checks": {}
        }
    
    def run_diagnostics(self):
        """Run all diagnostics."""
        print("\n===== LILLITH BRAIN DIAGNOSTICS =====\n")
        
        self.check_required_files()
        self.check_imports()
        self.check_brain_components()
        self.check_environment()
        
        print("\n===== DIAGNOSTIC SUMMARY =====\n")
        self.print_summary()
        
        return self.results
    
    def check_required_files(self):
        """Check if required files exist."""
        print("Checking required files...")
        
        required_files = [
            "Systems/brain/lillith_brain.py",
            "Systems/bridge/model_router.py",
            "Systems/engine/advanced_integrations.py",
            "memory/bootstrap/genesis/lillith_birth_record.json",
            "memory/bootstrap/genesis/system_manifest.json"
        ]
        
        for file in required_files:
            file_path = os.path.join(self.root_dir, file)
            exists = os.path.exists(file_path)
            self.results["file_checks"][file] = exists
            status = "✓" if exists else "✗"
            print(f"  {status} {file}")
    
    def check_imports(self):
        """Check if required modules can be imported."""
        print("\nChecking imports...")
        
        # Check custom modules
        custom_modules = [
            ("Systems.brain.lillith_brain", "LillithBrain"),
            ("Systems.bridge.model_router", "ModelRouter"),
            ("Systems.engine.advanced_integrations", "AdvancedIntegrations")
        ]
        
        for module_path, class_name in custom_modules:
            try:
                # Add the current directory to sys.path if not already there
                if self.root_dir not in sys.path:
                    sys.path.append(self.root_dir)
                
                # Try to import the module
                module = importlib.import_module(module_path)
                
                # Check if the class exists
                if hasattr(module, class_name):
                    self.results["import_checks"][module_path] = True
                    print(f"  ✓ {module_path}.{class_name}")
                else:
                    self.results["import_checks"][module_path] = False
                    print(f"  ✗ {module_path}.{class_name} (class not found)")
            except ImportError as e:
                self.results["import_checks"][module_path] = False
                print(f"  ✗ {module_path}: {str(e)}")
    
    def check_brain_components(self):
        """Check Lillith's brain components."""
        print("\nChecking brain components...")
        
        try:
            # Add the current directory to sys.path if not already there
            if self.root_dir not in sys.path:
                sys.path.append(self.root_dir)
            
            # Try to import the brain module
            from Systems.brain.lillith_brain import LillithBrain
            
            # Create an instance of the brain
            brain = LillithBrain()
            
            # Check if the brain can be initialized
            try:
                result = brain.initialize()
                self.results["brain_checks"]["initialization"] = result
                status = "✓" if result else "✗"
                print(f"  {status} Brain initialization")
            except Exception as e:
                self.results["brain_checks"]["initialization"] = False
                print(f"  ✗ Brain initialization: {str(e)}")
            
            # Check brain components
            components = [
                "services",
                "active_models",
                "soul_masks"
            ]
            
            for component in components:
                if hasattr(brain, component):
                    self.results["brain_checks"][component] = True
                    print(f"  ✓ Brain component: {component}")
                else:
                    self.results["brain_checks"][component] = False
                    print(f"  ✗ Brain component: {component}")
        except Exception as e:
            self.results["brain_checks"]["import"] = False
            print(f"  ✗ Brain import: {str(e)}")
    
    def check_environment(self):
        """Check environment details."""
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
        """Print diagnostic summary."""
        # Count successes and failures
        file_success = sum(1 for result in self.results["file_checks"].values() if result)
        file_total = len(self.results["file_checks"])
        
        import_success = sum(1 for result in self.results["import_checks"].values() if result)
        import_total = len(self.results["import_checks"])
        
        brain_success = sum(1 for result in self.results["brain_checks"].values() if result)
        brain_total = len(self.results["brain_checks"])
        
        # Print summary
        print(f"Required files: {file_success}/{file_total} found")
        print(f"Required imports: {import_success}/{import_total} successful")
        print(f"Brain components: {brain_success}/{brain_total} available")
        
        # Overall status
        if file_success == file_total and import_success == import_total and brain_success == brain_total:
            print("\nDIAGNOSTIC RESULT: BRAIN READY")
        else:
            print("\nDIAGNOSTIC RESULT: ISSUES DETECTED")
            print("Please fix the issues marked with ✗ above before running the brain.")

def main():
    """Main function."""
    diagnostic = BrainDiagnostic()
    diagnostic.run_diagnostics()

if __name__ == "__main__":
    main()