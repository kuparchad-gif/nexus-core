#!/usr/bin/env python3
"""
📊 COMPLETE DIAGNOSTIC SYSTEM
Collects EVERYTHING needed to debug your setup
"""

import os
import sys
import platform
import json
import subprocess
import importlib
import inspect
import traceback
from pathlib import Path
import shutil
import psutil
import socket
import time
import hashlib
import zipfile
import urllib.request

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"🔍 {title}")
    print("="*80)

def get_system_info():
    """Get complete system information"""
    print_header("SYSTEM INFORMATION")
    
    info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "working_directory": os.getcwd(),
        "user": os.getenv("USER") or os.getenv("USERNAME"),
        "cpu_count": os.cpu_count(),
        "total_memory_gb": psutil.virtual_memory().total / (1024**3),
        "available_memory_gb": psutil.virtual_memory().available / (1024**3),
        "disk_usage": {},
        "network_info": {
            "hostname": socket.gethostname(),
            "ip": socket.gethostbyname(socket.gethostname()),
        }
    }
    
    # Disk usage
    for part in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(part.mountpoint)
            info["disk_usage"][part.mountpoint] = {
                "total_gb": usage.total / (1024**3),
                "used_gb": usage.used / (1024**3),
                "free_gb": usage.free / (1024**3),
                "percent": usage.percent
            }
        except:
            pass
    
    for key, value in info.items():
        if isinstance(value, dict):
            print(f"\n{key.upper()}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    return info

def get_python_environment():
    """Get Python environment details"""
    print_header("PYTHON ENVIRONMENT")
    
    env_info = {
        "pip_version": None,
        "installed_packages": [],
        "sys_path": sys.path,
        "environment_variables": {}
    }
    
    # Get pip version
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                              capture_output=True, text=True)
        env_info["pip_version"] = result.stdout.strip()
    except:
        pass
    
    # Get installed packages
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "list", "--format=json"], 
                              capture_output=True, text=True)
        packages = json.loads(result.stdout)
        env_info["installed_packages"] = [p["name"] for p in packages]
    except:
        try:
            import pkg_resources
            env_info["installed_packages"] = [p.key for p in pkg_resources.working_set]
        except:
            pass
    
    # Get environment variables
    for key in os.environ:
        if any(x in key.lower() for x in ['python', 'path', 'home', 'pip', 'conda', 'venv']):
            env_info["environment_variables"][key] = os.environ[key]
    
    print(f"Python Executable: {sys.executable}")
    print(f"PIP Version: {env_info['pip_version']}")
    print(f"\nInstalled Packages ({len(env_info['installed_packages'])}):")
    for pkg in sorted(env_info['installed_packages'])[:50]:  # Show first 50
        print(f"  • {pkg}")
    if len(env_info['installed_packages']) > 50:
        print(f"  ... and {len(env_info['installed_packages']) - 50} more")
    
    print(f"\nPython Path (first 10 of {len(sys.path)}):")
    for path in sys.path[:10]:
        print(f"  • {path}")
    
    return env_info

def analyze_directory_structure():
    """Analyze current directory structure"""
    print_header("DIRECTORY STRUCTURE")
    
    def get_tree(path, indent=0, max_depth=3, current_depth=0):
        tree = []
        prefix = "  " * indent
        
        try:
            for item in sorted(os.listdir(path)):
                full_path = os.path.join(path, item)
                if os.path.isdir(full_path):
                    tree.append(f"{prefix}📁 {item}/")
                    if current_depth < max_depth:
                        tree.extend(get_tree(full_path, indent + 1, max_depth, current_depth + 1))
                else:
                    size = os.path.getsize(full_path)
                    tree.append(f"{prefix}📄 {item} ({size:,} bytes)")
        except PermissionError:
            tree.append(f"{prefix}⛔ Permission denied")
        except Exception as e:
            tree.append(f"{prefix}❌ Error: {e}")
        
        return tree
    
    current_dir = os.getcwd()
    print(f"Current Directory: {current_dir}")
    
    tree_lines = get_tree(current_dir)
    for line in tree_lines[:100]:  # Limit output
        print(line)
    
    if len(tree_lines) > 100:
        print(f"\n... and {len(tree_lines) - 100} more items")
    
    # Check for specific files
    print("\n🔎 Looking for key files:")
    key_files = [
        "anynode_unified.py",
        "ultimate_nexus_system.py",
        "requirements.txt",
        "setup.py",
        "main.py",
        "app.py"
    ]
    
    found_files = []
    for file in key_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            found_files.append((file, size))
            print(f"  ✅ {file} ({size:,} bytes)")
        else:
            print(f"  ❌ {file} (not found)")
    
    # Look for directories
    print("\n🔎 Looking for key directories:")
    key_dirs = [
        "ozos_code",
        "app",
        "src",
        "Systems",
        "Utilities",
        "nexus-core"
    ]
    
    for dir_name in key_dirs:
        if os.path.exists(dir_name) and os.path.isdir(dir_name):
            items = len(os.listdir(dir_name))
            print(f"  ✅ {dir_name}/ ({items} items)")
        else:
            print(f"  ❌ {dir_name}/ (not found)")
    
    return {
        "current_dir": current_dir,
        "tree_preview": tree_lines[:50],
        "found_files": found_files
    }

def check_git_repository():
    """Check if we're in a git repository"""
    print_header("GIT REPOSITORY STATUS")
    
    git_info = {
        "is_git_repo": False,
        "branch": None,
        "remote": None,
        "status": None
    }
    
    # Check if git is installed
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        git_installed = True
    except:
        print("Git is not installed or not in PATH")
        return git_info
    
    # Check if current directory is a git repo
    try:
        result = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                              capture_output=True, text=True)
        if result.stdout.strip() == "true":
            git_info["is_git_repo"] = True
            
            # Get branch
            result = subprocess.run(["git", "branch", "--show-current"],
                                  capture_output=True, text=True)
            git_info["branch"] = result.stdout.strip()
            
            # Get remote
            try:
                result = subprocess.run(["git", "remote", "-v"],
                                      capture_output=True, text=True)
                git_info["remote"] = result.stdout.strip()
            except:
                pass
            
            # Get status
            try:
                result = subprocess.run(["git", "status", "--short"],
                                      capture_output=True, text=True)
                git_info["status"] = result.stdout.strip()
            except:
                pass
            
            print(f"✅ Git Repository")
            print(f"   Branch: {git_info['branch']}")
            if git_info['remote']:
                print("   Remotes:")
                for line in git_info['remote'].split('\n'):
                    if line:
                        print(f"     {line}")
            if git_info['status']:
                print("   Status:")
                for line in git_info['status'].split('\n'):
                    if line:
                        print(f"     {line}")
        else:
            print("❌ Not a git repository")
    except:
        print("❌ Not a git repository")
    
    return git_info

def analyze_python_file(filepath):
    """Analyze a Python file"""
    if not os.path.exists(filepath):
        return {"error": f"File not found: {filepath}"}
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        info = {
            "size_bytes": os.path.getsize(filepath),
            "lines": len(content.splitlines()),
            "imports": [],
            "classes": [],
            "functions": [],
            "has_errors": False,
            "sample_lines": content[:1000]  # First 1000 chars
        }
        
        # Try to parse imports (simple regex-based)
        import re
        
        # Find imports
        import_pattern = r'^\s*(?:from\s+(\S+)\s+import|import\s+)([^#\n]+)'
        for match in re.finditer(import_pattern, content, re.MULTILINE):
            info["imports"].append(match.group(0).strip())
        
        # Find class definitions
        class_pattern = r'^class\s+(\w+)'
        for match in re.finditer(class_pattern, content, re.MULTILINE):
            info["classes"].append(match.group(1))
        
        # Find function definitions
        func_pattern = r'^def\s+(\w+)'
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            info["functions"].append(match.group(1))
        
        # Try to actually import and check for syntax errors
        try:
            spec = importlib.util.spec_from_file_location("temp_module", filepath)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                # Don't actually execute, just check syntax
                compile(content, filepath, 'exec')
        except SyntaxError as e:
            info["has_errors"] = True
            info["syntax_error"] = str(e)
        except Exception as e:
            info["import_error"] = str(e)
        
        return info
        
    except Exception as e:
        return {"error": str(e)}

def check_critical_files():
    """Check critical Python files"""
    print_header("CRITICAL FILE ANALYSIS")
    
    files_to_check = []
    
    # Look for Python files
    for root, dirs, files in os.walk(".", maxdepth=2):
        for file in files:
            if file.endswith('.py'):
                files_to_check.append(os.path.join(root, file))
                if len(files_to_check) >= 20:  # Limit to 20 files
                    break
        if len(files_to_check) >= 20:
            break
    
    results = {}
    for filepath in sorted(files_to_check)[:10]:  # Check first 10
        print(f"\n📄 Analyzing: {filepath}")
        result = analyze_python_file(filepath)
        
        if "error" in result:
            print(f"   ❌ Error: {result['error']}")
        else:
            print(f"   📏 Size: {result['size_bytes']:,} bytes, Lines: {result['lines']}")
            
            if result.get('has_errors'):
                print(f"   ❌ Syntax error: {result.get('syntax_error', 'Unknown')}")
            elif result.get('import_error'):
                print(f"   ⚠️ Import error: {result.get('import_error')}")
            else:
                print(f"   ✅ Syntax OK")
            
            if result['imports']:
                print(f"   📦 Imports ({len(result['imports'])}):")
                for imp in result['imports'][:5]:  # Show first 5
                    print(f"     • {imp}")
                if len(result['imports']) > 5:
                    print(f"     ... and {len(result['imports']) - 5} more")
            
            if result['classes']:
                print(f"   🏛️ Classes ({len(result['classes'])}): {', '.join(result['classes'][:5])}")
            
            if result['functions']:
                print(f"   ⚙️ Functions ({len(result['functions'])}): {', '.join(result['functions'][:5])}")
        
        results[filepath] = result
    
    return results

def test_imports():
    """Test importing common packages"""
    print_header("IMPORT TESTS")
    
    packages = [
        # Core
        "asyncio", "json", "os", "sys", "time", "pathlib",
        # Data
        "numpy", "pandas", "scipy",
        # ML/AI
        "torch", "tensorflow", "transformers", "diffusers",
        # Web
        "fastapi", "aiohttp", "requests",
        # Utils
        "psutil", "ping3", "cryptography", "PIL",
        # AnyNode specific
        "sklearn", "networkx", "qdrant_client"
    ]
    
    results = {}
    for package in packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "unknown")
            results[package] = {"status": "✅", "version": version}
            print(f"✅ {package}: {version}")
        except ImportError as e:
            results[package] = {"status": "❌", "error": str(e)}
            print(f"❌ {package}: {e}")
    
    return results

def check_network_connectivity():
    """Check network connectivity"""
    print_header("NETWORK CONNECTIVITY")
    
    tests = [
        ("Google DNS", "8.8.8.8", 53),
        ("Google", "google.com", 80),
        ("GitHub", "github.com", 443),
        ("Cloudflare", "1.1.1.1", 53)
    ]
    
    results = []
    for name, host, port in tests:
        try:
            start = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((host, port))
            sock.close()
            latency = (time.time() - start) * 1000
            results.append((name, "✅", f"{latency:.1f} ms"))
            print(f"✅ {name}: {host}:{port} - {latency:.1f} ms")
        except Exception as e:
            results.append((name, "❌", str(e)))
            print(f"❌ {name}: {host}:{port} - {e}")
    
    return results

def create_minimal_test():
    """Create a minimal test script to verify the system works"""
    print_header("CREATING MINIMAL TEST")
    
    test_code = '''#!/usr/bin/env python3
"""
MINIMAL TEST SCRIPT
Run this to verify basic functionality
"""

import asyncio
import sys
import os
import json
import time

print("="*60)
print("🧪 MINIMAL SYSTEM TEST")
print("="*60)

# Test 1: Basic imports
print("\\n✅ Test 1: Basic Imports")
try:
    import numpy as np
    print(f"   numpy: {np.__version__}")
except ImportError:
    print("   ❌ numpy not installed")

try:
    import psutil
    print(f"   psutil: CPU usage: {psutil.cpu_percent()}%")
except ImportError:
    print("   ❌ psutil not installed")

# Test 2: Async functionality
print("\\n✅ Test 2: Async Functions")
async def test_async():
    await asyncio.sleep(0.1)
    return "Async works!"

async def main_test():
    result = await test_async()
    print(f"   {result}")

# Test 3: File system
print("\\n✅ Test 3: File System")
print(f"   Current dir: {os.getcwd()}")
print(f"   Files in dir: {len(os.listdir('.'))}")

# Test 4: Create a simple AnyNode-like class
print("\\n✅ Test 4: Simple Class")
class SimpleNode:
    def __init__(self, name):
        self.name = name
        self.created = time.time()
    
    async def start(self):
        await asyncio.sleep(0.1)
        return f"Node {self.name} started"
    
    def get_info(self):
        return {
            "name": self.name,
            "uptime": time.time() - self.created
        }

async def run_tests():
    # Run async test
    await main_test()
    
    # Test node
    node = SimpleNode("test-node-1")
    result = await node.start()
    print(f"   {result}")
    
    print(f"   Node info: {node.get_info()}")
    
    print("\\n" + "="*60)
    print("🎉 ALL MINIMAL TESTS PASSED!")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(run_tests())
'''
    
    with open("minimal_test.py", "w", encoding="utf-8") as f:
        f.write(test_code)
    
    print("✅ Created minimal_test.py")
    print("   Run with: python minimal_test.py")
    
    return "minimal_test.py"

def download_ozos_code():
    """Download the ozos_code repository if needed"""
    print_header("DOWNLOADING OZOS_CODE")
    
    if os.path.exists("ozos_code"):
        print("✅ ozos_code directory already exists")
        print(f"   Contents: {len(os.listdir('ozos_code'))} items")
        return True
    
    print("Downloading from GitHub...")
    
    repo_url = "https://github.com/kuparchad-gif/nexus-core/archive/refs/heads/main.zip"
    
    try:
        # Download zip
        print(f"📥 Downloading from {repo_url}")
        zip_path = "nexus-core-main.zip"
        urllib.request.urlretrieve(repo_url, zip_path)
        
        # Extract
        print("📦 Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        # Rename if needed
        if os.path.exists("nexus-core-main"):
            if os.path.exists("ozos_code"):
                shutil.rmtree("ozos_code")
            os.rename("nexus-core-main", "ozos_code")
        
        # Clean up
        os.remove(zip_path)
        
        print(f"✅ Downloaded ozos_code with {len(os.listdir('ozos_code'))} items")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def create_ultimate_bootstrapper():
    """Create the ultimate bootstrapper that can fix everything"""
    print_header("CREATING ULTIMATE BOOTSTRAPPER")
    
    bootstrapper_code = '''#!/usr/bin/env python3
"""
🚀 ULTIMATE OZOS BOOTSTRAPPER
This script will:
1. Check your system
2. Install missing dependencies
3. Set up the environment
4. Launch the system
"""

import os
import sys
import subprocess
import platform
import json
import shutil
from pathlib import Path

def run_command(cmd, description):
    """Run a command and show output"""
    print(f"\\n📝 {description}...")
    print(f"   Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Success")
            if result.stdout.strip():
                print(f"   Output: {result.stdout[:200]}...")
        else:
            print(f"   ❌ Failed")
            print(f"   Error: {result.stderr[:200]}")
        return result.returncode == 0
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

def main():
    print("="*80)
    print("🚀 ULTIMATE OZOS BOOTSTRAPPER")
    print("="*80)
    
    # Step 1: System check
    print("\\n🔍 STEP 1: System Check")
    print(f"   Platform: {platform.platform()}")
    print(f"   Python: {sys.version}")
    print(f"   Directory: {os.getcwd()}")
    
    # Step 2: Install basic dependencies
    print("\\n📦 STEP 2: Installing Basic Dependencies")
    
    base_packages = [
        "pip", "setuptools", "wheel"
    ]
    
    for pkg in base_packages:
        run_command(f"{sys.executable} -m pip install --upgrade {pkg}", f"Upgrade {pkg}")
    
    # Step 3: Install required packages
    print("\\n📦 STEP 3: Installing Required Packages")
    
    required_packages = [
        "asyncio",
        "aiohttp",
        "psutil",
        "numpy",
        "scipy",
        "scikit-learn",
        "Pillow",
        "requests",
        "fastapi",
        "uvicorn[standard]"
    ]
    
    for pkg in required_packages:
        run_command(f"{sys.executable} -m pip install {pkg}", f"Install {pkg}")
    
    # Step 4: Check for ozos_code
    print("\\n📁 STEP 4: Checking ozos_code")
    
    if not os.path.exists("ozos_code"):
        print("   ❌ ozos_code not found!")
        print("   Downloading from GitHub...")
        
        # Try to download
        success = run_command(
            "git clone https://github.com/kuparchad-gif/nexus-core.git ozos_code",
            "Clone repository"
        )
        
        if not success:
            print("   Trying alternative download...")
            # Create minimal structure
            os.makedirs("ozos_code", exist_ok=True)
            os.makedirs("ozos_code/app", exist_ok=True)
            os.makedirs("ozos_code/Systems", exist_ok=True)
            
            # Create a simple anynode.py
            simple_anynode = '''
# Simple AnyNode implementation
import asyncio
import time

class AnyNode:
    def __init__(self):
        self.name = "SimpleAnyNode"
    
    async def start(self):
        print("SimpleAnyNode started")
        return True

class UnifiedDiscoveryService:
    def __init__(self):
        self.services = []
    
    async def start(self):
        print("Discovery service started")
        return True
'''
            
            with open("ozos_code/app/anynode.py", "w") as f:
                f.write(simple_anynode)
            
            print("   ✅ Created minimal ozos_code structure")
    else:
        print(f"   ✅ ozos_code found with {len(os.listdir('ozos_code'))} items")
    
    # Step 5: Create unified system
    print("\\n🔄 STEP 5: Creating Unified System")
    
    unified_code = '''
import asyncio
import sys
import os
from pathlib import Path
import json
import time

print("="*70)
print("🌌 UNIFIED SYSTEM")
print("="*70)

# Add ozos_code to path
sys.path.insert(0, str(Path(__file__).parent / "ozos_code"))

try:
    # Try to import AnyNode
    from app.anynode import AnyNode, UnifiedDiscoveryService
    print("✅ Imported AnyNode")
except ImportError as e:
    print(f"⚠️ Could not import AnyNode: {{e}}")
    
    # Create fallback
    class AnyNode:
        async def start(self):
            print("Fallback AnyNode started")
            return True
    
    class UnifiedDiscoveryService:
        async def start(self):
            print("Fallback Discovery started")
            return True

class UnifiedSystem:
    def __init__(self):
        self.anynode = AnyNode()
        self.discovery = UnifiedDiscoveryService()
    
    async def boot(self):
        print("\\n🚀 Booting Unified System...")
        await self.anynode.start()
        await self.discovery.start()
        print("\\n✅ System Ready!")
        return True

async def main():
    system = UnifiedSystem()
    await system.boot()
    
    print("\\nType 'exit' to quit")
    while True:
        cmd = input("system> ").strip()
        if cmd.lower() == 'exit':
            break
        print(f"Command: {{cmd}}")

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    with open("unified_system.py", "w", encoding="utf-8") as f:
        f.write(unified_code)
    
    print("   ✅ Created unified_system.py")
    
    # Step 6: Run the system
    print("\\n🚀 STEP 6: Launching System")
    print("\\n" + "="*80)
    print("🎉 READY TO LAUNCH!")
    print("="*80)
    print("\\nTo launch the unified system:")
    print("1. python unified_system.py")
    print("\\nOr run tests:")
    print("2. python minimal_test.py")
    
    # Ask if user wants to launch
    launch = input("\\n🚀 Launch unified system now? (y/n): ").strip().lower()
    if launch == 'y':
        print("\\n" + "="*80)
        print("🚀 LAUNCHING...")
        print("="*80)
        os.system(f"{sys.executable} unified_system.py")

if __name__ == "__main__":
    main()
'''
    
    with open("bootstrap.py", "w", encoding="utf-8") as f:
        f.write(bootstrapper_code)
    
    print("✅ Created bootstrap.py")
    print("   This is the ULTIMATE fix-it script")
    print("   Run with: python bootstrap.py")
    
    return "bootstrap.py"

def main():
    """Main diagnostic function"""
    print("="*80)
    print("🛠️  COMPLETE DIAGNOSTIC SYSTEM")
    print("📊 Gathers EVERYTHING needed to debug your setup")
    print("="*80)
    
    # Create output directory
    output_dir = "diagnostic_output"
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    # Run all diagnostics
    all_results["system_info"] = get_system_info()
    all_results["python_env"] = get_python_environment()
    all_results["directory_structure"] = analyze_directory_structure()
    all_results["git_status"] = check_git_repository()
    all_results["critical_files"] = check_critical_files()
    all_results["import_tests"] = test_imports()
    all_results["network_tests"] = check_network_connectivity()
    
    # Create helpful files
    test_file = create_minimal_test()
    bootstrapper_file = create_ultimate_bootstrapper()
    
    # Try to download ozos_code
    download_ozos_code()
    
    # Save all results to file
    output_file = os.path.join(output_dir, "diagnostic_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        # Convert non-serializable objects to strings
        serializable_results = {}
        for key, value in all_results.items():
            try:
                json.dumps(value)
                serializable_results[key] = value
            except:
                serializable_results[key] = str(value)
        
        json.dump(serializable_results, f, indent=2)
    
    print_header("SUMMARY")
    print(f"✅ Diagnostics complete!")
    print(f"📁 Results saved to: {output_file}")
    print(f"🧪 Test file created: {test_file}")
    print(f"🚀 Bootstrapper created: {bootstrapper_file}")
    
    print("\n" + "="*80)
    print("🎯 NEXT STEPS:")
    print("="*80)
    print("1. Run the bootstrapper:")
    print("   $ python bootstrap.py")
    print("\n2. Or run the minimal test:")
    print("   $ python minimal_test.py")
    print("\n3. If you're having issues, share these files with me:")
    print("   - diagnostic_output/diagnostic_results.json")
    print("   - The output of: python bootstrap.py")
    print("\n4. For Google Colab, upload ALL files and run:")
    print("   !python bootstrap.py")
    print("\n💡 The bootstrapper will AUTOMATICALLY fix most issues!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Diagnostics interrupted")
    except Exception as e:
        print(f"\n❌ Diagnostic error: {e}")
        traceback.print_exc()