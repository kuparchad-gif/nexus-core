"""
Setup script for Divine-Cosmic Synthesis System
"""

import sys
import subprocess
import importlib

def check_and_install(package, import_name=None):
    """Check if package is installed, install if not"""
    import_name = import_name or package
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package} already installed")
        return True
    except ImportError:
        print(f"📦 Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            return False

def setup_divine_cosmic():
    """Setup Divine-Cosmic Synthesis System"""
    print("="*60)
    print("✨ SETTING UP DIVINE-COSMIC SYNTHESIS SYSTEM")
    print("="*60)
    
    # Core dependencies
    dependencies = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("networkx", "networkx"),
        ("Flask", "flask"),
        ("torch", "torch")  # Optional but recommended
    ]
    
    print("\n📦 CHECKING AND INSTALLING DEPENDENCIES")
    print("-"*40)
    
    installed_all = True
    for package, import_name in dependencies:
        if not check_and_install(package, import_name):
            installed_all = False
    
    # Check for required files
    print("\n📁 CHECKING REQUIRED FILES")
    print("-"*40)
    
    required_files = [
        "divine_pipeline.py",
        "divine_cosmic_synthesis.py"
    ]
    
    missing_files = []
    for file in required_files:
        try:
            with open(file, 'r'):
                print(f"✅ {file} found")
        except FileNotFoundError:
            print(f"❌ {file} not found")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        print("Please ensure all required files are in the current directory")
    
    # Setup summary
    print("\n" + "="*60)
    print("📊 SETUP SUMMARY")
    print("="*60)
    
    if installed_all and not missing_files:
        print("✅ SETUP COMPLETE")
        print("\n🚀 To start the system:")
        print("   1. Ensure cosmic_breathing_network.py is available")
        print("   2. Run: python divine_cosmic_synthesis.py")
        print("\n🌐 Web interface will be available at:")
        print("   http://localhost:8889")
        print("\n📚 Available endpoints:")
        print("   • POST /api/divine-cosmic/synthesize")
        print("   • GET  /api/divine-cosmic/status")
        print("   • GET  /api/divine-cosmic/history")
        print("   • GET  /api/divine-cosmic/synergy-graph")
        print("   • GET  /api/divine-cosmic/recommendations")
    else:
        print("❌ SETUP INCOMPLETE")
        if missing_files:
            print(f"   Missing files: {missing_files}")
        if not installed_all:
            print("   Some dependencies failed to install")
        
        print("\n📝 Please manually install missing components:")
        print("   pip install numpy scipy scikit-learn networkx Flask torch")
    
    return installed_all and not missing_files

if __name__ == "__main__":
    setup_divine_cosmic()