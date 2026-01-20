# C:\Engineers\root\scripts\bridge_test.py
# Test script for supporting applications used in Bridge integration - Models do not load
# Must be run from scripts location

cd //d "C:\\Engineers\\root\\scripts"


import os
import subprocess
import json
import time
import webbrowser
import sqlite3
import threading
from pathlib import Path

# Import all the memory and boot systems
from bootstrap_environment import collect_environment, save_context
from memory_initializer import main as init_memory
from session_manager import load_sessions, save_session
from memory_db import MemoryDB
from model_autoloader import main as autoload_models

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")
BRIDGE_DIR = os.path.join(ROOT_DIR, "bridge")
PUBLIC_DIR = os.path.join(ROOT_DIR, "public")
MEMORY_DIR = os.path.join(ROOT_DIR, "memory")
DB_PATH = os.path.join(MEMORY_DIR, "engineers.db")
LOADED_MODELS_JSON = os.path.join(PUBLIC_DIR, "loaded_models.json")
LOAD_LIST = os.path.join(BASE_DIR, 'models_to_load.txt')
MANIFEST_PATH = r"C:\\Engineers\\root\\models\\model_manifest.json"
LOG_STREAM_FILE = r"C:\\Engineers\\root\\logs\\lms_prompt_debug.log"
ENV_CONTEXT_FILE = os.path.join(BASE_DIR, 'environment_context.json')
MAX_LOG_SIZE_MB = 20
LOAD_WAIT_SECS = 6
LMS_EXECUTABLE = "lms"
LMS_CWD = r"C:\\Engineers\\root\\Local\\Programs\\LM Studio\\resources\\app\\.webpack"
ANSI_SUPPORTED = sys.stdout.isatty() and os.name != "nt"

# Debug prints - add these temporarily
print(f"🔍 BASE_DIR: {os.path.abspath(BASE_DIR)}")
print(f"🔍 ROOT_DIR: {os.path.abspath(ROOT_DIR)}")
print(f"🔍 BRIDGE_DIR: {os.path.abspath(BRIDGE_DIR)}")
print(f"🔍 Bridge exists? {os.path.exists(BRIDGE_DIR)}")
print(f"🔍 Bridge contents: {os.listdir(BRIDGE_DIR) if os.path.exists(BRIDGE_DIR) else 'NOT FOUND'}")

class EngineerBootstrap:
    def __init__(self):
        self.loaded_models = []
        self.services_running = {}
        self.memory_db = None
        self.bridge_process = None
        self.main_app_process = None
        
        # Ensure all directories exist
        os.makedirs(MEMORY_DIR, exist_ok=True)
        os.makedirs(PUBLIC_DIR, exist_ok=True)

    def step_1_environment_check(self):
        """Bootstrap environment detection and logging"""
        print("🔍 Step 1: Environment Bootstrap...")
        try:
            env_context = collect_environment()
            save_context(env_context)
            
            if env_context.get("errors"):
                print("⚠️  Environment issues detected:")
                for error in env_context["errors"]:
                    print(f"   - {error}")
                return False
            
            print("✅ Environment check passed")
            return True
        except Exception as e:
            print(f"❌ Environment check failed: {e}")
            return False

    def step_2_memory_initialization(self):
        """Initialize all memory systems"""
        print("🧠 Step 2: Memory System Initialization...")
        try:
            # Initialize SQLite memory database
            self.memory_db = MemoryDB(DB_PATH)
            self.memory_db.log("INFO", "Engineers system startup initiated")
            
            # Run memory initializer
            init_memory()
            
            # Load any existing sessions
            sessions_dir = os.path.join(MEMORY_DIR, "sessions")
            os.makedirs(sessions_dir, exist_ok=True)
            existing_sessions = load_sessions(sessions_dir, DB_PATH)
            
            print(f"✅ Memory systems initialized. Found {len(existing_sessions)} previous sessions")
            return True
        except Exception as e:
            print(f"❌ Memory initialization failed: {e}")
            return False

    def step_4_launch_bridge(self):
        """Launch the enhanced bridge engine"""
        print("🌉 Step 4: Launching Bridge Engine...")
        try:
            bridge_script = os.path.join(BRIDGE_DIR, "bridge_engine.py")
            
            if not os.path.exists(bridge_script):
                print(f"⚠️  Bridge script not found at {bridge_script}")
                return False
            
            # Launch bridge in separate process
            self.bridge_process = subprocess.Popen([
                "python", bridge_script
            ], cwd=BRIDGE_DIR)
            
            # Wait a moment for bridge to start
            time.sleep(3)
            
            # Verify bridge is running
            if self.bridge_process.poll() is None:
                print("✅ Bridge Engine started successfully")
                self.services_running["bridge"] = True
                return True
            else:
                print("❌ Bridge Engine failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Bridge launch failed: {e}")
            return False

    def step_5_launch_main_app(self):
        """Launch the main Flask application"""
        print("🚀 Step 5: Launching Main Application...")
        try:
            app_script = os.path.join(PUBLIC_DIR, "app.py")
            
            if not os.path.exists(app_script):
                print(f"⚠️  Main app not found at {app_script}")
                return False
            
            # Launch main app
            self.main_app_process = subprocess.Popen([
                "python", app_script
            ], cwd=PUBLIC_DIR)
            
            # Wait for app to start
            time.sleep(2)
            
            if self.main_app_process.poll() is None:
                print("✅ Main application started successfully")
                self.services_running["main_app"] = True
                return True
            else:
                print("❌ Main application failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Main app launch failed: {e}")
            return False

    def step_6_launch_console(self):
        """Open the web console"""
        print("🌐 Step 6: Launching Web Console...")
        try:
            console_file = os.path.join(PUBLIC_DIR, "console.html")
            
            if os.path.exists(console_file):
                # Give the app time to fully start
                time.sleep(2)
                webbrowser.open("http://localhost:5000")
                print("✅ Web console opened in browser")
                return True
            else:
                print(f"⚠️  Console file not found at {console_file}")
                return False
                
        except Exception as e:
            print(f"❌ Console launch failed: {e}")
            return False
            
    def run_full_bootstrap(self):
        """Run the complete bootstrap sequence"""
        print("🚀 Starting Engineers Bootstrap Sequence...")
        print("=" * 50)
        
        steps = [
            # ("Environment Check", self.step_1_environment_check),
            # ("Memory Initialization", self.step_2_memory_initialization),
            # ("Model Loading", self.step_3_model_loading),
            ("Bridge Launch", self.step_4_launch_bridge),
            ("Main App Launch", self.step_5_launch_main_app),
            ("Web Console", self.step_6_launch_console)
        ]
        
        success_count = 0
        for step_name, step_func in steps:
            if step_func():
                success_count += 1
            else:
                print(f"⚠️  {step_name} had issues but continuing...")
            print("-" * 30)
        
        print(f"🎯 Bootstrap Complete: {success_count}/{len(steps)} steps successful")
        
        print("\n🌟 Engineers System Ready!")
        print("📊 Dashboard: http://localhost:5000")
        print("🌉 Bridge API: http://localhost:5001")
        print("\nPress Ctrl+C to shutdown...")
        
        # Keep the launcher running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self):
        """Graceful shutdown of all services"""
        print("\n🛑 Shutting down Engineers...")
        
        if self.memory_db:
            self.memory_db.log("INFO", "System shutdown initiated")
        
        if self.bridge_process:
            self.bridge_process.terminate()
            print("✅ Bridge Engine stopped")
        
        if self.main_app_process:
            self.main_app_process.terminate()
            print("✅ Main Application stopped")
        
        print("👋 Engineers shutdown complete")

def main():
    bootstrap = EngineerBootstrap()
    bootstrap.run_full_bootstrap()

if __name__ == "__main__":
    main()
