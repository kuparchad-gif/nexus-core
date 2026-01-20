# C:\Engineers\root\scripts\program_launcher.py
# Enhanced launch script for Engineers runtime with Bridge integration

# C:\Engineers\root\scripts\program_launcher.py
# Enhanced launch script for Engineers runtime with Bridge integration

import os
import subprocess
import json
import time
import webbrowser
import sqlite3
import threading
import requests
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

    def check_existing_models(self):
        """Check if models are already loaded via LM Studio or vLLM"""
        print("🔍 Checking for existing loaded models...")
        
        # Check LM Studio (if running)
        try:
            response = requests.get("http://127.0.0.1:1313/v1/models", timeout=3)
            if response.status_code == 200:
                lm_models = response.json().get("data", [])
                if lm_models:
                    print(f"✅ Found {len(lm_models)} models in LM Studio:")
                    for model in lm_models:
                        print(f"   - {model.get('id', 'Unknown')}")
                    return {"source": "lm_studio", "models": lm_models, "count": len(lm_models)}
        except:
            pass
        
        # Check vLLM instances on common ports
        vllm_ports = [8000, 8001, 8002, 8003, 8004]
        vllm_models = []
        
        for port in vllm_ports:
            try:
                response = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=2)
                if response.status_code == 200:
                    port_models = response.json().get("data", [])
                    for model in port_models:
                        model['vllm_port'] = port
                        vllm_models.append(model)
            except:
                continue
        
        if vllm_models:
            print(f"✅ Found {len(vllm_models)} models in vLLM:")
            for model in vllm_models:
                print(f"   - {model.get('id', 'Unknown')} (Port {model.get('vllm_port')})")
            return {"source": "vllm", "models": vllm_models, "count": len(vllm_models)}
        
        # Check CLI for loaded models
        try:
            result = subprocess.run(["lms", "ps"], capture_output=True, text=True, timeout=5)
            if "Identifier:" in result.stdout:
                cli_models = []
                for line in result.stdout.split('\n'):
                    if line.strip().startswith('Identifier:'):
                        model_id = line.replace('Identifier:', '').strip()
                        cli_models.append({"id": model_id})
                
                if cli_models:
                    print(f"✅ Found {len(cli_models)} models via CLI:")
                    for model in cli_models:
                        print(f"   - {model['id']}")
                    return {"source": "cli", "models": cli_models, "count": len(cli_models)}
        except:
            pass
        
        print("❌ No existing models found")
        return {"source": "none", "models": [], "count": 0}

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

    def step_3_model_loading(self):
        """Check for existing models first, then auto-load if needed"""
        print("🤖 Step 3: Model Loading Check...")
        
        # First, check if models are already loaded
        existing_models = self.check_existing_models()
        
        if existing_models["count"] > 0:
            print(f"✅ Using {existing_models['count']} existing models from {existing_models['source']}")
            self.loaded_models = existing_models["models"]
            
            # Save to public directory for web console
            with open(LOADED_MODELS_JSON, "w", encoding="utf-8") as f:
                json.dump({
                    "source": existing_models["source"],
                    "models": self.loaded_models,
                    "count": existing_models["count"],
                    "auto_loaded": False
                }, f, indent=2)
            
            return True
        
        # No existing models found, try auto-loading
        print("🔄 No existing models found, attempting auto-load...")
        try:
            # Run the model autoloader
            autoload_models()
            
            # Verify loaded models
            models_file = os.path.join(BASE_DIR, "environment_context.json")
            if os.path.exists(models_file):
                with open(models_file, "r", encoding="utf-8") as f:
                    context = json.load(f)
                    self.loaded_models = context.get("status", {}).get("loaded_models", [])
            
            # Save to public directory for web console
            with open(LOADED_MODELS_JSON, "w", encoding="utf-8") as f:
                json.dump({
                    "source": "autoloader",
                    "models": self.loaded_models,
                    "count": len(self.loaded_models),
                    "auto_loaded": True
                }, f, indent=2)
            
            if len(self.loaded_models) > 0:
                print(f"✅ Auto-loaded {len(self.loaded_models)} models")
                return True
            else:
                print("⚠️  Auto-loader completed but no models were loaded")
                return False
                
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
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

    def log_system_status(self):
        """Log final system status to memory database"""
        try:
            if self.memory_db:
                status = {
                    "loaded_models": len(self.loaded_models),
                    "services_running": self.services_running,
                    "bridge_running": self.services_running.get("bridge", False),
                    "main_app_running": self.services_running.get("main_app", False)
                }
                
                self.memory_db.log("INFO", f"System startup complete: {json.dumps(status)}")
                
                # Also save to session for Engineers to reference
                save_session(
                    os.path.join(MEMORY_DIR, "sessions"),
                    {
                        "timestamp": time.time(),
                        "type": "system_startup",
                        "status": status,
                        "models": self.loaded_models
                    },
                    DB_PATH
                )
                
        except Exception as e:
            print(f"⚠️  Failed to log system status: {e}")

    def monitor_services(self):
        """Background thread to monitor running services"""
        def monitor():
            while True:
                time.sleep(30)  # Check every 30 seconds
                
                # Check bridge
                if self.bridge_process and self.bridge_process.poll() is not None:
                    print("⚠️  Bridge process died, attempting restart...")
                    self.step_4_launch_bridge()
                
                # Check main app
                if self.main_app_process and self.main_app_process.poll() is not None:
                    print("⚠️  Main app process died, attempting restart...")
                    self.step_5_launch_main_app()
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

    def run_full_bootstrap(self):
        """Run the complete bootstrap sequence"""
        print("🚀 Starting Engineers Bootstrap Sequence...")
        print("=" * 50)
        
        steps = [
            ("Environment Check", self.step_1_environment_check),
            ("Memory Initialization", self.step_2_memory_initialization),
            ("Model Loading", self.step_3_model_loading),
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
        
        # Log final status
        self.log_system_status()
        
        # Start monitoring
        if success_count >= 4:  # At least core services running
            self.monitor_services()
            print("👁️  Service monitoring started")
        
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
