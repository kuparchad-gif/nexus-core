#!/usr/bin/env python3
"""
🚀 NEXUS MCP CONSOLE - Everything in One Console
MCP + Ray + LangChain + Local AI + Tools - Fixed Import Order
"""
import os
import sys
import json
import asyncio
import subprocess
import time
import uuid
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import threading
import signal

# ==================== AUTO-INSTALL ====================
def install_dependencies():
    """Install required packages"""
    print("🔧 Installing dependencies...")
    
    requirements = [
        "ray[default]>=2.8.0",
        "langchain>=0.0.340",
        "langchain-community>=0.0.10",
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "accelerate>=0.24.0",
        "sentencepiece>=0.1.99",
        "huggingface-hub>=0.20.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "websockets>=12.0",
        "aiohttp>=3.9.0",
        "rich>=13.0.0",
        "typer>=0.9.0",
        "watchdog>=3.0.0",
        "chromadb>=0.4.18",
        "pypdf>=3.17.0",
        "tiktoken>=0.5.0",
        "python-dotenv>=1.0.0",
        "psutil>=5.9.0",
        "pygments>=2.16.0",
        "requests>=2.31.0"
    ]
    
    import importlib.util
    import subprocess
    
    for package in requirements:
        pkg_name = package.split('>=')[0].split('[')[0]
        try:
            importlib.util.find_spec(pkg_name.replace("-", "_"))
            # Already installed
        except:
            print(f"📦 Installing {pkg_name}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except:
                pass
    
    print("✅ Dependencies installed!")

# Try to install
try:
    install_dependencies()
except:
    pass

# ==================== IMPORTS AFTER INSTALL ====================
# Import torch FIRST before using it in Config
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available, GPU features disabled")

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.layout import Layout
    from rich.live import Live
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.prompt import Prompt, Confirm
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich.text import Text
    from rich.box import ROUNDED
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Other imports
import requests
import aiohttp
from typing import Dict, List, Any, Optional

# ==================== FIXED CONFIGURATION ====================
class Config:
    """Global configuration - FIXED import order"""
    
    def __init__(self):
        self.name: str = "🔥 Nexus MCP Console"
        self.version: str = "1.0.0"
        
        # Paths
        self.base_dir: Path = Path.home() / ".nexus_mcp"
        self.models_dir: Path = self.base_dir / "models"
        self.workspace_dir: Path = self.base_dir / "workspace"
        self.db_path: Path = self.base_dir / "nexus.db"
        self.logs_dir: Path = self.base_dir / "logs"
        
        # Ray - Check GPU availability safely
        self.ray_cpus: int = os.cpu_count() or 4
        self.ray_gpus: int = self._get_gpu_count()
        
        # MCP
        self.mcp_host: str = "127.0.0.1"
        self.mcp_port: int = 3000
        
        # AI Models
        self.default_model: str = "TinyLlama-1.1B"
        
        # Create directories
        self._create_dirs()
    
    def _get_gpu_count(self) -> int:
        """Safely check for GPU"""
        if TORCH_AVAILABLE:
            try:
                return 1 if torch.cuda.is_available() else 0
            except:
                return 0
        return 0
    
    def _create_dirs(self):
        """Create necessary directories"""
        for path in [self.base_dir, self.models_dir, self.workspace_dir, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)

# ==================== RAY ORCHESTRATOR ====================
class RayOrchestrator:
    """Ray-based parallel task execution"""
    
    def __init__(self, config: Config):
        self.config = config
        self.initialized = False
        
    def init_ray(self):
        """Initialize Ray cluster"""
        if not RAY_AVAILABLE:
            print("❌ Ray not installed. Please install: pip install ray[default]")
            return None
        
        if ray.is_initialized():
            ray.shutdown()
        
        try:
            ray.init(
                num_cpus=self.config.ray_cpus,
                num_gpus=self.config.ray_gpus,
                ignore_reinit_error=True,
                logging_level=30  # WARNING level
            )
            
            self.initialized = True
            print(f"✅ Ray initialized with {self.config.ray_cpus} CPUs, {self.config.ray_gpus} GPUs")
            
            return self.get_cluster_info()
        except Exception as e:
            print(f"❌ Failed to initialize Ray: {e}")
            return None
    
    def get_cluster_info(self):
        """Get Ray cluster information"""
        if not self.initialized:
            return {"error": "Ray not initialized"}
        
        try:
            resources = ray.available_resources()
            return {
                "resources": dict(resources),
                "cpu_count": resources.get("CPU", 0),
                "gpu_count": resources.get("GPU", 0),
                "memory_gb": resources.get("memory", 0) / 1e9 if "memory" in resources else 0
            }
        except:
            return {"error": "Could not get cluster info"}
    
    @ray.remote
    class FileWorker:
        """Parallel file operations"""
        def read_file(self, path: str) -> Dict[str, Any]:
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                return {"success": True, "content": content, "size": len(content)}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        def write_file(self, path: str, content: str) -> Dict[str, Any]:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return {"success": True, "size": len(content)}
            except Exception as e:
                return {"success": False, "error": str(e)}

# ==================== LOCAL AI MANAGER ====================
class LocalAIManager:
    """Manage local AI models"""
    
    MODELS = {
        "tinyllama": {
            "name": "TinyLlama-1.1B",
            "huggingface": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "size": "2.2GB",
            "description": "Small but capable model"
        }
    }
    
    def __init__(self, config: Config):
        self.config = config
        self.loaded_models = {}
        self.current_model = None
        
    def download_model(self, model_key: str) -> Dict[str, Any]:
        """Download model from HuggingFace"""
        if model_key not in self.MODELS:
            return {"success": False, "error": f"Unknown model: {model_key}"}
        
        model_info = self.MODELS[model_key]
        model_path = self.config.models_dir / model_key
        
        print(f"⬇️  Downloading {model_info['name']}...")
        
        try:
            # Use huggingface_hub if available
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id=model_info['huggingface'],
                    local_dir=str(model_path),
                    local_dir_use_symlinks=False
                )
            except ImportError:
                # Fallback to git
                import subprocess
                subprocess.run([
                    "git", "clone",
                    f"https://huggingface.co/{model_info['huggingface']}",
                    str(model_path)
                ], capture_output=True)
            
            return {"success": True, "model": model_key, "path": str(model_path)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def list_models(self) -> Dict[str, Any]:
        """List available models"""
        return {"available": self.MODELS}

# ==================== SIMPLE TOOLS MANAGER ====================
class SimpleTools:
    """Simple tool implementations without complex dependencies"""
    
    def __init__(self):
        self.tools = {
            "read_file": self.read_file,
            "write_file": self.write_file,
            "execute_python": self.execute_python,
            "system_info": self.system_info
        }
    
    async def read_file(self, path: str) -> Dict[str, Any]:
        """Read file"""
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return {"success": True, "content": content, "size": len(content)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Write file"""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return {"success": True, "size": len(content)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def execute_python(self, code: str) -> Dict[str, Any]:
        """Execute Python code"""
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_path = f.name
            
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            os.unlink(temp_path)
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def system_info(self) -> Dict[str, Any]:
        """Get system information"""
        import platform
        import psutil
        
        return {
            "system": platform.system(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / 1e9 if hasattr(psutil, 'virtual_memory') else 0,
            "torch_available": TORCH_AVAILABLE,
            "ray_available": RAY_AVAILABLE
        }

# ==================== INTERACTIVE CONSOLE ====================
class InteractiveConsole:
    """Simple interactive console"""
    
    def __init__(self):
        self.config = Config()
        self.tools = SimpleTools()
        self.running = True
        
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None
        
        self.print_banner()
    
    def print_banner(self):
        """Print banner"""
        banner = """
╔══════════════════════════════════════════════════════════╗
║                 🔥 NEXUS MCP CONSOLE                    ║
║              Simplified All-in-One Edition               ║
╚══════════════════════════════════════════════════════════╝
"""
        print(banner)
        
        print("📦 Available features:")
        if TORCH_AVAILABLE:
            print("  • 🤖 PyTorch (AI models)")
        if RAY_AVAILABLE:
            print("  • ⚡ Ray (parallel processing)")
        print("  • 📁 File operations")
        print("  • 💻 Code execution")
        print("  • 💬 Interactive console")
        
        print("\n💡 Type 'help' for commands, 'exit' to quit")
        print("=" * 60)
    
    def print_help(self):
        """Print help"""
        help_text = """
Commands:
  help                    - Show this help
  exit                    - Exit console
  
  read <path>            - Read file
  write <path> <text>    - Write file
  run <python_code>      - Execute Python code
  system                 - System information
  
  models                 - List AI models
  download <model>       - Download AI model
  
  ray_init               - Initialize Ray cluster
  ray_status             - Ray cluster status
"""
        print(help_text)
    
    async def handle_command(self, command: str):
        """Handle command"""
        parts = command.strip().split()
        if not parts:
            return ""
        
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if cmd == "help":
            self.print_help()
            return ""
        
        elif cmd == "exit":
            self.running = False
            return "👋 Goodbye!"
        
        elif cmd == "read" and args:
            path = args[0]
            result = await self.tools.read_file(path)
            if result.get("success"):
                content = result.get("content", "")
                return f"📖 File content ({len(content)} chars):\n{content[:500]}..."
            else:
                return f"❌ Error: {result.get('error')}"
        
        elif cmd == "write" and len(args) >= 2:
            path = args[0]
            content = " ".join(args[1:])
            result = await self.tools.write_file(path, content)
            if result.get("success"):
                return f"✅ File written ({result.get('size', 0)} chars)"
            else:
                return f"❌ Error: {result.get('error')}"
        
        elif cmd == "run" and args:
            code = " ".join(args)
            result = await self.tools.execute_python(code)
            if result.get("success"):
                output = result.get("stdout", "")
                if output:
                    return f"✅ Execution output:\n{output}"
                else:
                    return "✅ Code executed (no output)"
            else:
                return f"❌ Error: {result.get('stderr', 'Unknown error')}"
        
        elif cmd == "system":
            result = await self.tools.system_info()
            info = []
            for key, value in result.items():
                info.append(f"{key}: {value}")
            return "\n".join(info)
        
        elif cmd == "models":
            # Create a simple AI manager
            ai_mgr = LocalAIManager(self.config)
            models = ai_mgr.list_models()
            output = ["🤖 Available AI Models:"]
            for key, info in models.get("available", {}).items():
                output.append(f"  • {key}: {info.get('name')} ({info.get('size')})")
            return "\n".join(output)
        
        elif cmd == "download" and args:
            model = args[0]
            ai_mgr = LocalAIManager(self.config)
            result = ai_mgr.download_model(model)
            if result.get("success"):
                return f"✅ Model downloaded: {result.get('model')}"
            else:
                return f"❌ Error: {result.get('error')}"
        
        elif cmd == "ray_init":
            if not RAY_AVAILABLE:
                return "❌ Ray not installed. Install with: pip install ray[default]"
            
            ray_orch = RayOrchestrator(self.config)
            result = ray_orch.init_ray()
            if result:
                return f"✅ Ray initialized: {json.dumps(result, indent=2)}"
            else:
                return "❌ Failed to initialize Ray"
        
        elif cmd == "ray_status":
            if not RAY_AVAILABLE:
                return "Ray not available"
            
            ray_orch = RayOrchestrator(self.config)
            if ray_orch.initialized:
                result = ray_orch.get_cluster_info()
                return f"Ray status: {json.dumps(result, indent=2)}"
            else:
                return "Ray not initialized. Use 'ray_init' first"
        
        else:
            return f"❌ Unknown command: {cmd}\n💡 Type 'help' for available commands"
    
    async def run(self):
        """Run console"""
        print("🚀 Starting Nexus MCP Console...")
        
        # Quick check
        if not TORCH_AVAILABLE:
            print("⚠️  PyTorch not installed. AI features limited.")
            print("💡 Install with: pip install torch")
        
        if not RAY_AVAILABLE:
            print("⚠️  Ray not installed. Parallel features limited.")
            print("💡 Install with: pip install ray[default]")
        
        print("\n✅ Console ready! Type commands below:")
        print("-" * 60)
        
        # Main loop
        while self.running:
            try:
                # Get input
                if self.console:
                    user_input = Prompt.ask("\n[bold cyan]nexus[/bold cyan]")
                else:
                    user_input = input("\nnexus> ").strip()
                
                if not user_input:
                    continue
                
                # Handle command
                result = await self.handle_command(user_input)
                
                # Print result
                if result:
                    if self.console:
                        self.console.print(Panel(result, border_style="green"))
                    else:
                        print(f"\n📋 {result}")
                
            except KeyboardInterrupt:
                print("\n\n💡 Type 'exit' to quit")
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()

# ==================== MAIN ====================
async def main():
    """Main entry point"""
    console = InteractiveConsole()
    await console.run()

if __name__ == "__main__":
    print("Starting Nexus MCP Console...")
    
    # Handle async in sync context
    if hasattr(asyncio, 'run'):
        asyncio.run(main())
    else:
        # Fallback for older Python
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(main())
        finally:
            loop.close()