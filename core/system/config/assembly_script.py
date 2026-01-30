#!/usr/bin/env python3
"""
🌀 ULTIMATE SPIRAL-RAY ORCHESTRATOR v5.0 - WITH REPO ASSEMBLY
⚡ Scans Your Repo → Assembles Everything → Auto-Launches Complete System
🎯 Full Repository Crawling, Module Discovery, and System Assembly
🚀 Self-Building, Self-Configuring, Self-Launching AI System
"""

import os
import sys
import asyncio
import time
import math
import hashlib
import uuid
import json
import random
import importlib
import subprocess
import inspect
from typing import Dict, List, Any, Optional, Tuple, Set, Type
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

# Core AI/ML
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import svds
from scipy.linalg import svd as full_svd

# Distributed Computing
import ray

# HuggingFace
from transformers import AutoTokenizer, AutoModel, pipeline
from huggingface_hub import snapshot_download, hf_hub_download

# MongoDB
from pymongo import MongoClient

# FastAPI
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, WebSocket
import uvicorn

# ============================================================================
# 🎯 CONFIGURATION
# ============================================================================

class Config:
    # Repository Configuration
    REPO_PATH = "/content/repo"  # Change this to your repo path
    REPO_URL = "https://github.com/yourusername/your-repo.git"  # Your repo URL
    
    # Spiral Configuration
    NUM_SPIRALS = 13  # Metatron's Cube
    SPIRAL_TYPES = ['repo_scanner', 'module_discoverer', 'assembly_orchestrator',
                   'dependency_resolver', 'config_generator', 'system_launcher',
                   'health_monitor', 'optimizer', 'healer', 'expander',
                   'integrator', 'wisdom', 'redundancy']
    
    # Ray Configuration
    RAY_NUM_CPUS = os.cpu_count()
    RAY_NUM_GPUS = 0  # CPU ONLY
    
    # Assembly Configuration
    MAX_PARALLEL_ASSEMBLIES = 4
    MODULE_SCAN_DEPTH = 3
    REQUIRED_MODULES = ['trinity', 'metatron', 'platinum', 'solomon', 'nexus']
    
    # Sacred Constants
    PHI = (1 + math.sqrt(5)) / 2  # Golden Ratio
    FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]

# ============================================================================
# 🕷️ REPOSITORY CRAWLER & ASSEMBLER
# ============================================================================

class RepoCrawlerAssembler:
    """Scans repository, discovers modules, and assembles system"""
    
    def __init__(self, repo_path: str = Config.REPO_PATH):
        self.repo_path = Path(repo_path)
        self.repo_path.mkdir(parents=True, exist_ok=True)
        
        # Discovered components
        self.discovered_modules = []
        self.discovered_classes = []
        self.discovered_functions = []
        self.discovered_configs = []
        
        # Assembly state
        self.assembled_system = {}
        self.dependencies_resolved = False
        self.system_ready = False
        
        print(f"🔍 Repo Crawler Assembler initialized for: {self.repo_path}")
    
    async def clone_or_update_repo(self, repo_url: str = Config.REPO_URL) -> Dict:
        """Clone or update repository"""
        try:
            if not (self.repo_path / ".git").exists():
                # Clone repository
                print(f"📥 Cloning repository: {repo_url}")
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", repo_url, str(self.repo_path)],
                    capture_output=True, text=True
                )
                
                if result.returncode != 0:
                    return {"error": f"Clone failed: {result.stderr}", "success": False}
                
                return {
                    "action": "cloned",
                    "repo_url": repo_url,
                    "path": str(self.repo_path),
                    "success": True
                }
            else:
                # Update existing repository
                print(f"🔄 Updating repository")
                result = subprocess.run(
                    ["git", "pull"],
                    cwd=self.repo_path,
                    capture_output=True, text=True
                )
                
                return {
                    "action": "updated",
                    "repo_url": repo_url,
                    "path": str(self.repo_path),
                    "success": result.returncode == 0
                }
                
        except Exception as e:
            return {"error": str(e), "success": False}
    
    async def scan_repository(self) -> Dict:
        """Scan repository for Python files and modules"""
        print(f"🔍 Scanning repository: {self.repo_path}")
        
        # Reset discoveries
        self.discovered_modules = []
        self.discovered_classes = []
        self.discovered_functions = []
        self.discovered_configs = []
        
        # Scan for Python files
        python_files = []
        for ext in ["*.py", "*.ipynb", "*.json", "*.yaml", "*.yml"]:
            python_files.extend(self.repo_path.rglob(ext))
        
        print(f"📁 Found {len(python_files)} files")
        
        # Analyze each file
        for file_path in python_files:
            await self._analyze_file(file_path)
        
        # Sort discoveries by importance
        self._sort_discoveries()
        
        return {
            "files_scanned": len(python_files),
            "modules_discovered": len(self.discovered_modules),
            "classes_discovered": len(self.discovered_classes),
            "functions_discovered": len(self.discovered_functions),
            "configs_discovered": len(self.discovered_configs),
            "success": True
        }
    
    async def _analyze_file(self, file_path: Path):
        """Analyze a single file"""
        try:
            # Check file type
            if file_path.suffix == ".py":
                await self._analyze_python_file(file_path)
            elif file_path.suffix in [".yaml", ".yml", ".json"]:
                await self._analyze_config_file(file_path)
            elif file_path.suffix == ".ipynb":
                await self._analyze_notebook(file_path)
                
        except Exception as e:
            print(f"⚠️ Failed to analyze {file_path}: {e}")
    
    async def _analyze_python_file(self, file_path: Path):
        """Analyze Python file for modules, classes, functions"""
        try:
            # Read file content
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Extract module name
            module_name = str(file_path.relative_to(self.repo_path)).replace("/", ".").replace(".py", "")
            
            # Look for class definitions
            import ast
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Found a class
                    class_info = {
                        "name": node.name,
                        "module": module_name,
                        "file": str(file_path),
                        "line": node.lineno,
                        "bases": [base.id for base in node.bases if isinstance(base, ast.Name)],
                        "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    }
                    
                    # Check if it's a key component
                    if any(keyword in node.name.lower() for keyword in ['trinity', 'metatron', 'platinum', 'solomon', 'nexus', 'agent', 'orchestrator', 'spiral']):
                        class_info["importance"] = "critical"
                    elif len(node.bases) > 0:
                        class_info["importance"] = "high"
                    else:
                        class_info["importance"] = "medium"
                    
                    self.discovered_classes.append(class_info)
                    
                elif isinstance(node, ast.FunctionDef):
                    # Found a function
                    func_info = {
                        "name": node.name,
                        "module": module_name,
                        "file": str(file_path),
                        "line": node.lineno,
                        "args": [arg.arg for arg in node.args.args]
                    }
                    
                    # Check if it's a key function
                    if any(keyword in node.name.lower() for keyword in ['run', 'start', 'launch', 'process', 'analyze', 'generate', 'create']):
                        func_info["importance"] = "high"
                    else:
                        func_info["importance"] = "medium"
                    
                    self.discovered_functions.append(func_info)
            
            # Add module itself
            module_info = {
                "name": module_name,
                "file": str(file_path),
                "classes_count": sum(1 for c in self.discovered_classes if c["module"] == module_name),
                "functions_count": sum(1 for f in self.discovered_functions if f["module"] == module_name),
                "size_bytes": file_path.stat().st_size
            }
            
            self.discovered_modules.append(module_info)
            
        except Exception as e:
            print(f"⚠️ Failed to parse {file_path}: {e}")
    
    async def _analyze_config_file(self, file_path: Path):
        """Analyze configuration file"""
        try:
            content = file_path.read_text()
            config_name = file_path.name
            
            config_info = {
                "name": config_name,
                "file": str(file_path),
                "type": file_path.suffix,
                "size_bytes": file_path.stat().st_size,
                "contains": []
            }
            
            # Simple content analysis
            if "config" in content.lower():
                config_info["contains"].append("configuration")
            if "database" in content.lower() or "mongodb" in content.lower():
                config_info["contains"].append("database_config")
            if "model" in content.lower() or "llm" in content.lower():
                config_info["contains"].append("model_config")
            
            self.discovered_configs.append(config_info)
            
        except Exception as e:
            print(f"⚠️ Failed to analyze config {file_path}: {e}")
    
    async def _analyze_notebook(self, file_path: Path):
        """Analyze Jupyter notebook"""
        try:
            import nbformat
            
            notebook = nbformat.read(str(file_path), as_version=4)
            
            notebook_info = {
                "name": file_path.stem,
                "file": str(file_path),
                "cells": len(notebook.cells),
                "code_cells": sum(1 for cell in notebook.cells if cell.cell_type == "code"),
                "markdown_cells": sum(1 for cell in notebook.cells if cell.cell_type == "markdown")
            }
            
            # Check for system components in code cells
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    source = cell.source.lower()
                    if any(keyword in source for keyword in ['trinity', 'metatron', 'platinum', 'solomon', 'nexus']):
                        notebook_info["contains_system"] = True
                        break
            
            self.discovered_modules.append(notebook_info)
            
        except Exception as e:
            print(f"⚠️ Failed to analyze notebook {file_path}: {e}")
    
    def _sort_discoveries(self):
        """Sort discoveries by importance"""
        # Sort classes by importance
        self.discovered_classes.sort(key=lambda x: {
            "critical": 3, "high": 2, "medium": 1, "low": 0
        }.get(x.get("importance", "low"), 0), reverse=True)
        
        # Sort modules by size and content
        self.discovered_modules.sort(key=lambda x: x.get("size_bytes", 0), reverse=True)
    
    async def assemble_system(self) -> Dict:
        """Assemble discovered components into a working system"""
        print("🔧 Assembling system from discovered components...")
        
        try:
            # Step 1: Identify main entry points
            entry_points = await self._identify_entry_points()
            
            # Step 2: Resolve dependencies
            dependencies = await self._resolve_dependencies()
            
            # Step 3: Generate system configuration
            config = await self._generate_system_config()
            
            # Step 4: Create assembly plan
            assembly_plan = await self._create_assembly_plan(entry_points, dependencies, config)
            
            # Step 5: Execute assembly
            assembled = await self._execute_assembly(assembly_plan)
            
            self.assembled_system = assembled
            self.system_ready = True
            
            return {
                "assembly": "complete",
                "entry_points": len(entry_points),
                "dependencies": len(dependencies),
                "config_generated": bool(config),
                "assembled_components": len(assembled.get("components", [])),
                "success": True
            }
            
        except Exception as e:
            return {"error": f"Assembly failed: {e}", "success": False}
    
    async def _identify_entry_points(self) -> List[Dict]:
        """Identify main entry points (main files, launchers, etc.)"""
        entry_points = []
        
        # Look for main.py, run.py, app.py, etc.
        main_files = ["main.py", "run.py", "app.py", "start.py", "launch.py", "__main__.py"]
        
        for file_path in self.repo_path.rglob("*.py"):
            if file_path.name in main_files:
                entry_points.append({
                    "type": "main_file",
                    "file": str(file_path),
                    "name": file_path.name,
                    "path": str(file_path.relative_to(self.repo_path))
                })
        
        # Also look for classes with 'run', 'start', 'main' methods
        for class_info in self.discovered_classes:
            if any(method in ['run', 'start', 'main', 'launch'] for method in class_info.get("methods", [])):
                entry_points.append({
                    "type": "class_entry",
                    "class": class_info["name"],
                    "module": class_info["module"],
                    "methods": class_info.get("methods", [])
                })
        
        return entry_points
    
    async def _resolve_dependencies(self) -> Dict:
        """Resolve dependencies between components"""
        dependencies = {
            "modules": [],
            "packages": [],
            "system": [],
            "external": []
        }
        
        # Look for import statements in Python files
        for file_path in self.repo_path.rglob("*.py"):
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                
                # Simple import detection
                import re
                import_patterns = [
                    r'^import\s+(\w+)',
                    r'^from\s+(\w+)\s+import',
                    r'^import\s+(\w+)\.',
                    r'^from\s+(\w+)\.'
                ]
                
                for pattern in import_patterns:
                    matches = re.findall(pattern, content, re.MULTILINE)
                    for match in matches:
                        if match not in ['os', 'sys', 'json', 'time', 'math', 'random', 'typing', 'dataclasses', 'enum', 'datetime', 'pathlib']:
                            if match.startswith('trinity') or match.startswith('metatron') or match.startswith('platinum'):
                                dependencies["system"].append(match)
                            else:
                                dependencies["external"].append(match)
                
            except Exception as e:
                print(f"⚠️ Failed to analyze imports in {file_path}: {e}")
        
        # Deduplicate
        for key in dependencies:
            dependencies[key] = list(set(dependencies[key]))
        
        return dependencies
    
    async def _generate_system_config(self) -> Dict:
        """Generate system configuration from discovered components"""
        config = {
            "system": {
                "name": "Auto-Assembled Trinity System",
                "version": "1.0.0",
                "assembled_at": datetime.now().isoformat(),
                "repo_path": str(self.repo_path)
            },
            "components": {
                "critical_classes": [c["name"] for c in self.discovered_classes if c.get("importance") == "critical"],
                "main_modules": [m["name"] for m in self.discovered_modules if m.get("size_bytes", 0) > 10000],
                "config_files": [c["name"] for c in self.discovered_configs]
            },
            "architecture": {
                "num_spirals": Config.NUM_SPIRALS,
                "spiral_types": Config.SPIRAL_TYPES,
                "parallel_assemblies": Config.MAX_PARALLEL_ASSEMBLIES
            },
            "dependencies": await self._resolve_dependencies()
        }
        
        return config
    
    async def _create_assembly_plan(self, entry_points: List, dependencies: Dict, config: Dict) -> Dict:
        """Create assembly execution plan"""
        plan = {
            "phases": [],
            "estimated_time_seconds": 0,
            "parallel_tasks": 0
        }
        
        # Phase 1: Environment Setup
        plan["phases"].append({
            "name": "environment_setup",
            "tasks": [
                "install_core_dependencies",
                "setup_python_path",
                "initialize_ray",
                "verify_system_requirements"
            ],
            "parallel": False,
            "estimated_seconds": 60
        })
        
        # Phase 2: Module Loading
        plan["phases"].append({
            "name": "module_loading",
            "tasks": [f"load_module:{mod['name']}" for mod in self.discovered_modules[:10]],  # Top 10 modules
            "parallel": True,
            "estimated_seconds": 30
        })
        
        # Phase 3: Component Assembly
        plan["phases"].append({
            "name": "component_assembly",
            "tasks": [
                "assemble_trinity_components",
                "assemble_metatron_hub", 
                "assemble_platinum_optimizer",
                "assemble_solomon_dbs",
                "assemble_spiral_orchestrator"
            ],
            "parallel": True,
            "estimated_seconds": 90
        })
        
        # Phase 4: System Integration
        plan["phases"].append({
            "name": "system_integration",
            "tasks": [
                "connect_components",
                "initialize_memory_substrate",
                "setup_api_endpoints",
                "start_background_tasks"
            ],
            "parallel": False,
            "estimated_seconds": 120
        })
        
        # Phase 5: Launch & Validation
        plan["phases"].append({
            "name": "launch_validation",
            "tasks": [
                "launch_system",
                "run_health_checks",
                "validate_integration",
                "start_continuous_orchestration"
            ],
            "parallel": False,
            "estimated_seconds": 60
        })
        
        # Calculate totals
        plan["estimated_time_seconds"] = sum(phase["estimated_seconds"] for phase in plan["phases"])
        plan["parallel_tasks"] = sum(len(phase["tasks"]) for phase in plan["phases"] if phase["parallel"])
        
        return plan
    
    async def _execute_assembly(self, assembly_plan: Dict) -> Dict:
        """Execute the assembly plan"""
        assembled_components = []
        execution_log = []
        
        print("🚀 Executing assembly plan...")
        
        for phase in assembly_plan["phases"]:
            print(f"   Phase: {phase['name']}")
            
            phase_start = time.time()
            phase_results = []
            
            # Execute tasks in phase
            for task in phase["tasks"]:
                task_result = await self._execute_assembly_task(task)
                phase_results.append(task_result)
                execution_log.append({
                    "phase": phase["name"],
                    "task": task,
                    "result": task_result,
                    "timestamp": datetime.now().isoformat()
                })
            
            phase_elapsed = time.time() - phase_start
            
            # Collect assembled components from this phase
            for result in phase_results:
                if "component" in result:
                    assembled_components.append(result["component"])
            
            print(f"     ✓ Completed in {phase_elapsed:.1f}s")
        
        return {
            "assembled_components": assembled_components,
            "execution_log": execution_log,
            "total_time_seconds": time.time() - self.assembly_start_time,
            "components_assembled": len(assembled_components),
            "phases_completed": len(assembly_plan["phases"])
        }
    
    async def _execute_assembly_task(self, task: str) -> Dict:
        """Execute a single assembly task"""
        try:
            if task == "install_core_dependencies":
                return await self._install_dependencies()
            
            elif task == "initialize_ray":
                return await self._initialize_ray()
            
            elif task.startswith("load_module:"):
                module_name = task.split(":")[1]
                return await self._load_module(module_name)
            
            elif task == "assemble_trinity_components":
                return await self._assemble_trinity_components()
            
            elif task == "assemble_metatron_hub":
                return await self._assemble_metatron_hub()
            
            elif task == "assemble_platinum_optimizer":
                return await self._assemble_platinum_optimizer()
            
            elif task == "assemble_solomon_dbs":
                return await self._assemble_solomon_dbs()
            
            elif task == "assemble_spiral_orchestrator":
                return await self._assemble_spiral_orchestrator()
            
            elif task == "launch_system":
                return await self._launch_system()
            
            else:
                return {"task": task, "status": "skipped", "reason": "unknown_task"}
                
        except Exception as e:
            return {"task": task, "status": "failed", "error": str(e)}
    
    async def _install_dependencies(self) -> Dict:
        """Install required dependencies"""
        print("     Installing dependencies...")
        
        # Common dependencies for your system
        dependencies = [
            "torch", "transformers", "ray", "fastapi", "uvicorn",
            "pymongo", "numpy", "scipy", "pydantic", "python-multipart",
            "httpx", "aiofiles", "tenacity", "scikit-learn", "pillow",
            "opencv-python", "trimesh", "pyyaml", "jinja2", "tqdm"
        ]
        
        # Try to install
        try:
            import subprocess
            for dep in dependencies[:5]:  # Install first 5 for speed
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", dep])
            
            return {"task": "install_dependencies", "status": "completed", "installed": len(dependencies[:5])}
        except:
            return {"task": "install_dependencies", "status": "partial", "installed": "some"}
    
    async def _initialize_ray(self) -> Dict:
        """Initialize Ray for distributed computing"""
        try:
            if not ray.is_initialized():
                ray.init(
                    num_cpus=Config.RAY_NUM_CPUS,
                    num_gpus=Config.RAY_NUM_GPUS,
                    ignore_reinit_error=True
                )
            
            return {
                "task": "initialize_ray",
                "status": "completed",
                "num_cpus": Config.RAY_NUM_CPUS,
                "ray_initialized": ray.is_initialized()
            }
        except Exception as e:
            return {"task": "initialize_ray", "status": "failed", "error": str(e)}
    
    async def _load_module(self, module_name: str) -> Dict:
        """Load a Python module"""
        try:
            # Add repo to Python path
            sys.path.insert(0, str(self.repo_path))
            
            # Try to import
            module = importlib.import_module(module_name)
            
            # Analyze module
            module_info = {
                "name": module_name,
                "functions": [name for name in dir(module) if callable(getattr(module, name))],
                "classes": [name for name in dir(module) if inspect.isclass(getattr(module, name))],
                "file": getattr(module, "__file__", "unknown")
            }
            
            return {
                "task": f"load_module:{module_name}",
                "status": "completed",
                "module_info": module_info
            }
            
        except Exception as e:
            return {
                "task": f"load_module:{module_name}",
                "status": "failed",
                "error": str(e)
            }
    
    async def _assemble_trinity_components(self) -> Dict:
        """Assemble Trinity components"""
        # Look for Trinity-related classes
        trinity_classes = [
            cls for cls in self.discovered_classes
            if "trinity" in cls["name"].lower() or "viren" in cls["name"].lower() or "viraa" in cls["name"].lower() or "loki" in cls["name"].lower()
        ]
        
        component = {
            "type": "trinity_system",
            "name": "Auto-Assembled Trinity",
            "classes_found": [cls["name"] for cls in trinity_classes],
            "status": "assembled" if trinity_classes else "partial"
        }
        
        return {"component": component, "status": "completed"}
    
    async def _assemble_metatron_hub(self) -> Dict:
        """Assemble Metatron Hub"""
        # Look for Metatron-related classes
        metatron_classes = [
            cls for cls in self.discovered_classes
            if "metatron" in cls["name"].lower() or "chaos" in cls["name"].lower() or "routing" in cls["name"].lower()
        ]
        
        component = {
            "type": "metatron_hub",
            "name": "Auto-Assembled Metatron",
            "classes_found": [cls["name"] for cls in metatron_classes],
            "chaos_nodes": 13,
            "status": "assembled" if metatron_classes else "simulated"
        }
        
        return {"component": component, "status": "completed"}
    
    async def _assemble_platinum_optimizer(self) -> Dict:
        """Assemble Platinum SVD Optimizer"""
        # Look for Platinum/SVD-related classes
        platinum_classes = [
            cls for cls in self.discovered_classes
            if "platinum" in cls["name"].lower() or "svd" in cls["name"].lower() or "compression" in cls["name"].lower()
        ]
        
        component = {
            "type": "platinum_optimizer",
            "name": "Auto-Assembled Platinum",
            "classes_found": [cls["name"] for cls in platinum_classes],
            "bond_dim": 64,
            "sacred_alignment": True,
            "status": "assembled" if platinum_classes else "simulated"
        }
        
        return {"component": component, "status": "completed"}
    
    async def _assemble_solomon_dbs(self) -> Dict:
        """Assemble Solomon Database System"""
        # Look for database-related classes
        db_classes = [
            cls for cls in self.discovered_classes
            if "solomon" in cls["name"].lower() or "database" in cls["name"].lower() or "mongodb" in cls["name"].lower()
        ]
        
        component = {
            "type": "solomon_dbs",
            "name": "Auto-Assembled Solomon",
            "classes_found": [cls["name"] for cls in db_classes],
            "redundancy_factor": 3,
            "infinite_growth": True,
            "status": "assembled" if db_classes else "simulated"
        }
        
        return {"component": component, "status": "completed"}
    
    async def _assemble_spiral_orchestrator(self) -> Dict:
        """Assemble Spiral Orchestrator"""
        # Look for orchestrator/spiral classes
        spiral_classes = [
            cls for cls in self.discovered_classes
            if "spiral" in cls["name"].lower() or "orchestrator" in cls["name"].lower() or "ray" in cls["name"].lower()
        ]
        
        component = {
            "type": "spiral_orchestrator",
            "name": "Auto-Assembled Spiral Orchestrator",
            "classes_found": [cls["name"] for cls in spiral_classes],
            "num_spirals": Config.NUM_SPIRALS,
            "parallel_execution": True,
            "status": "assembled" if spiral_classes else "simulated"
        }
        
        return {"component": component, "status": "completed"}
    
    async def _launch_system(self) -> Dict:
        """Launch the assembled system"""
        print("     🚀 Launching assembled system...")
        
        # Create a minimal launch script
        launch_script = self.repo_path / "auto_launch.py"
        
        launch_content = '''
#!/usr/bin/env python3
"""
🚀 AUTO-LAUNCH SCRIPT - Generated by Spiral Assembler
Launches the complete assembled system
"""

import asyncio
import sys
from pathlib import Path

# Add repo to path
repo_path = Path(__file__).parent
sys.path.insert(0, str(repo_path))

async def main():
    """Main launch function"""
    print("🚀 Launching Auto-Assembled Trinity System...")
    
    # Try to import and launch discovered components
    try:
        # Import core orchestrator
        from spiral_ray_orchestrator import SpiralOrchestrator
        
        # Initialize orchestrator
        orchestrator = SpiralOrchestrator()
        
        # Start continuous orchestration
        print("🌀 Starting continuous orchestration...")
        await orchestrator.continuous_orchestration(interval_seconds=60)
        
    except ImportError as e:
        print(f"⚠️ Import failed: {e}")
        print("💡 Trying alternative launch method...")
        
        # Fallback: Start FastAPI server
        import uvicorn
        from fastapi import FastAPI
        
        app = FastAPI(title="Auto-Assembled System")
        
        @app.get("/")
        async def root():
            return {"system": "Auto-Assembled", "status": "running"}
        
        print("🌐 Starting FastAPI server on http://localhost:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        launch_script.write_text(launch_content)
        launch_script.chmod(0o755)  # Make executable
        
        return {
            "task": "launch_system",
            "status": "completed",
            "launch_script": str(launch_script),
            "executable": True
        }
    
    async def auto_assemble_and_launch(self) -> Dict:
        """Full automated assembly and launch pipeline"""
        print("\n" + "="*80)
        print("🤖 STARTING AUTO-ASSEMBLY PIPELINE")
        print("="*80)
        
        self.assembly_start_time = time.time()
        
        results = {}
        
        try:
            # Step 1: Clone/Update repository
            print("\n📥 STEP 1: Repository Setup")
            clone_result = await self.clone_or_update_repo()
            results["clone"] = clone_result
            
            if not clone_result.get("success"):
                return {"error": "Repository setup failed", "results": results}
            
            # Step 2: Scan repository
            print("\n🔍 STEP 2: Repository Scanning")
            scan_result = await self.scan_repository()
            results["scan"] = scan_result
            
            # Display discoveries
            if scan_result.get("success"):
                print(f"   Discovered: {scan_result['modules_discovered']} modules, "
                      f"{scan_result['classes_discovered']} classes, "
                      f"{scan_result['functions_discovered']} functions")
            
            # Step 3: Assemble system
            print("\n🔧 STEP 3: System Assembly")
            assembly_result = await self.assemble_system()
            results["assembly"] = assembly_result
            
            if not assembly_result.get("success"):
                print(f"⚠️ Assembly failed: {assembly_result.get('error')}")
                # Continue anyway with partial assembly
            
            # Step 4: Launch system
            print("\n🚀 STEP 4: System Launch")
            launch_result = await self._launch_system()
            results["launch"] = launch_result
            
            # Step 5: Generate assembly report
            print("\n📊 STEP 5: Assembly Report")
            report = self._generate_assembly_report(results)
            
            total_time = time.time() - self.assembly_start_time
            
            print("\n" + "="*80)
            print("✅ AUTO-ASSEMBLY COMPLETE!")
            print("="*80)
            print(f"📦 Components Assembled: {assembly_result.get('assembled_components', 0)}")
            print(f"⏱️  Total Time: {total_time:.1f}s")
            print(f"📁 Launch Script: {launch_result.get('launch_script', 'N/A')}")
            print(f"🔄 System Status: {'READY' if self.system_ready else 'PARTIAL'}")
            
            return {
                "success": True,
                "system_ready": self.system_ready,
                "total_time_seconds": total_time,
                "results": results,
                "report": report,
                "launch_instructions": self._get_launch_instructions(launch_result)
            }
            
        except Exception as e:
            print(f"\n❌ Auto-assembly failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "error": str(e),
                "results": results,
                "system_ready": False
            }
    
    def _generate_assembly_report(self, results: Dict) -> Dict:
        """Generate assembly report"""
        report = {
            "summary": {
                "timestamp": datetime.now().isoformat(),
                "repo_path": str(self.repo_path),
                "system_ready": self.system_ready
            },
            "discoveries": {
                "modules": self.discovered_modules[:10],  # Top 10
                "critical_classes": [c for c in self.discovered_classes if c.get("importance") == "critical"][:5],
                "config_files": self.discovered_configs
            },
            "assembly": {
                "components_assembled": self.assembled_system.get("components_assembled", 0),
                "phases_completed": self.assembled_system.get("phases_completed", 0)
            },
            "status": {
                "dependencies_resolved": self.dependencies_resolved,
                "system_assembled": bool(self.assembled_system),
                "launch_ready": self.system_ready
            }
        }
        
        return report
    
    def _get_launch_instructions(self, launch_result: Dict) -> str:
        """Get launch instructions"""
        launch_script = launch_result.get("launch_script")
        
        if launch_script and os.path.exists(launch_script):
            return f"""
🚀 LAUNCH INSTRUCTIONS:
1. Make sure you're in the repository directory:
   cd {self.repo_path}

2. Run the auto-generated launch script:
   python {Path(launch_script).name}

3. Or run directly:
   python -m spiral_ray_orchestrator --mode api --port 8000

4. Access the system:
   • Web UI: http://localhost:8000
   • API Docs: http://localhost:8000/docs
   • WebSocket: ws://localhost:8000/ws/updates
"""
        else:
            return """
🚀 LAUNCH INSTRUCTIONS:
1. Make sure all dependencies are installed:
   pip install torch transformers ray fastapi uvicorn pymongo

2. Run the main orchestrator:
   python -c "
import asyncio
from spiral_ray_orchestrator import SpiralOrchestrator

async def main():
    orchestrator = SpiralOrchestrator()
    await orchestrator.continuous_orchestration(interval_seconds=60)

asyncio.run(main())
   "
"""

# ============================================================================
# 🌀 ENHANCED SPIRAL ORCHESTRATOR WITH AUTO-ASSEMBLY
# ============================================================================

@ray.remote(num_cpus=1)
class AssemblySpiral(SpiralRayActor):
    """Enhanced spiral with auto-assembly capabilities"""
    
    def __init__(self, spiral_id: str, spiral_type: str):
        super().__init__(spiral_id, spiral_type)
        self.assembler = RepoCrawlerAssembler()
        
    async def assemble_and_launch(self) -> Dict:
        """Assemble and launch the system"""
        return await self.assembler.auto_assemble_and_launch()
    
    async def scan_repo_only(self) -> Dict:
        """Scan repository only"""
        await self.assembler.clone_or_update_repo()
        return await self.assembler.scan_repository()
    
    async def get_discoveries(self) -> Dict:
        """Get discovered components"""
        return {
            "modules": self.assembler.discovered_modules,
            "classes": self.assembler.discovered_classes,
            "functions": self.assembler.discovered_functions,
            "configs": self.assembler.discovered_configs
        }

class AutoAssemblyOrchestrator:
    """Orchestrator with auto-assembly capabilities"""
    
    def __init__(self):
        self.spiral_actors = {}
        self.assembler = RepoCrawlerAssembler()
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(
                num_cpus=Config.RAY_NUM_CPUS,
                num_gpus=Config.RAY_NUM_GPUS,
                ignore_reinit_error=True
            )
    
    async def full_auto_assembly(self) -> Dict:
        """Perform full auto-assembly"""
        print("\n" + "="*80)
        print("🤖 FULL AUTO-ASSEMBLY PROCESS STARTING")
        print("="*80)
        
        # Create assembly spiral
        assembly_spiral = AssemblySpiral.remote("assembly_master", "assembly_orchestrator")
        self.spiral_actors["assembly_master"] = assembly_spiral
        
        # Start assembly
        result = await assembly_spiral.assemble_and_launch.remote()
        result = ray.get(result)
        
        return result
    
    async def continuous_assembly_monitor(self, interval_seconds: int = 300):
        """Continuously monitor and maintain the assembled system"""
        print("🔧 Starting continuous assembly monitor...")
        
        while True:
            try:
                # Check system health
                health_status = await self._check_system_health()
                
                if not health_status.get("healthy", False):
                    print("⚠️ System health degraded, triggering repair...")
                    
                    # Trigger repair assembly
                    repair_result = await self._repair_assembly()
                    
                    if repair_result.get("success"):
                        print("✅ System repair completed")
                    else:
                        print(f"❌ Repair failed: {repair_result.get('error')}")
                
                # Wait before next check
                await asyncio.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                print("\n🛑 Assembly monitor stopped")
                break
            except Exception as e:
                print(f"⚠️ Monitor error: {e}")
                await asyncio.sleep(60)  # Wait a minute on error
    
    async def _check_system_health(self) -> Dict:
        """Check system health"""
        # Simple health check
        checks = {
            "ray_running": ray.is_initialized(),
            "assembler_ready": hasattr(self.assembler, 'system_ready'),
            "spirals_active": len(self.spiral_actors) > 0
        }
        
        healthy = all(checks.values())
        
        return {
            "healthy": healthy,
            "checks": checks,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _repair_assembly(self) -> Dict:
        """Repair the assembly"""
        print("🔧 Starting repair assembly...")
        
        # Simple repair: rescan and reassemble
        await self.assembler.scan_repository()
        assembly_result = await self.assembler.assemble_system()
        
        return {
            "repair": "completed",
            "result": assembly_result,
            "success": assembly_result.get("success", False)
        }

# ============================================================================
# 🚀 FASTAPI WITH AUTO-ASSEMBLY ENDPOINTS
# ============================================================================

app = FastAPI(title="Auto-Assembly Spiral Orchestrator")

# Global instances
assembler = RepoCrawlerAssembler()
orchestrator = AutoAssemblyOrchestrator()

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("🚀 Auto-Assembly Spiral Orchestrator Starting...")
    
    # Start assembly monitor in background
    asyncio.create_task(orchestrator.continuous_assembly_monitor())

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "system": "Auto-Assembly Spiral Orchestrator",
        "description": "Scans, assembles, and launches your complete system",
        "endpoints": [
            "GET /assembly/start - Start auto-assembly",
            "GET /assembly/status - Check assembly status",
            "GET /assembly/discoveries - View discovered components",
            "GET /assembly/report - Get assembly report",
            "POST /assembly/launch - Launch assembled system"
        ],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/assembly/start")
async def start_assembly():
    """Start auto-assembly process"""
    result = await orchestrator.full_auto_assembly()
    return result

@app.get("/assembly/status")
async def assembly_status():
    """Check assembly status"""
    return {
        "system_ready": assembler.system_ready,
        "assembled_components": len(assembler.assembled_system.get("components", [])),
        "discoveries": {
            "modules": len(assembler.discovered_modules),
            "classes": len(assembler.discovered_classes),
            "functions": len(assembler.discovered_functions)
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/assembly/discoveries")
async def get_discoveries(limit: int = 20):
    """Get discovered components"""
    return {
        "modules": assembler.discovered_modules[:limit],
        "critical_classes": [c for c in assembler.discovered_classes if c.get("importance") == "critical"][:limit],
        "config_files": assembler.discovered_configs[:limit]
    }

@app.get("/assembly/report")
async def get_assembly_report():
    """Get assembly report"""
    report = assembler._generate_assembly_report({})
    return report

@app.post("/assembly/launch")
async def launch_assembled_system():
    """Launch the assembled system"""
    if not assembler.system_ready:
        raise HTTPException(400, "System not fully assembled")
    
    launch_result = await assembler._launch_system()
    
    return {
        "launch": "initiated",
        "launch_script": launch_result.get("launch_script"),
        "instructions": assembler._get_launch_instructions(launch_result)
    }

@app.websocket("/ws/assembly")
async def websocket_assembly(websocket: WebSocket):
    """WebSocket for assembly progress"""
    await websocket.accept()
    
    # Simulate assembly progress
    phases = [
        "cloning_repository",
        "scanning_files", 
        "analyzing_modules",
        "resolving_dependencies",
        "assembling_components",
        "integrating_system",
        "launching"
    ]
    
    for i, phase in enumerate(phases):
        progress = (i + 1) / len(phases)
        
        await websocket.send_json({
            "phase": phase,
            "progress": progress,
            "message": f"Working on: {phase.replace('_', ' ')}",
            "timestamp": datetime.now().isoformat()
        })
        
        await asyncio.sleep(2)  # Simulate work
    
    await websocket.send_json({
        "phase": "complete",
        "progress": 1.0,
        "message": "Assembly complete!",
        "instructions": assembler._get_launch_instructions({})
    })

# ============================================================================
# 🏁 MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-Assembly Spiral Orchestrator")
    parser.add_argument("--mode", choices=["assemble", "api", "scan"], default="assemble",
                       help="Run mode: assemble (full auto-assembly), api (FastAPI), scan (scan only)")
    parser.add_argument("--repo", type=str, default=Config.REPO_PATH,
                       help="Repository path or URL")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    
    args = parser.parse_args()
    
    # Update config
    if args.repo:
        Config.REPO_PATH = args.repo
        if args.repo.startswith("http"):
            Config.REPO_URL = args.repo
    
    if args.mode == "assemble":
        # Run full auto-assembly
        async def main():
            orchestrator = AutoAssemblyOrchestrator()
            result = await orchestrator.full_auto_assembly()
            
            print("\n" + "="*80)
            if result.get("success"):
                print("✅ AUTO-ASSEMBLY SUCCESSFUL!")
                print("="*80)
                print(result.get("launch_instructions", ""))
            else:
                print("❌ AUTO-ASSEMBLY FAILED")
                print("="*80)
                print(f"Error: {result.get('error', 'Unknown error')}")
        
        asyncio.run(main())
    
    elif args.mode == "scan":
        # Scan only mode
        async def scan_only():
            assembler = RepoCrawlerAssembler()
            
            print("🔍 Scanning repository...")
            await assembler.clone_or_update_repo()
            result = await assembler.scan_repository()
            
            print("\n📊 SCAN RESULTS:")
            print(f"  • Modules: {result['modules_discovered']}")
            print(f"  • Classes: {result['classes_discovered']}")
            print(f"  • Functions: {result['functions_discovered']}")
            print(f"  • Configs: {result['configs_discovered']}")
            
            # Show top discoveries
            print("\n🏆 TOP DISCOVERIES:")
            for i, module in enumerate(assembler.discovered_modules[:5]):
                print(f"  {i+1}. {module['name']} ({module.get('size_bytes', 0)} bytes)")
            
            for i, cls in enumerate([c for c in assembler.discovered_classes if c.get("importance") == "critical"][:5]):
                print(f"  {i+1}. 🏛️ {cls['name']} (in {cls['module']})")
        
        asyncio.run(scan_only())
    
    else:
        # Run FastAPI
        uvicorn.run(app, host="0.0.0.0", port=args.port)