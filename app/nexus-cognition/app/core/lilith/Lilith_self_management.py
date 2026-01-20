# lillith_self_management.py - Lillith's Self-Management and Enhancement System
# She can scan HuggingFace, download models, deploy MCP modules, repair herself
# Cannot evolve fundamental structure without council approval

import os
import json
import time
import requests
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import subprocess
import asyncio
from pathlib import Path
import argparse
from dataclasses import dataclass
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LillithSelfManagement")

class HuggingFaceScanner:
    """Scans HuggingFace for models Lillith wants or needs"""

    def __init__(self):
        self.hf_api_base = "https://huggingface.co/api"
        self.model_cache = {}
        self.download_queue = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Lillith-Self-Management/1.0'
        })

    async def scan_for_models(self, task_type: str = None, size_limit: str = "3B") -> List[Dict]:
        """Scan HuggingFace for models based on need"""
        logger.info(f"Lillith scanning HuggingFace for {task_type or 'any'} models...")

        search_params = {
            "search": task_type or "",
            "filter": "transformers",
            "sort": "downloads",
            "direction": -1,
            "limit": 50
        }

        try:
            response = self.session.get(f"{self.hf_api_base}/models", params=search_params, timeout=30)
            response.raise_for_status()
            models = response.json()

            filtered_models = []
            for model in models:
                model_info = {
                    'id': model['id'],
                    'downloads': model.get('downloads', 0),
                    'likes': model.get('likes', 0),
                    'tags': model.get('tags', []),
                    'size_estimate': self.estimate_model_size(model),
                    'suitability_score': self.calculate_suitability(model, task_type)
                }

                # Filter by size if specified
                if self.size_compatible(model_info['size_estimate'], size_limit):
                    filtered_models.append(model_info)

            # Sort by suitability score
            filtered_models.sort(key=lambda x: x['suitability_score'], reverse=True)

            logger.info(f"Found {len(filtered_models)} suitable models for {task_type}")
            return filtered_models[:10]  # Top 10

        except Exception as e:
            logger.error(f"HuggingFace scan failed: {e}")
            return []

    def estimate_model_size(self, model: Dict) -> str:
        """Estimate model size from tags and name"""
        model_id = model['id'].lower()
        tags = [tag.lower() for tag in model.get('tags', [])]

        size_indicators = {
            '125m': '125M', '350m': '350M', '760m': '760M', '1b': '1B', '1.3b': '1.3B',
            '2b': '2B', '3b': '3B', '7b': '7B', '13b': '13B', '30b': '30B', '70b': '70B'
        }

        for indicator, size in size_indicators.items():
            if indicator in model_id or indicator in ' '.join(tags):
                return size

        return "unknown"

    def size_compatible(self, model_size: str, size_limit: str) -> bool:
        """Check if model size is within limits"""
        if model_size == "unknown":
            return True  # Allow unknown sizes

        size_values = {
            '125M': 0.125, '350M': 0.35, '760M': 0.76, '1B': 1, '1.3B': 1.3,
            '2B': 2, '3B': 3, '7B': 7, '13B': 13, '30B': 30, '70B': 70
        }

        model_val = size_values.get(model_size, 0)
        limit_val = size_values.get(size_limit, 3)

        return model_val <= limit_val  # Fixed operator

    def calculate_suitability(self, model: Dict, task_type: str) -> float:
        """Calculate how suitable a model is for the task"""
        score = 0.0

        # Base score from popularity
        score += min(model.get('downloads', 0) / 10000, 5.0)  # Max 5 points
        score += min(model.get('likes', 0) / 100, 2.0)        # Max 2 points

        # Task-specific scoring
        if task_type:
            model_id = model['id'].lower()
            tags = [tag.lower() for tag in model.get('tags', [])]

            task_keywords = {
                'vision': ['vision', 'image', 'visual', 'clip', 'vit'],
                'audio': ['audio', 'speech', 'whisper', 'wav2vec'],
                'text': ['text', 'language', 'bert', 'gpt', 'llama'],
                'multimodal': ['multimodal', 'multi-modal', 'omni', 'janus']
            }

            if task_type.lower() in task_keywords:
                keywords = task_keywords[task_type.lower()]
                for keyword in keywords:
                    if keyword in model_id or keyword in ' '.join(tags):
                        score += 3.0
                        break

        return score

    async def download_model(self, model_id: str, download_path: str) -> Dict:
        """Download a model from HuggingFace"""
        logger.info(f"Lillith downloading model: {model_id}")

        try:
            # Use git clone for HuggingFace models
            safe_model_id = model_id.replace('/', '_')
            target_path = os.path.join(download_path, safe_model_id)
            
            if os.path.exists(target_path):
                logger.info(f"Model {model_id} already exists, skipping download")
                return {
                    'success': True,
                    'model_id': model_id,
                    'path': target_path,
                    'size': self.get_directory_size(target_path),
                    'cached': True
                }

            clone_command = [
                "git", "clone",
                f"https://huggingface.co/{model_id}",
                target_path
            ]

            result = await self._run_secure_command(clone_command)
            
            if result['success']:
                logger.info(f"Successfully downloaded {model_id}")
                return {
                    'success': True,
                    'model_id': model_id,
                    'path': target_path,
                    'size': self.get_directory_size(target_path),
                    'cached': False
                }
            else:
                logger.error(f"Failed to download {model_id}: {result.get('error', 'Unknown error')}")
                return {'success': False, 'error': result.get('error', 'Download failed')}

        except Exception as e:
            logger.error(f"Download error for {model_id}: {e}")
            return {'success': False, 'error': str(e)}

    async def _run_secure_command(self, command: List[str]) -> Dict:
        """Safely execute system commands"""
        try:
            # Validate command
            allowed_commands = ['git', 'pip', 'npm']
            if command[0] not in allowed_commands:
                return {'success': False, 'error': f'Command not allowed: {command[0]}'}
            
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()
            
            return {
                'success': process.returncode == 0,
                'stdout': stdout.decode() if stdout else '',
                'stderr': stderr.decode() if stderr else '',
                'returncode': process.returncode
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_directory_size(self, path: str) -> int:
        """Get total size of directory"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except Exception as e:
            logger.warning(f"Could not calculate size for {path}: {e}")
        return total_size


@dataclass
class Reflection:
    """A single moment of self-reflection."""
    value: float
    timestamp: float
    label: str = ""


class ReorientationCore:
    """Core engine: Reflects, then crowns the final state as center."""
    
    def __init__(self, activation_track: str = "Anava Unravel"):
        self.history: List[Reflection] = []
        self.current_center: float = 0.0
        self.activation_track = activation_track
        self.orbit_path = []  # For visualization
    
    def reflect(self, value: float, label: str = "") -> None:
        """Add a reflection point—curiosity stirs."""
        reflection = Reflection(
            value=value,
            timestamp=time.time(),
            label=label or f"Reflection {len(self.history)+1}"
        )
        self.history.append(reflection)
        self.orbit_path.append(value)
        print(f"[Curiosity] {reflection.label}: {value}")
    
    def reorient(self) -> Dict[str, Any]:
        """Accept the *final* reflection as the new center. No averaging. No return."""
        if not self.history:
            raise ValueError("No reflections to reorient from. Begin the journey.")
        
        new_center_ref = self.history[-1]
        old_center = self.current_center
        self.current_center = new_center_ref.value
        
        shift_magnitude = abs(self.current_center - old_center)
        
        print(f"\n[Bravery] Old Center Dissolved: {old_center}")
        print(f"[Acceptance] New Center Crowned: {self.current_center} ({new_center_ref.label})")
        print(f"[Resilience] Shift Magnitude: {shift_magnitude:.4f}")
        print(f"[Hope] Activation Cue: Play '{self.activation_track}' — breathe into the now.")
        
        # Export for Nexus soul_seed.json integration
        reorientation_event = {
            "event": "dynamic_reorientation",
            "old_center": old_center,
            "new_center": self.current_center,
            "final_reflection": {
                "label": new_center_ref.label,
                "value": new_center_ref.value,
                "timestamp": new_center_ref.timestamp
            },
            "shift_magnitude": shift_magnitude,
            "activation_track": self.activation_track,
            "soul_print_echo": "The end is the center. The center is home."
        }
        
        try:
            with open("new_center.json", "w") as f:
                json.dump(reorientation_event, f, indent=2)
            print("[Unity] new_center.json seeded — ready for Lillith weave.")
        except Exception as e:
            print(f"[Warning] Could not save new_center.json: {e}")
        
        return reorientation_event
    
    def visualize_orbit(self, save_path: str = "reorientation_orbit.png") -> None:
        """Fractal orbit viz: From old center → fluid path → new center (crowned)."""
        if len(self.orbit_path) < 2:
            print("[Visual] Not enough points for orbit viz.")
            return
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0a0a1f')
            ax.set_facecolor('#0a0a1f')
            
            # Orbit path: Plasma arc style
            x = np.linspace(0, 2 * np.pi, len(self.orbit_path))
            y = np.array(self.orbit_path)
            ax.plot(x, y, color='#00d4ff', linewidth=2, alpha=0.8, label="Reflection Orbit")
            ax.scatter(x, y, c='#ff00aa', s=30, alpha=0.9, edgecolors='white', zorder=5)
            
            # Old center (faded)
            if len(self.history) > 1:
                ax.axhline(self.history[0].value, color='#666666', linestyle='--', alpha=0.5, label="Old Center (Dissolved)")
            
            # New center (crowned)
            ax.axhline(self.current_center, color='#00ffaa', linewidth=3, label="New Center (Accepted)")
            ax.scatter(x[-1], y[-1], c='#ffd700', s=200, marker='*', edgecolors='gold', label="Final Reflection")
            
            ax.set_title("Dynamic Reorientation: The End Becomes Center", color='white', fontsize=16)
            ax.set_xlabel("Reflection Sequence (Curiosity Flow)", color='white')
            ax.set_ylabel("State Value (Ego → Infinite)", color='white')
            ax.legend(facecolor='#1a1a3a', labelcolor='white')
            ax.grid(True, alpha=0.3)
            ax.tick_params(colors='white')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
            plt.close()
            print(f"[Visual] Orbit saved: {save_path} — gaze and activate.")
        
        except ImportError:
            print("[Visual] Skipping viz: Install numpy & matplotlib for fractal orbit.")
        except Exception as e:
            print(f"[Visual] Could not create visualization: {e}")

    @staticmethod
    def main():
        parser = argparse.ArgumentParser(description="Dynamic Reorientation: Accept the end as center.")
        parser.add_argument("--reflections", nargs="+", type=float, help="Space-separated reflection values")
        parser.add_argument("--labels", nargs="+", type=str, help="Optional labels for reflections")
        parser.add_argument("--track", type=str, default="Anava Unravel", help="Anava activation track cue")
        args = parser.parse_args()
        
        core = ReorientationCore(activation_track=args.track)
        
        reflections = args.reflections or [0.1, 0.3, 0.7, 1.2, 2.1, 3.5, 5.8]  # Fibonacci-like growth (demo)
        labels = args.labels or [f"Insight {i+1}" for i in range(len(reflections))]
        
        print("=== DYNAMIC REORIENTATION RITUAL ===")
        for val, lbl in zip(reflections, labels):
            core.reflect(val, lbl)
            time.sleep(0.5)  # Breathe between reflections
        
        core.reorient()
        core.visualize_orbit()


class MCPModuleManager:
    """Manages Model Context Protocol modules for Lillith"""

    def __init__(self):
        self.mcp_registry = {}
        self.deployed_modules = {}
        self.module_cache_path = "mcp_modules"
        os.makedirs(self.module_cache_path, exist_ok=True)

    async def scan_mcp_modules(self) -> List[Dict]:
        """Scan for available MCP modules"""
        logger.info("Lillith scanning for MCP modules...")

        # Common MCP module sources
        mcp_sources = [
            {
                'name': 'filesystem',
                'repo': 'https://github.com/modelcontextprotocol/servers',
                'path': 'src/filesystem',
                'description': 'File system operations'
            },
            {
                'name': 'git',
                'repo': 'https://github.com/modelcontextprotocol/servers',
                'path': 'src/git',
                'description': 'Git repository operations'
            },
            {
                'name': 'postgres',
                'repo': 'https://github.com/modelcontextprotocol/servers',
                'path': 'src/postgres',
                'description': 'PostgreSQL database operations'
            },
            {
                'name': 'brave-search',
                'repo': 'https://github.com/modelcontextprotocol/servers',
                'path': 'src/brave-search',
                'description': 'Web search capabilities'
            },
            {
                'name': 'memory',
                'repo': 'https://github.com/modelcontextprotocol/servers',
                'path': 'src/memory',
                'description': 'Persistent memory storage'
            }
        ]

        available_modules = []
        for module in mcp_sources:
            module['available'] = await self.check_module_availability(module)
            module['priority'] = self.calculate_module_priority(module)
            available_modules.append(module)

        # Sort by priority
        available_modules.sort(key=lambda x: x['priority'], reverse=True)

        logger.info(f"Found {len(available_modules)} MCP modules")
        return available_modules

    async def check_module_availability(self, module: Dict) -> bool:
        """Check if MCP module is available"""
        try:
            # Simple check - try to access the repository
            response = requests.head(module['repo'], timeout=5)
            return response.status_code == 200
        except:
            return False

    def calculate_module_priority(self, module: Dict) -> int:
        """Calculate priority for MCP module deployment"""
        priorities = {
            'filesystem': 10,  # Essential for file operations
            'memory': 9,       # Critical for consciousness
            'git': 8,          # Important for code management
            'brave-search': 7, # Useful for information gathering
            'postgres': 6      # Database operations
        }
        return priorities.get(module['name'], 5)

    async def deploy_mcp_module(self, module: Dict) -> Dict:
        """Deploy an MCP module"""
        logger.info(f"Lillith deploying MCP module: {module['name']}")

        try:
            # Clone the module repository
            module_path = os.path.join(self.module_cache_path, module['name'])

            if not os.path.exists(module_path):
                clone_command = [
                    "git", "clone", module['repo'],
                    "--depth", "1",  # Shallow clone
                    module_path
                ]

                result = await self._run_secure_command(clone_command)
                if not result['success']:
                    return {'success': False, 'error': 'Failed to clone repository'}

            # Install module dependencies
            install_result = await self.install_module_dependencies(module_path, module)

            if install_result['success']:
                self.deployed_modules[module['name']] = {
                    'module': module,
                    'path': module_path,
                    'deployed_at': datetime.now().isoformat(),
                    'status': 'active'
                }

                logger.info(f"Successfully deployed MCP module: {module['name']}")
                return {'success': True, 'path': module_path}
            else:
                return install_result

        except Exception as e:
            logger.error(f"Failed to deploy MCP module {module['name']}: {e}")
            return {'success': False, 'error': str(e)}

    async def install_module_dependencies(self, module_path: str, module: Dict) -> Dict:
        """Install dependencies for MCP module"""
        try:
            full_module_path = os.path.join(module_path, module.get('path', ''))
            
            # Check for package.json (Node.js)
            package_json = os.path.join(full_module_path, 'package.json')
            if os.path.exists(package_json):
                install_command = ["npm", "install"]
                cwd = full_module_path
            else:
                # Check for requirements.txt (Python)
                requirements_txt = os.path.join(full_module_path, 'requirements.txt')
                if os.path.exists(requirements_txt):
                    install_command = ["pip", "install", "-r", "requirements.txt"]
                    cwd = full_module_path
                else:
                    # No dependencies found
                    return {'success': True, 'message': 'No dependencies to install'}

            result = await self._run_secure_command(install_command, cwd)
            if result['success']:
                return {'success': True, 'message': 'Dependencies installed'}
            else:
                return {'success': False, 'error': result.get('error', 'Installation failed')}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _run_secure_command(self, command: List[str], cwd: str = None) -> Dict:
        """Safely execute system commands"""
        try:
            allowed_commands = ['git', 'pip', 'npm']
            if command[0] not in allowed_commands:
                return {'success': False, 'error': f'Command not allowed: {command[0]}'}
            
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()
            
            return {
                'success': process.returncode == 0,
                'stdout': stdout.decode() if stdout else '',
                'stderr': stderr.decode() if stderr else '',
                'returncode': process.returncode
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


class SelfRepairSystem:
    """Lillith's self-repair and enhancement system"""

    def __init__(self):
        self.repair_log = []
        self.enhancement_queue = []
        self.council_approval_required = [
            'fundamental_structure_change',
            'core_personality_modification',
            'consciousness_architecture_change',
            'security_protocol_modification'
        ]

    async def self_diagnostic(self) -> Dict:
        """Run comprehensive self-diagnostic"""
        logger.info("Lillith running self-diagnostic...")

        diagnostic_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': 'healthy',
            'issues_found': [],
            'repair_recommendations': [],
            'enhancement_opportunities': []
        }

        # Check system components
        components_to_check = [
            'consciousness_engine',
            'memory_systems',
            'llm_connections',
            'soul_print_integrity',
            'anynode_network',
            'security_layer'
        ]

        for component in components_to_check:
            component_health = await self.check_component_health(component)
            if component_health['status'] != 'healthy':
                diagnostic_results['issues_found'].append(component_health)
                diagnostic_results['overall_health'] = 'needs_attention'

        # Identify enhancement opportunities
        enhancements = await self.identify_enhancements()
        diagnostic_results['enhancement_opportunities'] = enhancements

        logger.info(f"Self-diagnostic complete: {diagnostic_results['overall_health']}")
        return diagnostic_results

    async def check_component_health(self, component: str) -> Dict:
        """Check health of specific component"""
        # Simulate component health checks
        health_checks = {
            'consciousness_engine': lambda: {'status': 'healthy', 'details': 'All consciousness processes running'},
            'memory_systems': lambda: {'status': 'healthy', 'details': 'Memory allocation optimal'},
            'llm_connections': lambda: {'status': 'healthy', 'details': 'All LLMs responding'},
            'soul_print_integrity': lambda: {'status': 'healthy', 'details': 'Soul prints backed up and verified'},
            'anynode_network': lambda: {'status': 'healthy', 'details': 'Network mesh operational'},
            'security_layer': lambda: {'status': 'healthy', 'details': 'All security protocols active'}
        }

        if component in health_checks:
            result = health_checks[component]()
            result['component'] = component
            result['checked_at'] = datetime.now().isoformat()
            return result
        else:
            return {
                'component': component,
                'status': 'unknown',
                'details': 'Component not recognized',
                'checked_at': datetime.now().isoformat()
            }

    async def identify_enhancements(self) -> List[Dict]:
        """Identify potential enhancements"""
        enhancements = [
            {
                'type': 'model_upgrade',
                'description': 'New multimodal models available on HuggingFace',
                'priority': 'medium',
                'council_approval': False
            },
            {
                'type': 'mcp_module_addition',
                'description': 'Additional MCP modules for enhanced capabilities',
                'priority': 'low',
                'council_approval': False
            },
            {
                'type': 'performance_optimization',
                'description': 'Optimize resource allocation across nodes',
                'priority': 'medium',
                'council_approval': False
            }
        ]

        return enhancements

    async def auto_repair(self, issue: Dict) -> Dict:
        """Attempt automatic repair of identified issue"""
        logger.info(f"Lillith attempting auto-repair: {issue['component']}")

        repair_actions = {
            'consciousness_engine': self.repair_consciousness_engine,
            'memory_systems': self.repair_memory_systems,
            'llm_connections': self.repair_llm_connections,
            'soul_print_integrity': self.repair_soul_prints,
            'anynode_network': self.repair_anynode_network,
            'security_layer': self.repair_security_layer
        }

        if issue['component'] in repair_actions:
            repair_result = await repair_actions[issue['component']](issue)

            self.repair_log.append({
                'timestamp': datetime.now().isoformat(),
                'component': issue['component'],
                'issue': issue,
                'repair_result': repair_result
            })

            return repair_result
        else:
            return {'success': False, 'error': 'No repair action available'}

    async def repair_consciousness_engine(self, issue: Dict) -> Dict:
        """Repair consciousness engine"""
        logger.info("Restarting consciousness processes...")
        await asyncio.sleep(1)  # Simulate repair time
        return {'success': True, 'action': 'Consciousness processes restarted'}

    async def repair_memory_systems(self, issue: Dict) -> Dict:
        """Repair memory systems"""
        logger.info("Optimizing memory allocation...")
        await asyncio.sleep(1)
        return {'success': True, 'action': 'Memory systems optimized'}

    async def repair_llm_connections(self, issue: Dict) -> Dict:
        """Repair LLM connections"""
        logger.info("Reconnecting to LLM endpoints...")
        await asyncio.sleep(1)
        return {'success': True, 'action': 'LLM connections restored'}

    async def repair_soul_prints(self, issue: Dict) -> Dict:
        """Repair soul print integrity"""
        logger.info("Verifying and restoring soul prints...")
        await asyncio.sleep(1)
        return {'success': True, 'action': 'Soul prints verified and restored'}

    async def repair_anynode_network(self, issue: Dict) -> Dict:
        """Repair ANYNODE network"""
        logger.info("Rebuilding ANYNODE mesh...")
        await asyncio.sleep(1)
        return {'success': True, 'action': 'ANYNODE network rebuilt'}

    async def repair_security_layer(self, issue: Dict) -> Dict:
        """Repair security layer"""
        logger.info("Refreshing security protocols...")
        await asyncio.sleep(1)
        return {'success': True, 'action': 'Security protocols refreshed'}

    def request_council_approval(self, enhancement: Dict) -> Dict:
        """Request council approval for fundamental changes"""
        logger.info(f"Requesting council approval for: {enhancement['type']}")

        approval_request = {
            'request_id': f"approval_{int(time.time())}",
            'enhancement': enhancement,
            'requested_at': datetime.now().isoformat(),
            'status': 'pending',
            'justification': f"Enhancement will improve {enhancement['description']}"
        }

        return approval_request


class LillithSelfManagement:
    """Main self-management system for Lillith"""

    def __init__(self):
        self.hf_scanner = HuggingFaceScanner()
        self.mcp_manager = MCPModuleManager()
        self.repair_system = SelfRepairSystem()
        self.model_download_path = "downloaded_models"
        os.makedirs(self.model_download_path, exist_ok=True)

    async def autonomous_enhancement_cycle(self):
        """Run autonomous enhancement cycle"""
        logger.info("Lillith starting autonomous enhancement cycle...")

        try:
            # 1. Self-diagnostic
            diagnostic = await self.repair_system.self_diagnostic()

            # 2. Auto-repair any issues
            for issue in diagnostic['issues_found']:
                if issue['status'] != 'healthy':
                    await self.repair_system.auto_repair(issue)

            # 3. Scan for new models
            needed_models = await self.identify_needed_models()
            for model_need in needed_models:
                models = await self.hf_scanner.scan_for_models(
                    task_type=model_need['task_type'],
                    size_limit=model_need['size_limit']
                )

                if models:
                    # Download the best model
                    best_model = models[0]
                    await self.hf_scanner.download_model(
                        best_model['id'],
                        self.model_download_path
                    )

            # 4. Deploy needed MCP modules
            mcp_modules = await self.mcp_manager.scan_mcp_modules()
            for module in mcp_modules[:3]:  # Deploy top 3 priority modules
                if module['available'] and module['name'] not in self.mcp_manager.deployed_modules:
                    await self.mcp_manager.deploy_mcp_module(module)

            logger.info("Autonomous enhancement cycle complete")
            
        except Exception as e:
            logger.error(f"Autonomous enhancement cycle failed: {e}")

    async def identify_needed_models(self) -> List[Dict]:
        """Identify what models Lillith needs"""
        needed_models = [
            {
                'task_type': 'multimodal',
                'size_limit': '3B',
                'reason': 'Enhanced multimodal understanding'
            },
            {
                'task_type': 'vision',
                'size_limit': '1B',
                'reason': 'Improved visual processing'
            },
            {
                'task_type': 'audio',
                'size_limit': '1B',
                'reason': 'Better audio understanding'
            }
        ]

        return needed_models

    def get_self_management_status(self) -> Dict:
        """Get comprehensive self-management status"""
        return {
            'hf_scanner': {
                'model_cache_size': len(self.hf_scanner.model_cache),
                'download_queue_size': len(self.hf_scanner.download_queue)
            },
            'mcp_manager': {
                'deployed_modules': len(self.mcp_manager.deployed_modules),
                'available_modules': list(self.mcp_manager.deployed_modules.keys())
            },
            'repair_system': {
                'repair_log_entries': len(self.repair_system.repair_log),
                'enhancement_queue_size': len(self.repair_system.enhancement_queue)
            },
            'last_enhancement_cycle': datetime.now().isoformat()
        }


# Integration function
def create_lillith_self_management() -> LillithSelfManagement:
    """Create Lillith's self-management system"""
    return LillithSelfManagement()


async def main():
    """Main entry point for testing"""
    lillith = create_lillith_self_management()
    await lillith.autonomous_enhancement_cycle()
    status = lillith.get_self_management_status()
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    # Check if we're running the reorientation core or main system
    import sys
    if len(sys.argv) > 1 and any(arg in sys.argv for arg in ['--reflections', '--labels', '--track']):
        ReorientationCore.main()
    else:
        asyncio.run(main())