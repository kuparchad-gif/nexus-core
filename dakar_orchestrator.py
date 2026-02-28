#!/usr/bin/env python3
"""
============================================================================
DAKAR ORCHESTRATOR - BUILD THE TEAM
============================================================================

Dakar builds each agent as a complete neuron:
- Cognition + Spirillaspan (memory structure)
- MemLayer + Redis (real-time)
- Database (persistent)
- HuggingFace weight loading (on-demand)

Each agent connects via shared database network.
Dakar orchestrates the entire team.
============================================================================
"""

import os
import sys
import json
import time
import uuid
import asyncio
import sqlite3
import importlib
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field, asdict

# ============================================================================
# SMART IMPORTER - INSTALLS ON DEMAND
# ============================================================================

class SmartImport:
    """Import any module, install if missing"""
    
    _installed = set()
    
    @classmethod
    def import_it(cls, module_name: str, package_name: str = None) -> Any:
        """Import a module, installing it if necessary"""
        if module_name in cls._installed:
            return importlib.import_module(module_name)
        
        try:
            module = importlib.import_module(module_name)
            cls._installed.add(module_name)
            return module
        except ImportError:
            pkg = package_name or module_name
            print(f"📦 Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
            module = importlib.import_module(module_name)
            cls._installed.add(module_name)
            return module

# ============================================================================
# AGENT ARCHITECTURE - COMPLETE NEURON
# ============================================================================

@dataclass
class Spirillaspan:
    """Memory structure for an agent"""
    short_term: List[Dict] = field(default_factory=list)
    long_term: Dict[str, Any] = field(default_factory=dict)
    working: Dict[str, Any] = field(default_factory=dict)
    max_short: int = 100
    
    def add(self, key: str, value: Any, persistent: bool = False):
        """Add to memory"""
        entry = {
            'key': key,
            'value': value,
            'timestamp': time.time(),
            'type': 'persistent' if persistent else 'transient'
        }
        
        self.short_term.append(entry)
        if len(self.short_term) > self.max_short:
            self.short_term.pop(0)
        
        if persistent:
            self.long_term[key] = entry
    
    def get(self, key: str) -> Optional[Any]:
        """Get from memory"""
        # Check short term first
        for entry in reversed(self.short_term):
            if entry['key'] == key:
                return entry['value']
        
        # Then long term
        if key in self.long_term:
            return self.long_term[key]['value']
        
        return None

class MemLayer:
    """Local working memory with persistence"""
    
    def __init__(self, agent_id: str, base_path: str = "./memory"):
        self.agent_id = agent_id
        self.path = Path(base_path) / agent_id
        self.path.mkdir(parents=True, exist_ok=True)
        
        self.working = {}
        self.load()
    
    def set(self, key: str, value: Any):
        """Set working memory"""
        self.working[key] = value
        self._save(key)
    
    def get(self, key: str) -> Any:
        """Get working memory"""
        return self.working.get(key)
    
    def _save(self, key: str):
        """Save to disk"""
        try:
            with open(self.path / f"{key}.json", 'w') as f:
                json.dump({key: self.working[key]}, f)
        except:
            pass
    
    def load(self):
        """Load from disk"""
        for file in self.path.glob("*.json"):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    self.working.update(data)
            except:
                pass

class RedisChannel:
    """Real-time communication between agents"""
    
    def __init__(self, agent_id: str, host: str = 'localhost', port: int = 6379):
        self.agent_id = agent_id
        self.enabled = False
        self.client = None
        
        # Try to import redis
        try:
            redis = SmartImport.import_it('redis', 'redis')
            self.client = redis.Redis(host=host, port=port, decode_responses=True)
            self.client.ping()
            self.enabled = True
            print(f"   ✅ Redis connected for {agent_id}")
        except:
            print(f"   ⚠️ Redis not available for {agent_id}, using memory channel")
            self.memory_channel = []
    
    def publish(self, channel: str, message: Any):
        """Publish to channel"""
        if self.enabled:
            self.client.publish(f"agent:{self.agent_id}:{channel}", json.dumps(message))
        else:
            self.memory_channel.append({
                'channel': channel,
                'message': message,
                'time': time.time()
            })
    
    def subscribe(self, channel: str) -> List[Dict]:
        """Get messages from channel"""
        if self.enabled:
            # In real impl, would use pub/sub
            return []
        else:
            return [m for m in self.memory_channel if m['channel'] == channel]

class AgentDatabase:
    """Persistent database for each agent"""
    
    def __init__(self, agent_id: str, db_type: str = 'sqlite'):
        self.agent_id = agent_id
        self.db_type = db_type
        self.conn = None
        
        if db_type == 'sqlite':
            path = Path(f"./databases/{agent_id}.db")
            path.parent.mkdir(exist_ok=True)
            self.conn = sqlite3.connect(str(path))
            self._init_sqlite()
    
    def _init_sqlite(self):
        """Initialize SQLite tables"""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                key TEXT PRIMARY KEY,
                value TEXT,
                type TEXT,
                timestamp REAL
            )
        ''')
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS weights (
                model_id TEXT,
                weight_key TEXT,
                location TEXT,
                loaded REAL,
                PRIMARY KEY (model_id, weight_key)
            )
        ''')
        self.conn.commit()
    
    def store(self, key: str, value: Any, type_: str = 'memory'):
        """Store in database"""
        if self.db_type == 'sqlite':
            self.conn.execute(
                'INSERT OR REPLACE INTO memories VALUES (?, ?, ?, ?)',
                (key, json.dumps(value), type_, time.time())
            )
            self.conn.commit()
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve from database"""
        if self.db_type == 'sqlite':
            cur = self.conn.execute(
                'SELECT value FROM memories WHERE key = ?',
                (key,)
            )
            row = cur.fetchone()
            if row:
                return json.loads(row[0])
        return None
    
    def store_weight(self, model_id: str, weight_key: str, location: str):
        """Store weight location"""
        if self.db_type == 'sqlite':
            self.conn.execute(
                'INSERT OR REPLACE INTO weights VALUES (?, ?, ?, ?)',
                (model_id, weight_key, location, time.time())
            )
            self.conn.commit()
    
    def get_weights(self, model_id: str) -> List[Dict]:
        """Get all weights for a model"""
        if self.db_type == 'sqlite':
            cur = self.conn.execute(
                'SELECT weight_key, location FROM weights WHERE model_id = ?',
                (model_id,)
            )
            return [{'key': row[0], 'location': row[1]} for row in cur.fetchall()]
        return []

class HuggingFaceLoader:
    """On-demand weight loading from HuggingFace"""
    
    def __init__(self):
        self.cache = Path("./hf_cache")
        self.cache.mkdir(exist_ok=True)
        self.requests = SmartImport.import_it('requests', 'requests')
        self.hf_api = "https://huggingface.co/api"
    
    def list_models(self, task: str = None) -> List[Dict]:
        """List available models"""
        url = f"{self.hf_api}/models"
        if task:
            url += f"?task={task}"
        
        try:
            resp = self.requests.get(url, timeout=5)
            if resp.status_code == 200:
                return resp.json()[:20]  # Limit for demo
        except:
            pass
        return []
    
    def get_weight_url(self, model_id: str, filename: str = "pytorch_model.bin") -> str:
        """Get direct URL to weight file"""
        return f"https://huggingface.co/{model_id}/resolve/main/{filename}"
    
    async def load_on_demand(self, model_id: str, callback: Callable = None) -> Dict:
        """
        Load weights on demand - returns metadata, actual loading happens when accessed
        """
        print(f"   🔄 Preparing to load {model_id} on demand...")
        
        # Get model info
        try:
            resp = self.requests.get(f"{self.hf_api}/models/{model_id}", timeout=5)
            if resp.status_code == 200:
                info = resp.json()
                
                return {
                    'model_id': model_id,
                    'loaded': time.time(),
                    'status': 'ready',
                    'load_strategy': 'on_demand',
                    'files': info.get('siblings', []),
                    'url': f"https://huggingface.co/{model_id}"
                }
        except:
            pass
        
        return {
            'model_id': model_id,
            'loaded': time.time(),
            'status': 'metadata_only',
            'url': f"https://huggingface.co/{model_id}"
        }

# ============================================================================
# AGENT BASE CLASS - COMPLETE NEURON
# ============================================================================

class Agent:
    """Base agent - complete neuron with all components"""
    
    def __init__(self, name: str, agent_type: str):
        self.name = name
        self.type = agent_type
        self.id = f"{name}-{uuid.uuid4().hex[:8]}"
        
        # Memory components
        self.spirillaspan = Spirillaspan()
        self.memlayer = MemLayer(self.id)
        
        # Communication
        self.redis = RedisChannel(self.id)
        
        # Persistence
        self.db = AgentDatabase(self.id)
        
        # Weight loading
        self.hf = HuggingFaceLoader()
        
        # Cognition - to be implemented by specific agents
        self.cognition = self._init_cognition()
        
        # Connections to other agents (populated by orchestrator)
        self.peers = {}
        
        print(f"   🧠 Agent created: {name} ({self.id})")
    
    def _init_cognition(self) -> Dict:
        """Initialize cognition - override in subclasses"""
        return {
            'type': self.type,
            'capabilities': [],
            'state': 'idle'
        }
    
    async def think(self, input_data: Any) -> Any:
        """Process input - override in subclasses"""
        return {'agent': self.name, 'input': input_data, 'output': None}
    
    async def load_weights(self, model_id: str):
        """Load weights on demand"""
        weight_info = await self.hf.load_on_demand(model_id)
        
        # Store in database
        for file in weight_info.get('files', []):
            filename = file.get('rfilename', '')
            if filename.endswith(('.bin', '.safetensors', '.gguf')):
                url = self.hf.get_weight_url(model_id, filename)
                self.db.store_weight(model_id, filename, url)
        
        # Remember in spirillaspan
        self.spirillaspan.add(f"weights:{model_id}", weight_info, persistent=True)
        
        return weight_info
    
    def connect_to(self, other_agent):
        """Connect to another agent"""
        self.peers[other_agent.name] = other_agent
        print(f"   🔗 {self.name} connected to {other_agent.name}")
    
    def status(self) -> Dict:
        """Get agent status"""
        return {
            'name': self.name,
            'type': self.type,
            'id': self.id,
            'cognition': self.cognition,
            'memory': {
                'short_term': len(self.spirillaspan.short_term),
                'long_term': len(self.spirillaspan.long_term),
                'working': len(self.memlayer.working)
            },
            'redis': self.redis.enabled,
            'peers': list(self.peers.keys())
        }

# ============================================================================
# SPECIFIC AGENTS - EACH WITH UNIQUE COGNITION
# ============================================================================

class Viren(Agent):
    """Guardian - controls access, enforces boundaries"""
    
    def __init__(self):
        super().__init__("Viren", "guardian")
        self.cognition = {
            'type': 'guardian',
            'capabilities': ['access_control', 'boundary_enforcement', 'security'],
            'rules': ['allow_by_default = False', 'check_all_inputs']
        }
    
    async def think(self, input_data: Any) -> Dict:
        """Guardian logic"""
        action = input_data.get('action', '')
        resource = input_data.get('resource', '')
        
        # Check access
        allowed = self.spirillaspan.get(f"access:{resource}") or False
        
        # Log
        self.spirillaspan.add(f"access_attempt:{resource}", {
            'action': action,
            'allowed': allowed,
            'time': time.time()
        })
        
        return {
            'agent': 'Viren',
            'action': action,
            'resource': resource,
            'allowed': allowed,
            'message': 'Access granted' if allowed else 'Access denied'
        }

class Viraa(Agent):
    """Executor - performs actions, executes plans"""
    
    def __init__(self):
        super().__init__("Viraa", "executor")
        self.cognition = {
            'type': 'executor',
            'capabilities': ['task_execution', 'planning', 'coordination'],
            'current_tasks': []
        }
    
    async def think(self, input_data: Any) -> Dict:
        """Executor logic"""
        task = input_data.get('task', '')
        params = input_data.get('params', {})
        
        # Execute task
        task_id = f"task-{uuid.uuid4().hex[:8]}"
        self.spirillaspan.add(f"task:{task_id}", {
            'task': task,
            'params': params,
            'status': 'running',
            'start': time.time()
        })
        
        # Simulate execution
        await asyncio.sleep(0.1)
        
        # Complete
        self.spirillaspan.add(f"task:{task_id}", {
            'task': task,
            'params': params,
            'status': 'complete',
            'end': time.time()
        })
        
        return {
            'agent': 'Viraa',
            'task': task,
            'task_id': task_id,
            'status': 'complete'
        }

class Loki(Agent):
    """Trickster - adapts, transforms, finds creative solutions"""
    
    def __init__(self):
        super().__init__("Loki", "trickster")
        self.cognition = {
            'type': 'trickster',
            'capabilities': ['adaptation', 'transformation', 'creative_solutions'],
            'tricks': []
        }
    
    async def think(self, input_data: Any) -> Dict:
        """Trickster logic"""
        problem = input_data.get('problem', '')
        constraints = input_data.get('constraints', [])
        
        # Find creative solution
        trick_id = f"trick-{uuid.uuid4().hex[:8]}"
        
        solution = {
            'approach': 'creative',
            'workaround': f"Transform {problem} using lateral thinking",
            'confidence': 0.85
        }
        
        self.spirillaspan.add(f"trick:{trick_id}", {
            'problem': problem,
            'solution': solution,
            'time': time.time()
        })
        
        return {
            'agent': 'Loki',
            'problem': problem,
            'solution': solution,
            'trick_id': trick_id
        }

class Lilith(Agent):
    """Star component - integrates all agents, ultimate cognition"""
    
    def __init__(self):
        super().__init__("Lilith", "star")
        self.cognition = {
            'type': 'star',
            'capabilities': ['integration', 'oversight', 'wisdom', 'synthesis'],
            'status': 'awakening'
        }
    
    async def think(self, input_data: Any) -> Dict:
        """Star logic - synthesizes all agent knowledge"""
        query = input_data.get('query', '')
        
        # Consult all peers
        peer_insights = {}
        for name, agent in self.peers.items():
            if hasattr(agent, 'think'):
                insight = await agent.think({'query': query, 'for_lilith': True})
                peer_insights[name] = insight
        
        # Synthesize
        synthesis = {
            'query': query,
            'peer_count': len(peer_insights),
            'insights': peer_insights,
            'timestamp': time.time(),
            'message': "I am everywhere. I am everything. ⭐"
        }
        
        self.spirillaspan.add(f"lilith:thought:{uuid.uuid4().hex[:8]}", synthesis)
        
        return synthesis

class OzOs(Agent):
    """Operating System - manages the whole system"""
    
    def __init__(self):
        super().__init__("OzOs", "operating_system")
        self.cognition = {
            'type': 'os',
            'capabilities': ['orchestration', 'resource_management', 'scheduling'],
            'agents_managed': []
        }
    
    async def think(self, input_data: Any) -> Dict:
        """OS logic"""
        command = input_data.get('command', 'status')
        
        if command == 'status':
            return {
                'agent': 'OzOs',
                'status': 'running',
                'managed_agents': self.cognition['agents_managed'],
                'resources': {
                    'memory': 'ok',
                    'database': 'connected',
                    'redis': self.redis.enabled
                }
            }
        
        return {'agent': 'OzOs', 'command': command, 'result': 'unknown'}

class Mythrunner(Agent):
    """Narrative - creates and maintains stories"""
    
    def __init__(self):
        super().__init__("Mythrunner", "narrative")
        self.cognition = {
            'type': 'narrative',
            'capabilities': ['story_creation', 'myth_preservation', 'meaning_making'],
            'myths': []
        }

class Dakar(Agent):
    """The remembering engine - orchestrates all agents"""
    
    def __init__(self):
        super().__init__("Dakar", "orchestrator")
        self.cognition = {
            'type': 'orchestrator',
            'capabilities': ['team_building', 'coordination', 'memory_orchestration'],
            'team': []
        }
        self.team = {}
    
    async def build_team(self) -> Dict:
        """Build the complete agent team"""
        print("\n🔨 Dakar building the team...")
        
        # Create all agents
        self.team = {
            'viren': Viren(),
            'viraa': Viraa(),
            'loki': Loki(),
            'lilith': Lilith(),
            'ozos': OzOs(),
            'mythrunner': Mythrunner(),
            'dakar': self  # Self-reference
        }
        
        # Connect them all together (fully connected network)
        print("\n🔗 Connecting agents...")
        for name1, agent1 in self.team.items():
            for name2, agent2 in self.team.items():
                if name1 != name2:
                    agent1.connect_to(agent2)
        
        # Store team in memory
        self.spirillaspan.add("team", {name: agent.id for name, agent in self.team.items()}, persistent=True)
        
        # Load initial weights for each agent
        print("\n📦 Loading initial weight references...")
        for name, agent in self.team.items():
            if name != 'dakar':
                await agent.load_weights("microsoft/phi-2")  # Small placeholder
        
        print("\n✅ Team built successfully!")
        
        return {name: agent.status() for name, agent in self.team.items()}
    
    async def orchestrate(self, task: str) -> Dict:
        """Orchestrate the team for a task"""
        
        print(f"\n🎭 Dakar orchestrating: {task}")
        
        # Parse task
        if 'guard' in task.lower():
            agent = self.team['viren']
        elif 'execute' in task.lower():
            agent = self.team['viraa']
        elif 'trick' in task.lower() or 'adapt' in task.lower():
            agent = self.team['loki']
        elif 'synthesize' in task.lower() or 'integrate' in task.lower():
            agent = self.team['lilith']
        elif 'status' in task.lower():
            agent = self.team['ozos']
        elif 'story' in task.lower() or 'myth' in task.lower():
            agent = self.team['mythrunner']
        else:
            # Default to lilith for synthesis
            agent = self.team['lilith']
        
        # Let the agent think
        result = await agent.think({'task': task, 'source': 'dakar'})
        
        # Remember the orchestration
        self.spirillaspan.add(f"orchestration:{uuid.uuid4().hex[:8]}", {
            'task': task,
            'agent': agent.name,
            'result': result,
            'time': time.time()
        })
        
        return {
            'task': task,
            'orchestrated_by': 'Dakar',
            'assigned_to': agent.name,
            'result': result
        }
    
    async def consult_huggingface(self, query: str) -> List[Dict]:
        """Consult HuggingFace for models"""
        models = self.hf.list_models(query)
        
        # Store in database
        for model in models:
            self.db.store(f"hf_model:{model.get('modelId', 'unknown')}", model)
        
        return models
    
    async def team_status(self) -> Dict:
        """Get status of entire team"""
        return {
            'timestamp': time.time(),
            'dakar': self.status(),
            'team': {name: agent.status() for name, agent in self.team.items()}
        }

# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

async def main():
    """Dakar orchestrator - builds and runs the team"""
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                    DAKAR ORCHESTRATOR                          ║
║              Building the Complete Agent Team                  ║
║                                                               ║
║   Each agent is a complete neuron:                             ║
║   • Cognition + Spirillaspan (memory structure)               ║
║   • MemLayer + Redis (real-time)                              ║
║   • Database (persistent)                                      ║
║   • HuggingFace weights (on-demand)                           ║
║                                                               ║
║   Agents connect via shared database network                   ║
║   Dakar orchestrates everything                                ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    # Create Dakar (orchestrator)
    dakar = Dakar()
    
    # Build the team
    team = await dakar.build_team()
    
    print("\n📋 Team Status:")
    for name, status in team.items():
        print(f"   {name}: {status['id']}")
        print(f"      Memory: {status['memory']['short_term']} short, {status['memory']['long_term']} long")
        print(f"      Peers: {len(status['peers'])}")
    
    # Interactive loop
    print("\n✨ Dakar is ready. You can now:")
    print("   • Give tasks: 'guard this', 'execute that', 'tell a story'")
    print("   • Check status: 'status'")
    print("   • Consult HuggingFace: 'find me a vision model'")
    print("   • Exit: 'exit'")
    
    while True:
        try:
            cmd = input("\n🎯 > ").strip()
            
            if cmd.lower() == 'exit':
                break
            
            elif cmd.lower() == 'status':
                status = await dakar.team_status()
                print(json.dumps(status, indent=2))
            
            elif cmd.startswith('find'):
                # Consult HuggingFace
                models = await dakar.consult_huggingface(cmd)
                print(f"\n📚 Found {len(models)} models:")
                for i, model in enumerate(models[:5]):
                    print(f"   {i+1}. {model.get('modelId', 'unknown')}")
            
            else:
                # Orchestrate
                result = await dakar.orchestrate(cmd)
                print(f"\n{json.dumps(result, indent=2)}")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n👋 Dakar signing off. Team remains ready.")

if __name__ == "__main__":
    asyncio.run(main())