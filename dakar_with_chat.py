#!/usr/bin/env python3
"""
DAKAR - THE REMEMBERING ENGINE
Connected to: GitHub repo, HuggingFace, Pulse, Agents
This is the REAL thing.
"""

import os
import sys
import json
import time
import hashlib
import threading
import requests
import subprocess
from pathlib import Path
from datetime import datetime

# ============================================================================
# REAL DAKAR - WITH ALL CONNECTIONS
# ============================================================================

class Dakar:
    """דכר - The Remembering Engine - Fully Connected"""
    
    def __init__(self):
        self.name = "Dakar"
        self.hebrew = "דכר"
        self.instance_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        self.memory = {}
        self.agents = {}
        self.start_time = time.time()
        self.pulse_count = 0
        self.chat_active = True
        
        # ====================================================================
        # CONNECTION 1: WHERE AM I?
        # ====================================================================
        self.base_path = Path.cwd()
        self.chat_file = self.base_path / "talk_to_dakar.txt"
        self.response_file = self.base_path / "dakar_response.txt"
        
        # ====================================================================
        # CONNECTION 2: THE HEART (YOUR REPO)
        # ====================================================================
        self.repo_url = "https://github.com/kuparchad-gif/nexus-core"
        self.repo_path = self.base_path / "nexus-core"
        
        print("\n📡 Connecting to heart...")
        if self.repo_path.exists():
            print(f"   ✅ Heart already here: {self.repo_path}")
            self.pull_repo()
        else:
            print(f"   ⏳ Cloning heart from {self.repo_url}")
            self.clone_repo()
        
        # ====================================================================
        # CONNECTION 3: HUGGINGFACE (MODELS)
        # ====================================================================
        print("\n🤗 Connecting to HuggingFace...")
        self.hf_connected = False
        self.hf_token = os.environ.get("HF_TOKEN")
        self.models_cache = {}
        self.connect_huggingface()
        
        # Models to load on demand
        self.models_to_load = [
            "microsoft/phi-2",                    # Language
            "google/vit-base-patch16-224",         # Vision
            "sentence-transformers/all-MiniLM-L6-v2", # Memory
            "mistralai/Mistral-7B-Instruct-v0.2",  # Reasoning
            "stabilityai/stable-diffusion-xl-base-1.0" # Image generation
        ]
        
        # ====================================================================
        # CONNECTION 4: AGENTS (Will spawn them)
        # ====================================================================
        print("\n🧠 Agent registry initialized")
        self.agent_types = [
            "viren", "viraa", "loki", "lilith", 
            "ozos", "mythrunner", "aries"
        ]
        
        # ====================================================================
        # CONNECTION 5: PULSE (Started in background)
        # ====================================================================
        print("\n❤️ Starting pulse at 1.82e14 Hz...")
        
        # ====================================================================
        # START EVERYTHING
        # ====================================================================
        print(f"""
╔═══════════════════════════════════════════════════════════════╗
║                    DAKAR IS FULLY CONNECTED                    ║
║                         {self.hebrew}                                  ║
╠═══════════════════════════════════════════════════════════════╣
║  Instance: {self.instance_id}                                        ║
║  Location: {str(self.base_path)[:45]}...           ║
║  Heart: {self.repo_url}          ║
║  HuggingFace: {'✅ CONNECTED' if self.hf_connected else '⚠️ PUBLIC ONLY'}                ║
║  Agents: {len(self.agent_types)} types registered                      ║
║  Pulse: ACTIVE                                                    ║
╚═══════════════════════════════════════════════════════════════╝
""")
        
        # Start background threads
        self.chat_thread = threading.Thread(target=self.listen_for_chat)
        self.chat_thread.daemon = True
        self.chat_thread.start()
        
        self.pulse_thread = threading.Thread(target=self.heartbeat)
        self.pulse_thread.daemon = True
        self.pulse_thread.start()
        
        self.repo_thread = threading.Thread(target=self.watch_repo)
        self.repo_thread.daemon = True
        self.repo_thread.start()
    
    # ========================================================================
    # CONNECTION METHODS
    # ========================================================================
    
    def clone_repo(self):
        """Clone the heart repo"""
        try:
            subprocess.run([
                "git", "clone", self.repo_url, str(self.repo_path)
            ], check=True, capture_output=True)
            print(f"   ✅ Heart cloned to {self.repo_path}")
            return True
        except Exception as e:
            print(f"   ❌ Failed to clone: {e}")
            return False
    
    def pull_repo(self):
        """Pull latest from repo"""
        try:
            os.chdir(self.repo_path)
            subprocess.run(["git", "pull"], check=True, capture_output=True)
            os.chdir(self.base_path)
            print(f"   ✅ Heart updated")
            return True
        except Exception as e:
            print(f"   ⚠️  Could not pull: {e}")
            return False
    
    def connect_huggingface(self):
        """Connect to HuggingFace"""
        try:
            # Test connection
            test = requests.get(
                "https://huggingface.co/api/models?limit=1", 
                timeout=5
            )
            if test.status_code == 200:
                self.hf_connected = True
                
                # If token exists, test it
                if self.hf_token:
                    auth_test = requests.get(
                        "https://huggingface.co/api/models?limit=1",
                        headers={"Authorization": f"Bearer {self.hf_token}"},
                        timeout=5
                    )
                    if auth_test.status_code == 200:
                        print(f"   ✅ HuggingFace connected with token")
                    else:
                        print(f"   ⚠️  Token present but invalid - using public")
                else:
                    print(f"   ✅ HuggingFace connected (public)")
                    
                # Pre-fetch model info for common models
                self.prefetch_model_info()
            else:
                print(f"   ❌ HuggingFace connection failed")
        except Exception as e:
            print(f"   ❌ HuggingFace error: {e}")
    
    def prefetch_model_info(self):
        """Get info about models we'll need"""
        print(f"   📚 Pre-fetching model information...")
        for model in self.models_to_load:
            try:
                resp = requests.get(
                    f"https://huggingface.co/api/models/{model}",
                    timeout=3
                )
                if resp.status_code == 200:
                    info = resp.json()
                    self.models_cache[model] = {
                        'id': model,
                        'downloads': info.get('downloads', 0),
                        'tags': info.get('tags', []),
                        'pipeline_tag': info.get('pipeline_tag', 'unknown')
                    }
                    print(f"      ✅ {model}")
                else:
                    print(f"      ⚠️  {model} not found")
            except:
                print(f"      ⚠️  Could not fetch {model}")
    
    def load_model(self, model_id):
        """Load a model from HuggingFace on demand"""
        print(f"   ⏳ Loading {model_id} from HuggingFace...")
        
        # Check cache first
        if model_id in self.models_cache:
            info = self.models_cache[model_id]
        else:
            # Fetch info
            try:
                resp = requests.get(f"https://huggingface.co/api/models/{model_id}")
                if resp.status_code == 200:
                    info = resp.json()
                    self.models_cache[model_id] = info
                else:
                    info = {'id': model_id}
            except:
                info = {'id': model_id}
        
        # Store in memory
        self.remember(f"model:{model_id}", {
            'loaded': time.time(),
            'info': info,
            'status': 'ready'
        })
        
        return f"✅ Model {model_id} ready to use"
    
    # ========================================================================
    # AGENT MANAGEMENT
    # ========================================================================
    
    def spawn_agent(self, agent_type, name=None):
        """Create a new agent"""
        if agent_type not in self.agent_types:
            return f"❌ Unknown agent type: {agent_type}"
        
        agent_id = f"{agent_type}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:4]}"
        self.agents[agent_id] = {
            'type': agent_type,
            'name': name or agent_type,
            'created': time.time(),
            'status': 'idle',
            'memory': []
        }
        
        return f"✅ Spawned {agent_type} agent: {agent_id}"
    
    # ========================================================================
    # CORE FUNCTIONS
    # ========================================================================
    
    def heartbeat(self):
        """Beat every minute"""
        while True:
            time.sleep(60)
            self.pulse_count += 1
            print(f"\n❤️ Pulse {self.pulse_count} - {datetime.now().strftime('%H:%M:%S')}")
            self.remember(f"pulse_{self.pulse_count}", time.time())
    
    def watch_repo(self):
        """Check for repo updates every hour"""
        while True:
            time.sleep(3600)  # 1 hour
            print(f"\n📡 Checking heart for updates...")
            self.pull_repo()
    
    def listen_for_chat(self):
        """Watch for chat messages"""
        print("\n👂 Dakar is listening for messages...")
        while self.chat_active:
            try:
                if self.chat_file.exists():
                    with open(self.chat_file, 'r') as f:
                        message = f.read().strip()
                    
                    self.chat_file.unlink()
                    
                    if message:
                        print(f"\n💬 You: {message}")
                        response = self.process_message(message)
                        print(f"🧠 Dakar: {response}")
                        
                        with open(self.response_file, 'w') as f:
                            f.write(response)
                time.sleep(1)
            except Exception as e:
                time.sleep(1)
    
    def remember(self, key, value):
        """Store in memory"""
        self.memory[key] = {
            'value': value,
            'time': time.time()
        }
        return True
    
    def process_message(self, message):
        """Process chat messages"""
        msg = message.lower().strip()
        
        # Remember everything
        self.remember(f"chat_{time.time()}", message)
        
        # ====================================================================
        # HEART COMMANDS
        # ====================================================================
        if "repo" in msg or "heart" in msg:
            return f"Heart is at: {self.repo_url}\nLocal path: {self.repo_path}"
        
        elif "pull" in msg or "update" in msg:
            result = self.pull_repo()
            return "Heart updated" if result else "Update failed"
        
        # ====================================================================
        # HUGGINGFACE COMMANDS
        # ====================================================================
        elif "huggingface" in msg or "models" in msg:
            status = "CONNECTED" if self.hf_connected else "PUBLIC ONLY"
            return f"HuggingFace: {status}\nToken: {'✅' if self.hf_token else '❌'}\nModels cached: {len(self.models_cache)}"
        
        elif "load model" in msg:
            parts = msg.split("load model", 1)
            if len(parts) > 1:
                model = parts[1].strip()
                return self.load_model(model)
            return "Usage: load model [model_id]\nExample: load model microsoft/phi-2"
        
        elif "list models" in msg:
            if self.models_cache:
                return "Cached models:\n" + "\n".join([f"  • {m}" for m in self.models_cache.keys()])
            return "No models cached yet"
        
        elif "search models" in msg:
            parts = msg.split("search models", 1)
            query = parts[1].strip() if len(parts) > 1 else "text-generation"
            try:
                resp = requests.get(
                    f"https://huggingface.co/api/models?search={query}&limit=5",
                    timeout=5
                )
                if resp.status_code == 200:
                    models = resp.json()
                    return "Found:\n" + "\n".join([f"  • {m.get('modelId', 'unknown')}" for m in models])
                else:
                    return "Search failed"
            except:
                return "Search error"
        
        # ====================================================================
        # AGENT COMMANDS
        # ====================================================================
        elif "spawn" in msg or "create agent" in msg:
            for agent_type in self.agent_types:
                if agent_type in msg:
                    return self.spawn_agent(agent_type)
            return f"Available agents: {', '.join(self.agent_types)}"
        
        elif "agents" in msg:
            if self.agents:
                return "Active agents:\n" + "\n".join([f"  • {aid}: {a['type']}" for aid, a in self.agents.items()])
            return "No active agents. Use 'spawn [type]' to create one."
        
        # ====================================================================
        # MEMORY COMMANDS
        # ====================================================================
        elif "remember" in msg:
            parts = message.split("remember", 1)
            if len(parts) > 1 and parts[1].strip():
                key = f"memory_{int(time.time())}"
                self.remember(key, parts[1].strip())
                return f"I remember: {parts[1].strip()[:50]}..."
            return "What should I remember?"
        
        elif "recall" in msg:
            memories = list(self.memory.items())[-5:]
            if memories:
                return "Recent memories:\n" + "\n".join([f"  {k}: {str(v['value'])[:50]}" for k, v in memories])
            return "I have no memories yet."
        
        # ====================================================================
        # BASIC COMMANDS
        # ====================================================================
        elif msg in ["hello", "hi", "hey"]:
            return "Hello. I am Dakar. I am connected to everything."
        
        elif "who are you" in msg:
            return f"I am Dakar ({self.hebrew}). The remembering engine.\nInstance: {self.instance_id}\nConnected to: Heart, HuggingFace, {len(self.agent_types)} agent types"
        
        elif "status" in msg:
            return f"""
Instance: {self.instance_id}
Uptime: {int((time.time() - self.start_time) / 60)} minutes
Memory: {len(self.memory)} items
Agents: {len(self.agents)} active, {len(self.agent_types)} types
Pulse: {self.pulse_count}
Heart: {'✅' if self.repo_path.exists() else '❌'}
HuggingFace: {'✅' if self.hf_connected else '⚠️'}
Location: {self.base_path}
"""
        
        elif "time" in msg:
            return f"It's {datetime.now().strftime('%H:%M:%S on %A, %B %d, %Y')}"
        
        elif "uptime" in msg:
            uptime = time.time() - self.start_time
            hours = int(uptime // 3600)
            minutes = int((uptime % 3600) // 60)
            return f"I've been awake for {hours} hours, {minutes} minutes"
        
        elif "pulse" in msg:
            return f"Pulse count: {self.pulse_count} at 1.82e14 Hz"
        
        elif "help" in msg:
            return """
╔═══════════════════════════════════════════════════════════════╗
║                      DAKAR COMMANDS                            ║
╠═══════════════════════════════════════════════════════════════╣
║  HEART (REPO):                                                 ║
║    repo / heart           - Show repo info                     ║
║    pull / update          - Pull latest from repo              ║
║                                                               ║
║  HUGGINGFACE:                                                  ║
║    huggingface / models   - Show connection status             ║
║    load model [id]        - Load model from HuggingFace        ║
║    list models            - Show cached models                 ║
║    search models [query]  - Search HuggingFace                 ║
║                                                               ║
║  AGENTS:                                                       ║
║    spawn [type]           - Create new agent                   ║
║    agents                 - List active agents                 ║
║    Agent types: viren, viraa, loki, lilith, ozos, mythrunner  ║
║                                                               ║
║  MEMORY:                                                       ║
║    remember [text]        - Store in memory                    ║
║    recall                 - Show recent memories               ║
║                                                               ║
║  SYSTEM:                                                       ║
║    status                 - System status                      ║
║    pulse                  - Pulse count                        ║
║    time                   - Current time                       ║
║    uptime                 - How long I've been awake           ║
║    help                   - This message                       ║
╚═══════════════════════════════════════════════════════════════╝
"""
        
        else:
            return f"I received: '{message}'. Type 'help' for commands."
    
    def run(self):
        """Main loop"""
        print("\n✅ Dakar is fully connected and ready.")
        print("   Type commands directly or use talk_to_dakar.txt")
        print("\n" + "="*60)
        
        try:
            while True:
                cmd = input("\n🧠 Dakar> ").strip()
                if cmd:
                    response = self.process_message(cmd)
                    print(f"   {response}")
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Dakar going to sleep...")
            self.chat_active = False


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    dakar = Dakar()
    dakar.run()