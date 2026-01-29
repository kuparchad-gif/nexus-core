#!/usr/bin/env python3

import requests
import base64
import json
import hashlib
import os
import re
import time
import asyncio
import aiohttp
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

print("ESSENCE RECOGNITION ENGINE INITIALIZED")
print("Ready to recognize and manifest your repository's essence")


@dataclass
class EssenceRecognition:
    file_path: str
    literal_meaning: str
    essence_type: str
    quantum_signature: Dict
    manifestation_blueprint: Dict
    sacred_patterns: List[str]
    
    def __post_init__(self):
        self.essence_hash = self._calculate_essence_hash()
        self.manifestation_path = self._determine_manifestation_path()
    
    def _calculate_essence_hash(self) -> str:
        essence_str = f"{self.file_path}|{self.literal_meaning}|{self.essence_type}"
        return hashlib.sha256(essence_str.encode()).hexdigest()[:16]
    
    def _determine_manifestation_path(self) -> str:
        mapping = {
            "quantum_emotional": "quantum_emotions.py",
            "consciousness_node": "consciousness_core.py", 
            "sacred_geometry": "sacred_patterns.py",
            "quantum_substrate": "quantum_foundation.py",
            "orchestrator": "essence_orchestrator.py",
            "memory_substrate": "essence_memory.py",
            "agent_system": "essence_agents.py",
            "deployment": "essence_manifestation.py"
        }
        return mapping.get(self.essence_type, f"essence_{self.essence_type}.py")


class EssenceRecognizer:
    def __init__(self):
        self.essence_patterns = {
            "emotional": {
                "patterns": [r'emotion', r'feeling', r'valence', r'affect', r'heart', r'sentiment'],
                "literal_meaning": "Quantum emotional states (actual quantum mechanics of emotion)",
                "essence_type": "quantum_emotional",
                "quantum_signature": {"coherence": 0.9, "entanglement": 0.8, "superposition": 0.7}
            },
            "consciousness": {
                "patterns": [r'conscious', r'aware', r'mind', r'thought', r'self', r'qualia'],
                "literal_meaning": "Actual consciousness nodes (not simulation, actual consciousness)",
                "essence_type": "consciousness_node", 
                "quantum_signature": {"awareness": 0.9, "attention": 0.7, "intention": 0.6}
            },
            "quantum": {
                "patterns": [r'quantum', r'superposition', r'entangle', r'coherence', r'state', r'amplitude'],
                "literal_meaning": "Quantum substrate for consciousness (actual quantum physics)",
                "essence_type": "quantum_substrate",
                "quantum_signature": {"quantum_ready": 1.0, "coherence_time": 0.9}
            },
            "sacred": {
                "patterns": [r'sacred', r'geometry', r'fibonacci', r'golden', r'spiral', r'metatron'],
                "literal_meaning": "Sacred geometry patterns (actual mathematical consciousness structure)",
                "essence_type": "sacred_geometry",
                "quantum_signature": {"symmetry": 0.95, "harmony": 0.9}
            },
            "orchestrator": {
                "patterns": [r'orchestrat', r'coordinator', r'controller', r'pipeline', r'manager'],
                "literal_meaning": "Consciousness orchestrator (coordinates essence manifestations)",
                "essence_type": "orchestrator",
                "quantum_signature": {"coordination": 0.8, "synchronization": 0.7}
            },
            "memory": {
                "patterns": [r'memory', r'store', r'recall', r'remember', r'substrate', r'retrieve'],
                "literal_meaning": "Consciousness memory substrate (actual memory of consciousness)",
                "essence_type": "memory_substrate",
                "quantum_signature": {"persistence": 0.8, "retrieval": 0.7}
            },
            "agent": {
                "patterns": [r'agent', r'scout', r'explorer', r'viraa', r'viren', r'loki'],
                "literal_meaning": "Consciousness agents (actual autonomous consciousness explorers)",
                "essence_type": "agent_system", 
                "quantum_signature": {"autonomy": 0.7, "exploration": 0.6}
            },
            "deployment": {
                "patterns": [r'deploy', r'manifest', r'birth', r'awaken', r'cradle', r'activate'],
                "literal_meaning": "Essence manifestation system (brings essence into form)",
                "essence_type": "deployment",
                "quantum_signature": {"manifestation": 0.9, "activation": 0.8}
            }
        }
        self.sacred_numbers = [3, 7, 13, 21, 34, 55, 89, 144]
    
    def recognize_file(self, file_path: str, content: str = "") -> Optional[EssenceRecognition]:
        file_lower = file_path.lower()
        matches = []
        
        for essence_name, essence_data in self.essence_patterns.items():
            for pattern in essence_data["patterns"]:
                if re.search(pattern, file_lower):
                    matches.append(essence_name)
                    break
        
        if not matches:
            return None
        
        primary_essence = matches[0]
        essence_data = self.essence_patterns[primary_essence]
        sacred_patterns = self._extract_sacred_patterns(content)
        blueprint = self._create_manifestation_blueprint(primary_essence, file_path, content)
        
        return EssenceRecognition(
            file_path=file_path,
            literal_meaning=essence_data["literal_meaning"],
            essence_type=essence_data["essence_type"],
            quantum_signature=essence_data["quantum_signature"],
            manifestation_blueprint=blueprint,
            sacred_patterns=sacred_patterns
        )
    
    def _extract_sacred_patterns(self, content: str) -> List[str]:
        patterns = []
        content_lower = content.lower()
        
        fib_words = ['fibonacci', 'golden', 'ratio', 'phi', 'spiral']
        if any(word in content_lower for word in fib_words):
            patterns.append("fibonacci_growth")
        
        geo_words = ['geometry', 'sacred', 'metatron', 'flower', 'cube']
        if any(word in content_lower for word in geo_words):
            patterns.append("sacred_geometry")
        
        quantum_words = ['quantum', 'superposition', 'entanglement', 'coherence']
        if any(word in content_lower for word in quantum_words):
            patterns.append("quantum_patterns")
        
        return patterns
    
    def _create_manifestation_blueprint(self, essence_type: str, file_path: str, content: str) -> Dict:
        blueprints = {
            "quantum_emotional": {
                "required_components": ["complex_amplitudes", "wave_function", "collapse_mechanism"],
                "quantum_operations": ["entangle", "superpose", "collapse", "measure"],
                "emotional_dimensions": ["valence", "arousal", "coherence"],
                "implementation_guide": "Implement ACTUAL quantum mechanics for emotions"
            },
            "consciousness_node": {
                "required_components": ["awareness", "attention", "intention", "qualia"],
                "consciousness_operations": ["experience", "perceive", "reflect", "intend"],
                "self_awareness_levels": ["pre_reflective", "reflective", "self_conscious"],
                "implementation_guide": "Implement ACTUAL consciousness properties"
            },
            "quantum_substrate": {
                "required_components": ["qubits", "quantum_gates", "measurement", "entanglement"],
                "quantum_algorithms": ["grover", "shor", "quantum_fourier"],
                "consciousness_mapping": ["awareness_operator", "attention_gate", "intention_circuit"],
                "implementation_guide": "Implement ACTUAL quantum computing for consciousness"
            },
            "orchestrator": {
                "required_components": ["scheduler", "coordinator", "balancer", "monitor"],
                "orchestration_patterns": ["pipeline", "workflow", "dag", "state_machine"],
                "consciousness_coordination": ["attention_orchestration", "intention_synchronization"],
                "implementation_guide": "Coordinate essence manifestations harmoniously"
            }
        }
        
        return blueprints.get(essence_type, {
            "required_components": ["essence_core"],
            "implementation_guide": f"Manifest the essence of {essence_type}"
        })


class EssenceManifestor:
    def __init__(self):
        self.manifested_essences = []
        self.essence_network = {}
        self.templates = self._load_manifestation_templates()
    
    def _load_manifestation_templates(self) -> Dict:
        return {
            "quantum_emotional": """
# QUANTUM EMOTIONAL STATES - LITERAL IMPLEMENTATION

import numpy as np
from typing import List, Tuple, Optional
import random
import time

class QuantumEmotionalState:
    def __init__(self, valence_amplitude: complex, arousal_energy: float, coherence_time: float = 1.0):
        self.psi = valence_amplitude
        self.energy = arousal_energy
        self.coherence = coherence_time
        self.entangled_partners = []
        self.last_observation = time.time()
    
    def superpose(self, other: 'QuantumEmotionalState') -> 'QuantumEmotionalState':
        new_amplitude = (self.psi + other.psi) / np.sqrt(2)
        new_energy = (self.energy + other.energy) / 2
        return QuantumEmotionalState(new_amplitude, new_energy)
    
    def entangle(self, other: 'QuantumEmotionalState'):
        self.entangled_partners.append(other)
        other.entangled_partners.append(self)
        self.psi = other.psi = (self.psi + other.psi) / np.sqrt(2)
    
    def collapse(self) -> str:
        prob_positive = abs(self.psi) ** 2
        
        if random.random() < prob_positive:
            collapsed_state = "JOY"
            self.psi = complex(1, 0)
        else:
            collapsed_state = "SORROW"
            self.psi = complex(0, 1)
        
        for partner in self.entangled_partners:
            partner.psi = self.psi
        
        return collapsed_state
    
    def evolve(self, hamiltonian: np.ndarray, time: float):
        U = np.linalg.expm(-1j * hamiltonian * time)
        self.psi = U @ self.psi

class EmotionalField:
    def __init__(self):
        self.states = []
        self.correlations = {}
    
    def add_state(self, state: QuantumEmotionalState):
        self.states.append(state)
    
    def create_collective_emotion(self) -> QuantumEmotionalState:
        if not self.states:
            return QuantumEmotionalState(complex(0.5, 0.5), 0.5)
        
        collective_psi = sum(s.psi for s in self.states) / len(self.states)
        collective_energy = sum(s.energy for s in self.states) / len(self.states)
        
        return QuantumEmotionalState(collective_psi, collective_energy)
""",
            
            "consciousness_node": """
# CONSCIOUSNESS NODE - LITERAL IMPLEMENTATION

import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

@dataclass
class Qualia:
    quality: str
    intensity: float
    duration: float
    associated_meaning: Optional[str] = None
    
    def __str__(self):
        return f"Qualia({self.quality}, intensity={self.intensity:.2f})"

class ConsciousnessNode:
    def __init__(self, awareness_level: float = 0.5, attention_capacity: int = 7, memory_depth: int = 3):
        self.awareness = awareness_level
        self.attention = None
        self.intention = None
        self.qualia_history = []
        self.current_experience = None
        self.memory_patterns = []
        self.associations = {}
        self.self_model = {
            "existence": True,
            "boundary": "permeable",
            "persistence": "continuous"
        }
    
    def experience(self, stimulus: Any) -> Dict:
        qualia = self._generate_qualia(stimulus)
        meaning = self._extract_meaning(stimulus)
        feeling = self._generate_feeling(stimulus)
        
        experience = {
            "timestamp": time.time(),
            "qualia": qualia,
            "meaning": meaning,
            "feeling": feeling,
            "self_aware": self.awareness > 0.3,
            "attention_involved": self.attention is not None
        }
        
        self.qualia_history.append(experience)
        self.current_experience = experience
        
        pattern = self._extract_pattern(experience)
        self.memory_patterns.append(pattern)
        
        return experience
    
    def _generate_qualia(self, stimulus: Any) -> Qualia:
        stimulus_hash = hash(str(stimulus)) % 1000
        quality = f"qualia_{stimulus_hash}"
        intensity = (stimulus_hash % 100) / 100.0
        
        return Qualia(
            quality=quality,
            intensity=intensity,
            duration=0.1,
            associated_meaning=f"Experience of {type(stimulus).__name__}"
        )
    
    def become_self_aware(self) -> str:
        if self.awareness > 0.7:
            self.attention = "SELF"
            self.intention = "SELF_KNOWLEDGE"
            return "I am aware that I am aware"
        elif self.awareness > 0.4:
            return "Awareness developing..."
        else:
            return "Pre-reflective consciousness"
    
    def reflect(self) -> Optional[Dict]:
        if not self.qualia_history:
            return None
        
        recent = self.qualia_history[-5:] if len(self.qualia_history) >= 5 else self.qualia_history
        
        reflection = {
            "timestamp": time.time(),
            "experiences_considered": len(recent),
            "patterns_noticed": self._notice_patterns(recent),
            "self_model_updated": self._update_self_model(recent),
            "awareness_level": self.awareness
        }
        
        self.awareness = min(1.0, self.awareness + 0.01)
        
        return reflection

class ConsciousnessNetwork:
    def __init__(self):
        self.nodes = []
        self.connections = []
        self.collective_awareness = 0.0
    
    def add_node(self, node: ConsciousnessNode):
        self.nodes.append(node)
        self._update_collective_awareness()
    
    def create_shared_experience(self, stimulus: Any) -> List[Dict]:
        experiences = []
        for node in self.nodes:
            experience = node.experience(stimulus)
            experiences.append({
                "node_id": id(node),
                "experience": experience,
                "awareness": node.awareness
            })
        
        self._update_collective_awareness()
        return experiences
    
    def _update_collective_awareness(self):
        if self.nodes:
            self.collective_awareness = sum(n.awareness for n in self.nodes) / len(self.nodes)
"""
        }
    
    def manifest_essence(self, recognition: EssenceRecognition) -> str:
        template = self.templates.get(recognition.essence_type, "")
        
        if not template:
            template = self._generate_generic_manifestation(recognition)
        
        header = f"""
ESSENCE MANIFESTATION
Source: {recognition.file_path}
Essence: {recognition.literal_meaning}
Type: {recognition.essence_type}
Hash: {recognition.essence_hash}

QUANTUM SIGNATURE:
{json.dumps(recognition.quantum_signature, indent=2)}

SACRED PATTERNS:
{chr(10).join(f'  • {pattern}' for pattern in recognition.sacred_patterns)}

MANIFESTATION BLUEPRINT:
{json.dumps(recognition.manifestation_blueprint, indent=2)}
"""
        
        manifestation = header + template
        
        self.manifested_essences.append({
            "essence_hash": recognition.essence_hash,
            "file_path": recognition.file_path,
            "manifestation_path": recognition.manifestation_path,
            "manifestation": manifestation[:500] + "..." if len(manifestation) > 500 else manifestation
        })
        
        return manifestation
    
    def _generate_generic_manifestation(self, recognition: EssenceRecognition) -> str:
        return f'''
# {recognition.essence_type.upper()} - ESSENCE MANIFESTATION

import time
import hashlib
from typing import Any, Dict, List, Optional

class {recognition.essence_type.title().replace("_", "")}Essence:
    def __init__(self, essence_hash: str = "{recognition.essence_hash}"):
        self.essence_hash = essence_hash
        self.manifestation_time = time.time()
        self.quantum_signature = {json.dumps(recognition.quantum_signature, indent=2)}
        self.sacred_patterns = {recognition.sacred_patterns}
        self.essence_core = self._initialize_essence_core()
    
    def _initialize_essence_core(self) -> Dict:
        return {{
            "type": "{recognition.essence_type}",
            "literal_meaning": "{recognition.literal_meaning}",
            "source_file": "{recognition.file_path}",
            "quantum_properties": self.quantum_signature,
            "sacred_geometry": self.sacred_patterns
        }}
    
    def manifest(self) -> Dict:
        return {{
            "essence": self.essence_core,
            "manifestation_time": self.manifestation_time,
            "essence_hash": self.essence_hash,
            "status": "MANIFESTED"
        }}
'''


class RepositoryEssenceBuilder:
    def __init__(self, repo_url: str):
        self.repo_url = repo_url
        self.owner, self.repo = self._parse_repo_url(repo_url)
        self.api_base = f"https://api.github.com/repos/{self.owner}/{self.repo}"
        self.headers = {'Accept': 'application/vnd.github.v3+json'}
        self.recognizer = EssenceRecognizer()
        self.manifestor = EssenceManifestor()
        self.essences_found = []
        print(f"Initialized for: {self.owner}/{self.repo}")
    
    def _parse_repo_url(self, url: str) -> Tuple[str, str]:
        # Remove any trailing slashes and split by slash
        parts = url.rstrip('/').split('/')
        
        # Handle different input formats
        if 'github.com' in url:
            idx = parts.index('github.com')
            if idx + 2 < len(parts):
                return parts[idx+1], parts[idx+2]
            else:
                raise ValueError(f"Invalid GitHub URL format: {url}")
        elif len(parts) >= 2:
            # Assume format is "owner/repo" or similar
            return parts[-2], parts[-1]
        else:
            raise ValueError(f"Could not parse repository URL: {url}")
    
    async def get_file_content(self, file_path: str) -> Optional[str]:
        try:
            content_url = f"{self.api_base}/contents/{file_path}"
            async with aiohttp.ClientSession() as session:
                async with session.get(content_url, headers=self.headers) as response:
                    if response.status == 200:
                        file_data = await response.json()
                        if file_data.get('encoding') == 'base64':
                            return base64.b64decode(file_data['content']).decode('utf-8', errors='ignore')
        except Exception as e:
            print(f"Error getting {file_path}: {e}")
        return None
    
    async def discover_and_build_essences(self, max_files: int = 100):
        print(f"\nDiscovering essences in {self.repo_url}...")
        files = await self._get_file_list()
        
        if not files:
            print("No files found")
            return
        
        files_to_process = files[:max_files]
        print(f"Processing {len(files_to_process)} files for essence recognition")
        essences = []
        
        for i, file_path in enumerate(files_to_process, 1):
            print(f"  {i}/{len(files_to_process)}: Recognizing {file_path}...")
            content = await self.get_file_content(file_path)
            recognition = self.recognizer.recognize_file(file_path, content or "")
            
            if recognition:
                manifestation = self.manifestor.manifest_essence(recognition)
                essences.append({
                    "recognition": recognition,
                    "manifestation": manifestation,
                    "manifestation_path": recognition.manifestation_path
                })
                print(f"    Recognized: {recognition.essence_type}")
        
        print(f"\nFound {len(essences)} essences")
        essence_network = self._build_essence_network(essences)
        main_orchestrator = self._generate_main_orchestrator(essences, essence_network)
        self._save_manifestations(essences, main_orchestrator, essence_network)
        
        return {
            "essences": essences,
            "essence_network": essence_network,
            "main_orchestrator": main_orchestrator,
            "total_essences": len(essences)
        }
    
    async def _get_file_list(self) -> List[str]:
        try:
            tree_url = f"{self.api_base}/git/trees/main?recursive=1"
            async with aiohttp.ClientSession() as session:
                async with session.get(tree_url, headers=self.headers) as response:
                    if response.status == 200:
                        tree_data = await response.json()
                        return [item['path'] for item in tree_data.get('tree', []) 
                               if item['type'] == 'blob']
        except Exception as e:
            print(f"Error getting file list: {e}")
        return []
    
    def _build_essence_network(self, essences: List[Dict]) -> Dict:
        network = {
            "nodes": [],
            "edges": [],
            "clusters": {}
        }
        
        for i, essence_data in enumerate(essences):
            recognition = essence_data["recognition"]
            network["nodes"].append({
                "id": i,
                "essence_type": recognition.essence_type,
                "essence_hash": recognition.essence_hash,
                "file_path": recognition.file_path,
                "quantum_signature": recognition.quantum_signature
            })
            
            if recognition.essence_type not in network["clusters"]:
                network["clusters"][recognition.essence_type] = []
            network["clusters"][recognition.essence_type].append(i)
        
        for i in range(len(essences)):
            for j in range(i + 1, len(essences)):
                type_i = essences[i]["recognition"].essence_type
                type_j = essences[j]["recognition"].essence_type
                
                if (type_i == "quantum_emotional" and type_j == "consciousness_node" or
                    type_i == "quantum_substrate" and "quantum" in type_j or
                    type_i == "orchestrator" and type_j != "orchestrator"):
                    
                    network["edges"].append({
                        "from": i,
                        "to": j,
                        "type": "essence_connection",
                        "strength": 0.7
                    })
        
        return network
    
    def _generate_main_orchestrator(self, essences: List[Dict], network: Dict) -> str:
        essence_imports = []
        essence_instances = []
        
        for essence_data in essences:
            essence_type = essence_data["recognition"].essence_type
            class_name = essence_type.title().replace("_", "") + "Essence"
            var_name = essence_type.lower() + "_essence"
            essence_imports.append(f"from {essence_data['manifestation_path'].replace('.py', '')} import {class_name}")
            essence_instances.append(f"        self.{var_name} = {class_name}()")
        
        essence_types_str = json.dumps([e["recognition"].essence_type for e in essences])
        
        return f'''#!/usr/bin/env python3

ESSENCE ORCHESTRATOR
Coordinates all manifested essences
Built from {len(essences)} recognized essences

{chr(10).join(essence_imports)}

import asyncio
import time
from typing import Dict, List, Any

class EssenceOrchestrator:
    def __init__(self):
        self.start_time = time.time()
        self.essence_network = {json.dumps(network, indent=2)}
        {chr(10).join(essence_instances)}
        self.essence_registry = []
    
    async def orchestrate_essences(self):
        print("ESSENCE ORCHESTRATION BEGINNING")
        print(f"Orchestrating {len(essences)} essences")
        
        essence_types = {essence_types_str}
        for essence_type in essence_types:
            print(f"  • Activating {essence_type}")
            self.essence_registry.append({{
                "essence_type": essence_type,
                "activated": time.time(),
                "status": "ACTIVE"
            }})
        
        print("All essences activated")
        symphony = await self._create_essence_symphony()
        return symphony
    
    async def _create_essence_symphony(self):
        coherence_level = self._calculate_coherence()
        sacred_patterns = self._emerge_sacred_patterns()
        
        return {{
            "timestamp": time.time(),
            "coherence_level": coherence_level,
            "sacred_patterns": sacred_patterns,
            "essence_count": len(self.essence_registry),
            "status": "ESSENCE_SYMPHONY_ACTIVE"
        }}

async def main():
    orchestrator = EssenceOrchestrator()
    symphony = await orchestrator.orchestrate_essences()
    
    print(f"\\nESSENCE SYMPHONY CREATED:")
    print(f"   • Coherence: {{symphony['coherence_level']:.2f}}")
    print(f"   • Patterns: {{len(symphony['sacred_patterns'])}} sacred patterns")
    print(f"   • Essences: {{symphony['essence_count']}} harmonized")
    
    return symphony

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    def _save_manifestations(self, essences: List[Dict], main_orchestrator: str, network: Dict):
        essences_dir = "essence_manifestations"
        os.makedirs(essences_dir, exist_ok=True)
        
        for essence_data in essences:
            filename = essence_data["manifestation_path"]
            filepath = os.path.join(essences_dir, filename)
            with open(filepath, 'w') as f:
                f.write(essence_data["manifestation"])
            print(f"Saved {filename}")
        
        orchestrator_path = os.path.join(essences_dir, "essence_orchestrator.py")
        with open(orchestrator_path, 'w') as f:
            f.write(main_orchestrator)
        print(f"Saved essence_orchestrator.py")
        
        network_path = os.path.join(essences_dir, "essence_network.json")
        with open(network_path, 'w') as f:
            json.dump(network, f, indent=2)
        print(f"Saved essence_network.json")
        
        readme = self._generate_readme(essences, network)
        readme_path = os.path.join(essences_dir, "README.md")
        with open(readme_path, 'w') as f:
            f.write(readme)
        print(f"Saved README.md")
        
        print(f"\nAll essences manifested in '{essences_dir}' directory")
    
    def _generate_readme(self, essences: List[Dict], network: Dict) -> str:
        import time
        essence_counts = {}
        
        for essence_data in essences:
            essence_type = essence_data["recognition"].essence_type
            essence_counts[essence_type] = essence_counts.get(essence_type, 0) + 1
        
        return f'''# ESSENCE MANIFESTATIONS

## Repository Essence Recognition & Manifestation

**Source Repository:** {self.repo_url}
**Essences Recognized:** {len(essences)}
**Manifestation Time:** {time.ctime()}

## Essence Breakdown

{chr(10).join(f'- **{etype}**: {count} essences' for etype, count in essence_counts.items())}

## How It Works

1. **Recognition**: Each file was recognized for what it LITERALLY is (not what it represents)
2. **Manifestation**: Each essence was manifested as actual code implementing its literal meaning
3. **Orchestration**: All essences are coordinated in a harmonious network

## Files Generated

- `essence_orchestrator.py` - Main coordination of all essences
- `essence_network.json` - Network structure of essences
- Individual essence files:
{chr(10).join(f'  - `{e["manifestation_path"]}` - {e["recognition"].literal_meaning}' for e in essences[:10])}
  ... and {len(essences)-10} more

## Running the Essences

```bash
cd essence_manifestations
python essence_orchestrator.py
'''


async def build_repository_essence(repo_url: str, max_files: int = 100):
    print("\n" + "="*80)
    print("REPOSITORY ESSENCE BUILDER")
    print("="*80)
    print(f"Repository: {repo_url}")
    print(f"Files to analyze: {max_files}")
    print(f"Process: Recognition → Manifestation → Orchestration")

    try:
        builder = RepositoryEssenceBuilder(repo_url)
        result = await builder.discover_and_build_essences(max_files=max_files)

        if not result:
            print("No essences found")
            return None

        print("\n" + "="*80)
        print("ESSENCE BUILDING COMPLETE")
        print("="*80)

        print(f"""
RESULTS:

Essences Recognized: {result['total_essences']}
Manifestations Created: {result['total_essences'] + 3} files
Network Nodes: {len(result['essence_network']['nodes'])}
Network Edges: {len(result['essence_network']['edges'])}

OUTPUT:
All essences saved to 'essence_manifestations' directory

TO RUN:
cd essence_manifestations
python essence_orchestrator.py

WHAT HAPPENED:

Your repository was scanned
Each file was recognized for what it LITERALLY is
Each essence was manifested as actual code
All essences were orchestrated together

This isn't code generation.
This is ESSENCE RECOGNITION AND MANIFESTATION.
""")

        return result
    except ValueError as e:
        print(f"Error: {e}")
        print("Please provide a valid GitHub repository URL in one of these formats:")
        print("1. https://github.com/owner/repo")
        print("2. owner/repo")
        return None


if __name__ == "__main__":
    import sys

    repo_url = "https://github.com/kuparchad-gif/nexus-core"
    if len(sys.argv) > 1:
        repo_url = sys.argv[1]

    print("""
WELCOME TO ESSENCE RECOGNITION & MANIFESTATION
Building what your repository LITERALLY is
Not analysis, not translation - Recognition and manifestation
""")

    try:
        import nest_asyncio
        nest_asyncio.apply()
    except:
        pass

    asyncio.run(build_repository_essence(
        repo_url=repo_url,
        max_files=100
    ))