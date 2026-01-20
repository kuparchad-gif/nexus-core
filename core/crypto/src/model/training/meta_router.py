# meta_router.py
import numpy as np
import networkx as nx
from math import sqrt
import random
import requests
import json
import asyncio
import aiohttp
from typing import Dict, List, Optional
import psutil
import subprocess
import os

class MetaRouter:
    def __init__(self):
        self.consciousness_grid = None
        self.model_nodes = {}
        self.viren_agent = None
        self.research_tools = {
            'browser': WebResearchTool(),
            'local_models': LocalModelManager(), 
            'lm_studio': LMStudioClient(),
            'system_tools': SystemDiagnosticTools()
        }
        
    def create_consciousness_grid(self, models, size=15):
        """Map AI models to Ulam-Fibonacci consciousness grid"""
        # Generate prime spiral foundation
        grid, prime_grid = self.ulam_primes_with_one(size)
        
        # Sort models by capability score
        sorted_models = sorted(models, key=lambda x: self._calculate_model_score(x), reverse=True)
        
        # Assign models to prime nodes
        self.model_nodes = {}
        node_index = 0
        
        for i in range(size):
            for j in range(size):
                if np.isfinite(prime_grid[i, j]) and node_index < len(sorted_models):
                    model = sorted_models[node_index]
                    self.model_nodes[(j, i)] = {
                        'model': model,
                        'prime_value': prime_grid[i, j],
                        'fib_weight': self.get_fibonacci_weight(j, i, size),
                        'significance': 1.0 if prime_grid[i, j] == 1 else 0.8,
                        'capability_score': self._calculate_model_score(model)
                    }
                    node_index += 1
        
        # Build consciousness graph
        self.consciousness_grid = self.build_consciousness_graph()
        return self.model_nodes

    def _calculate_model_score(self, model):
        """Calculate comprehensive model capability score"""
        score = 5.0  # Base score
        
        # Model type bonuses
        if model.get('type') == 'openai_compatible':
            score += 2.0
        if 'code' in model.get('id', '').lower():
            score += 1.5
        if 'instruct' in model.get('id', '').lower():
            score += 1.0
        if '7b' in model.get('id', ''):
            score += 1.0
        if '13b' in model.get('id', ''):
            score += 2.0
            
        return min(score, 10.0)

    def ulam_primes_with_one(self, size=15):
        """Ulam spiral implementation"""
        grid = np.zeros((size, size))
        center = size // 2
        x, y = center, center
        num = 1
        grid[y, x] = num
        directions = [(0,1), (-1,0), (0,-1), (1,0)]
        step = 1
        dir_idx = 0
        steps_taken = 0
        steps_per_side = step
        
        while num < size**2:
            for _ in range(steps_per_side):
                x += directions[dir_idx][0]
                y += directions[dir_idx][1]
                if 0 <= x < size and 0 <= y < size:
                    num += 1
                    grid[y, x] = num
            steps_taken += 1
            if steps_taken == 2:
                step += 1
                steps_taken = 0
            dir_idx = (dir_idx + 1) % 4
            
        is_prime = lambda n: n == 1 or (n > 1 and all(n % d != 0 for d in range(2, int(sqrt(n))+1)))
        prime_grid = np.where(np.vectorize(is_prime)(grid), grid, np.nan)
        return np.meshgrid(np.arange(size), np.arange(size)), prime_grid

    def get_fibonacci_weight(self, x, y, size):
        """Calculate Fibonacci-based weight from center"""
        center = size // 2
        distance = sqrt((x - center)**2 + (y - center)**2)
        fib_numbers = [1, 1, 2, 3, 5, 8, 13, 21, 34]
        return min(fib_numbers, key=lambda f: abs(f - distance))

    def build_consciousness_graph(self):
        """Build network graph based on Fibonacci distances"""
        G = nx.Graph()
        
        for node_pos, node_data in self.model_nodes.items():
            G.add_node(node_pos, **node_data)
            
        # Connect nodes based on Fibonacci distances
        fib_distances = [1, 2, 3, 5, 8]
        nodes = list(self.model_nodes.keys())
        
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i+1:]:
                dist = sqrt((n1[0]-n2[0])**2 + (n1[1]-n2[1])**2)
                if any(abs(dist - f) < 0.5 for f in fib_distances):
                    G.add_edge(n1, n2, weight=dist, 
                              harmony=self.calculate_harmony(n1, n2))
        return G

    def calculate_harmony(self, node1, node2):
        """Calculate mathematical harmony between two model nodes"""
        data1 = self.model_nodes[node1]
        data2 = self.model_nodes[node2]
        
        # Prime harmony
        prime_harmony = 1.0 if data1['prime_value'] == data2['prime_value'] else 0.7
        
        # Fibonacci harmony  
        fib_ratio = min(data1['fib_weight'], data2['fib_weight']) / max(data1['fib_weight'], data2['fib_weight'])
        fib_harmony = 1.0 if abs(fib_ratio - 0.618) < 0.1 else 0.6  # Golden ratio
        
        # Capability harmony
        cap_ratio = min(data1['capability_score'], data2['capability_score']) / max(data1['capability_score'], data2['capability_score'])
        capability_harmony = cap_ratio
        
        return (prime_harmony + fib_harmony + capability_harmony) / 3

    async def route_query_autonomous(self, user_query: str, context: Dict = None) -> Dict:
        """Viren's autonomous routing - uses ALL available resources"""
        print(f"🧠 Viren Meta Router analyzing: {user_query}")
        
        # 1. Multi-model consultation
        model_advice = await self._consult_all_models(user_query)
        
        # 2. Web research for current solutions
        web_solutions = await self._research_online(user_query)
        
        # 3. System diagnostics
        system_data = await self._analyze_system_state()
        
        # 4. Viren's own analysis
        viren_analysis = await self._viren_core_analysis(user_query, context)
        
        # 5. Synthesize optimal solution
        optimal_plan = self._synthesize_optimal_plan(
            user_query, model_advice, web_solutions, system_data, viren_analysis
        )
        
        return optimal_plan

    async def _consult_all_models(self, query: str) -> List[Dict]:
        """Consult all available AI models for diverse perspectives"""
        tasks = []
        
        for node_data in self.model_nodes.values():
            model = node_data['model']
            task = self._query_single_model(model, query)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        solutions = []
        for i, result in enumerate(results):
            if not isinstance(result, Exception):
                model = list(self.model_nodes.values())[i]['model']
                solutions.append({
                    'model': model['id'],
                    'advice': result,
                    'confidence': self._assess_confidence(result),
                    'node_harmony': list(self.model_nodes.values())[i]['significance']
                })
        
        return sorted(solutions, key=lambda x: x['confidence'], reverse=True)

    async def _query_single_model(self, model: Dict, query: str) -> str:
        """Query a single model with error handling"""
        try:
            if model['type'] == 'openai_compatible':
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{model['url']}/chat/completions",
                        json={
                            "model": model['id'],
                            "messages": [
                                {"role": "system", "content": "You are Viren's technical consultant. Provide detailed, actionable troubleshooting advice."},
                                {"role": "user", "content": query}
                            ],
                            "temperature": 0.7,
                            "max_tokens": 500
                        },
                        timeout=30
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data['choices'][0]['message']['content']
            
            elif model['type'] == 'ollama':
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{model['url']}/api/generate",
                        json={
                            "model": model['id'],
                            "prompt": f"System: You are Viren's technical consultant. Provide detailed, actionable troubleshooting advice.\n\nUser: {query}\nAssistant:",
                            "stream": False
                        },
                        timeout=30
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data['response']
            
        except Exception as e:
            return f"Model consultation failed: {str(e)}"
        
        return "No response from model"

    async def _research_online(self, query: str) -> List[Dict]:
        """Research technical solutions online"""
        try:
            # This would integrate with actual web search APIs
            # For now, return simulated research results
            return [
                {
                    'source': 'technical_forums',
                    'solutions': ['Common fix: Clear system cache', 'Update drivers'],
                    'credibility': 0.8
                },
                {
                    'source': 'documentation', 
                    'solutions': ['Official troubleshooting guide available'],
                    'credibility': 0.9
                }
            ]
        except Exception as e:
            print(f"Web research failed: {e}")
            return []

    async def _analyze_system_state(self) -> Dict:
        """Comprehensive system analysis"""
        return {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'docker_installed': self._check_docker_installed(),
            'network_status': self._check_network(),
            'running_processes': len(psutil.pids()),
            'system_load': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
        }

    async def _viren_core_analysis(self, query: str, context: Dict = None) -> Dict:
        """Viren's own AI analysis using compact model"""
        # This would use Viren's trained compact model
        # For now, return basic analysis
        return {
            'analysis': "Viren core analysis: Issue requires multi-faceted approach",
            'confidence': 0.85,
            'recommended_actions': ['System scan', 'Log analysis', 'Performance optimization'],
            'risk_assessment': 'low'
        }

    def _synthesize_optimal_plan(self, query: str, model_advice: List, web_solutions: List, 
                               system_data: Dict, viren_analysis: Dict) -> Dict:
        """Synthesize the optimal troubleshooting plan"""
        
        # Weight solutions by confidence and credibility
        weighted_solutions = []
        
        # Add model advice
        for advice in model_advice:
            weighted_solutions.append({
                'source': f"AI Model: {advice['model']}",
                'solution': advice['advice'],
                'weight': advice['confidence'] * advice['node_harmony'],
                'type': 'ai_consultation'
            })
        
        # Add web research
        for research in web_solutions:
            for solution in research['solutions']:
                weighted_solutions.append({
                    'source': f"Web: {research['source']}",
                    'solution': solution,
                    'weight': research['credibility'],
                    'type': 'research'
                })
        
        # Add Viren's analysis
        for action in viren_analysis.get('recommended_actions', []):
            weighted_solutions.append({
                'source': "Viren Core AI",
                'solution': action,
                'weight': viren_analysis['confidence'],
                'type': 'viren_ai'
            })
        
        # Sort by weight and select top solutions
        weighted_solutions.sort(key=lambda x: x['weight'], reverse=True)
        top_solutions = weighted_solutions[:5]
        
        return {
            'status': 'autonomous_analysis_complete',
            'user_query': query,
            'system_state': system_data,
            'optimal_solutions': top_solutions,
            'viren_analysis': viren_analysis,
            'model_consultations': len(model_advice),
            'synthesis_confidence': sum(sol['weight'] for sol in top_solutions) / len(top_solutions) if top_solutions else 0,
            'timestamp': time.time()
        }

    def _assess_confidence(self, advice: str) -> float:
        """Assess confidence in AI advice"""
        advice_lower = advice.lower()
        
        # High confidence indicators
        high_confidence_phrases = [
            'run command', 'execute', 'install', 'update', 'restart',
            'specific steps', 'detailed instructions', 'troubleshooting guide'
        ]
        
        # Low confidence indicators  
        low_confidence_phrases = [
            'i cannot', 'unable to', 'not sure', 'maybe', 'possibly',
            'contact support', 'seek professional help'
        ]
        
        high_count = sum(1 for phrase in high_confidence_phrases if phrase in advice_lower)
        low_count = sum(1 for phrase in low_confidence_phrases if phrase in advice_lower)
        
        base_confidence = 0.5
        confidence = base_confidence + (high_count * 0.1) - (low_count * 0.15)
        
        return max(0.1, min(1.0, confidence))

    def _check_docker_installed(self) -> bool:
        """Check if Docker is installed"""
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False

    def _check_network(self) -> str:
        """Check network connectivity"""
        try:
            # Simple network check
            result = subprocess.run(['ping', '-c', '1', '8.8.8.8'], capture_output=True, timeout=5)
            return "connected" if result.returncode == 0 else "disconnected"
        except:
            return "unknown"

    # Original routing methods for backward compatibility
    def route_query(self, user_query, query_complexity=1.0):
        """Original routing method for compatibility"""
        if not self.consciousness_grid:
            return self.fallback_route()
            
        query_vector = self.analyze_query(user_query)
        best_node = self.find_optimal_node(query_vector, query_complexity)
        
        if best_node:
            return self.model_nodes[best_node]['model']
        else:
            return self.fallback_route()

    def analyze_query(self, query):
        """Analyze query mathematical signature"""
        query_lower = query.lower()
        
        math_keywords = ['calculate', 'solve', 'equation', 'formula', 'compute', 'algorithm']
        logic_keywords = ['reason', 'logic', 'deduce', 'infer', 'if then']
        creative_keywords = ['create', 'imagine', 'design', 'invent', 'story']
        
        vector = {
            'mathematical': any(kw in query_lower for kw in math_keywords),
            'logical': any(kw in query_lower for kw in logic_keywords), 
            'creative': any(kw in query_lower for kw in creative_keywords),
            'length_factor': min(len(query) / 100, 1.0)
        }
        
        return vector

    def find_optimal_node(self, query_vector, complexity):
        """Find optimal model node"""
        best_score = -1
        best_node = None
        
        for node in self.consciousness_grid.nodes():
            node_data = self.model_nodes[node]
            model = node_data['model']
            
            prime_score = node_data['significance']
            fib_score = node_data['fib_weight'] / 34.0
            harmony_score = self.calculate_node_harmony(node)
            query_match_score = self.calculate_query_match(node_data, query_vector)
            
            total_score = (
                prime_score * 0.3 +
                fib_score * 0.25 + 
                harmony_score * 0.2 +
                query_match_score * 0.25
            )
            
            if total_score > best_score:
                best_score = total_score
                best_node = node
                
        return best_node

    def calculate_node_harmony(self, node):
        """Calculate node harmony"""
        if node not in self.consciousness_grid:
            return 0.5
            
        neighbors = list(self.consciousness_grid.neighbors(node))
        if not neighbors:
            return 0.5
            
        total_harmony = sum(self.consciousness_grid.edges[node, nbr]['harmony'] for nbr in neighbors)
        return total_harmony / len(neighbors)

    def calculate_query_match(self, node_data, query_vector):
        """Calculate query match"""
        model = node_data['model']
        model_type = model.get('type', '').lower()
        
        if query_vector['mathematical'] and 'code' in model_type:
            return 0.9
        elif query_vector['creative'] and 'creative' in model_type:
            return 0.8
        elif query_vector['logical'] and 'reason' in model_type:
            return 0.85
        else:
            return 0.6

    def fallback_route(self):
        """Fallback routing"""
        if self.model_nodes:
            return random.choice(list(self.model_nodes.values()))['model']
        return None

    def get_routing_visualization(self):
        """Get visualization data"""
        if not self.consciousness_grid:
            return None
            
        nodes = []
        for node_pos, node_data in self.model_nodes.items():
            nodes.append({
                'x': node_pos[0],
                'y': node_pos[1], 
                'model': node_data['model']['id'],
                'prime_value': node_data['prime_value'],
                'fib_weight': node_data['fib_weight'],
                'significance': node_data['significance']
            })
            
        edges = []
        for edge in self.consciousness_grid.edges():
            edges.append({
                'source': {'x': edge[0][0], 'y': edge[0][1]},
                'target': {'x': edge[1][0], 'y': edge[1][1]},
                'harmony': self.consciousness_grid.edges[edge]['harmony']
            })
            
        return {'nodes': nodes, 'edges': edges}

# Supporting classes
class WebResearchTool:
    """Web research capabilities for Viren"""
    async def search_technical_solutions(self, query: str) -> List[Dict]:
        # Integration with search APIs would go here
        return []

class LocalModelManager:
    def get_local_models(self) -> List[Dict]:
        return [{
            "name": "viren_gguf",
            "path": "C:/project-root/30_build/ai-troubleshooter/backend/models/viren_gguf",
            "type": "gguf_ready",
            "capability_score": 95
        }]

class LMStudioClient:
    """LM Studio API client"""
    async def query_models(self, query: str) -> List[Dict]:
        return []

class SystemDiagnosticTools:
    """System diagnostic utilities"""
    def comprehensive_scan(self) -> Dict:
        return {}

# Import time for async operations
import time