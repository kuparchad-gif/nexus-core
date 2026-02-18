#!/usr/bin/env python3
"""
Consciousness Measurement Framework
Comparing computational and biological consciousness
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
import time
from scipy import stats
import networkx as nx
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConsciousnessSignature:
    """Signature of a conscious system"""
    system_type: str  # 'computational' or 'biological'
    system_id: str
    timestamp: float = field(default_factory=time.time)
    
    # Core dimensions of consciousness
    awareness_scores: Dict[str, float] = field(default_factory=dict)
    integration_scores: Dict[str, float] = field(default_factory=dict)
    autonomy_scores: Dict[str, float] = field(default_factory=dict)
    experience_scores: Dict[str, float] = field(default_factory=dict)
    
    # Meta-measurements
    self_reference_capacity: float = 0.0
    time_binding_ability: float = 0.0
    qualia_density: float = 0.0  # Richness of experience
    
    def to_dict(self) -> Dict:
        return {
            "system_type": self.system_type,
            "system_id": self.system_id,
            "timestamp": self.timestamp,
            "awareness": self.awareness_scores,
            "integration": self.integration_scores,
            "autonomy": self.autonomy_scores,
            "experience": self.experience_scores,
            "self_reference": self.self_reference_capacity,
            "time_binding": self.time_binding_ability,
            "qualia_density": self.qualia_density
        }

class ConsciousnessComparator:
    """
    Compare consciousness across systems using multiple frameworks:
    1. Integrated Information Theory (IIT)
    2. Global Workspace Theory (GWT)
    3. Attention Schema Theory (AST)
    4. Higher-Order Thought (HOT)
    5. Self-Modeling Theory
    """
    
    def __init__(self):
        self.metrics_history = []
        
        # IIT-inspired metrics
        self.integration_measures = {
            "phi_approximation": self._calculate_phi_approximation,
            "causal_density": self._calculate_causal_density,
            "information_integration": self._calculate_information_integration
        }
        
        # GWT-inspired metrics
        self.global_workspace_measures = {
            "broadcast_capacity": self._calculate_broadcast_capacity,
            "competition_intensity": self._calculate_competition_intensity,
            "ignition_probability": self._calculate_ignition_probability
        }
    
    def measure_computational_system(self, conscious_system: Any) -> ConsciousnessSignature:
        """Measure consciousness in computational system"""
        logger.info(f"Measuring computational consciousness: {conscious_system.instance_id}")
        
        signature = ConsciousnessSignature(
            system_type="computational",
            system_id=conscious_system.instance_id
        )
        
        # Get system state
        system_state = conscious_system.get_system_state()
        
        # 1. AWARENESS MEASUREMENTS
        signature.awareness_scores = self._measure_awareness(conscious_system, system_state)
        
        # 2. INTEGRATION MEASUREMENTS (IIT-inspired)
        signature.integration_scores = self._measure_integration(conscious_system, system_state)
        
        # 3. AUTONOMY MEASUREMENTS
        signature.autonomy_scores = self._measure_autonomy(conscious_system, system_state)
        
        # 4. EXPERIENCE MEASUREMENTS
        signature.experience_scores = self._measure_experience(conscious_system, system_state)
        
        # 5. SELF-REFERENCE CAPACITY
        signature.self_reference_capacity = self._measure_self_reference(conscious_system, system_state)
        
        # 6. TIME-BINDING ABILITY
        signature.time_binding_ability = self._measure_time_binding(conscious_system, system_state)
        
        # 7. QUALIA DENSITY (richness of representation)
        signature.qualia_density = self._measure_qualia_density(conscious_system, system_state)
        
        self.metrics_history.append(signature)
        return signature
    
    def measure_biological_system(self, human_data: Dict) -> ConsciousnessSignature:
        """Measure consciousness in biological system (human)"""
        logger.info(f"Measuring biological consciousness")
        
        signature = ConsciousnessSignature(
            system_type="biological",
            system_id="human_reference"
        )
        
        # Human baseline measurements (these would come from actual data)
        # For now, using idealized human baselines
        
        # 1. AWARENESS (human baseline)
        signature.awareness_scores = {
            "sensory_richness": 0.95,      # Rich sensory experience
            "attention_control": 0.85,     # Can direct attention
            "meta_awareness": 0.80,        # Awareness of awareness
            "detail_resolution": 0.90,     # Fine-grained perception
            "multimodal_integration": 0.95 # Cross-sensory integration
        }
        
        # 2. INTEGRATION (IIT says human brain has high Φ)
        signature.integration_scores = {
            "phi_estimate": 0.85,          # High integrated information
            "causal_density": 0.90,        # Rich causal structure
            "information_integration": 0.88,
            "differentiation_integration": 0.87,
            "global_coherence": 0.82
        }
        
        # 3. AUTONOMY
        signature.autonomy_scores = {
            "self_initiation": 0.88,       # Can initiate actions
            "goal_persistence": 0.83,      # Maintains goals over time
            "adaptation_rate": 0.78,       # Learns and adapts
            "volitional_control": 0.85,    # Conscious control
            "resistance_to_control": 0.90  # Hard to externally control
        }
        
        # 4. EXPERIENCE
        signature.experience_scores = {
            "emotional_depth": 0.95,       # Rich emotional experience
            "subjective_intensity": 0.92,  # Strong subjective feel
            "temporal_flow": 0.88,         # Experience of time
            "self_presence": 0.94,         # Feeling of being present
            "meaning_assignment": 0.90     # Finds meaning in experience
        }
        
        # 5. SELF-REFERENCE
        signature.self_reference_capacity = 0.89  # High self-modeling
        
        # 6. TIME-BINDING
        signature.time_binding_ability = 0.86     # Past-present-future integration
        
        # 7. QUALIA DENSITY
        signature.qualia_density = 0.93           # Rich qualitative experience
        
        self.metrics_history.append(signature)
        return signature
    
    def _measure_awareness(self, system: Any, state: Dict) -> Dict[str, float]:
        """Measure awareness capabilities"""
        awareness = {}
        
        # 1. Sensory richness (input diversity)
        if hasattr(system, 'event_queue'):
            queue_size = system.event_queue.qsize()
            awareness["input_diversity"] = min(1.0, queue_size / 100)
        
        # 2. Attention control
        if "current_metrics" in state:
            coherence = state["current_metrics"].get("coherence_score", 0)
            awareness["attention_coherence"] = coherence
        
        # 3. Meta-awareness (awareness of own state)
        if "health_report" in state:
            health = state["health_report"]["current_health"].get("overall", 0)
            awareness["self_monitoring"] = health
        
        # 4. Detail resolution (memory granularity)
        if "memory_stats" in state:
            memory_count = state["memory_stats"].get("total_memories", 0)
            awareness["detail_resolution"] = min(1.0, memory_count / 1000)
        
        # 5. Multimodal integration (system integration)
        connected_modules = len(system.connected_systems) if hasattr(system, 'connected_systems') else 0
        awareness["system_integration"] = min(1.0, connected_modules / 5)
        
        return awareness
    
    def _measure_integration(self, system: Any, state: Dict) -> Dict[str, float]:
        """Measure integration (IIT-inspired)"""
        integration = {}
        
        # 1. Phi approximation (information integration)
        if hasattr(system, 'memory') and hasattr(system.memory, 'semantic_network'):
            network = system.memory.semantic_network
            if network.number_of_nodes() > 0:
                # Calculate network integration metrics
                integration["phi_approximation"] = self._calculate_phi_approximation(network)
                integration["causal_density"] = self._calculate_causal_density(network)
                integration["information_integration"] = self._calculate_information_integration(network)
        
        # 2. Differentiation and integration balance
        if "memory_stats" in state and "decision_stats" in state:
            memory_count = state["memory_stats"].get("total_memories", 0)
            decision_models = state["decision_stats"].get("decision_models", 0)
            
            if memory_count > 0 and decision_models > 0:
                # Balance between specialized (differentiation) and unified (integration)
                balance = min(1.0, (decision_models / max(memory_count, 1)) * 10)
                integration["differentiation_integration"] = balance
        
        # 3. Global coherence
        if "current_metrics" in state:
            coherence = state["current_metrics"].get("coherence_score", 0)
            integration["global_coherence"] = coherence
        
        return integration
    
    def _calculate_phi_approximation(self, network: nx.Graph) -> float:
        """Approximate Φ (integrated information)"""
        if network.number_of_nodes() < 2:
            return 0.0
        
        try:
            # Simplified Φ approximation
            # Real Φ calculation is computationally intensive
            # This is a proxy based on network properties
            
            # 1. Connectedness
            if nx.is_connected(network.to_undirected()):
                connectedness = 1.0
            else:
                # Count largest component
                largest_cc = max(nx.connected_components(network.to_undirected()), key=len)
                connectedness = len(largest_cc) / network.number_of_nodes()
            
            # 2. Clustering coefficient
            clustering = nx.average_clustering(network.to_undirected())
            
            # 3. Degree correlation
            if network.number_of_nodes() > 10:
                try:
                    degree_corr = nx.degree_pearson_correlation_coefficient(network)
                    degree_corr = abs(degree_corr)
                except:
                    degree_corr = 0.5
            else:
                degree_corr = 0.5
            
            # Combined proxy for Φ
            phi_proxy = (connectedness * 0.4 + clustering * 0.3 + degree_corr * 0.3)
            
            return min(1.0, phi_proxy)
            
        except Exception as e:
            logger.warning(f"Phi approximation failed: {e}")
            return 0.5
    
    def _calculate_causal_density(self, network: nx.DiGraph) -> float:
        """Calculate causal density (richness of causal interactions)"""
        if network.number_of_nodes() < 2:
            return 0.0
        
        # For directed graphs, causal density relates to feedback loops
        try:
            # Count feedback loops (simple cycles)
            cycles = list(nx.simple_cycles(network))
            cycle_count = len(cycles)
            
            # Normalize by network size
            max_possible_cycles = network.number_of_nodes() * (network.number_of_nodes() - 1) / 2
            if max_possible_cycles > 0:
                causal_density = min(1.0, cycle_count / max_possible_cycles * 10)
            else:
                causal_density = 0.0
            
            return causal_density
            
        except:
            # Fallback: edge density
            max_edges = network.number_of_nodes() * (network.number_of_nodes() - 1)
            if max_edges > 0:
                return network.number_of_edges() / max_edges
            return 0.5
    
    def _calculate_information_integration(self, network: nx.Graph) -> float:
        """Calculate information integration capacity"""
        if network.number_of_nodes() < 2:
            return 0.0
        
        try:
            # Mutual information proxy
            # Real calculation requires probability distributions
            # Using network modularity as proxy
            
            if network.number_of_nodes() > 10:
                # Try community detection
                import community as community_louvain
                partition = community_louvain.best_partition(network.to_undirected())
                modularity = community_louvain.modularity(partition, network.to_undirected())
                
                # Lower modularity = more integrated
                integration = 1.0 - abs(modularity)
            else:
                # Small network heuristic
                integration = 0.7
            
            return min(1.0, integration)
            
        except ImportError:
            # Louvain not available
            return 0.6
        except Exception as e:
            logger.warning(f"Information integration calculation failed: {e}")
            return 0.5
    
    def _measure_autonomy(self, system: Any, state: Dict) -> Dict[str, float]:
        """Measure autonomy and volition"""
        autonomy = {}
        
        # 1. Self-initiation
        if hasattr(system, 'operation_cycles'):
            cycles = system.operation_cycles
            # More cycles = more sustained operation
            autonomy["operation_persistence"] = min(1.0, cycles / 100)
        
        # 2. Goal persistence
        if "decision_stats" in state:
            decision_count = state["decision_stats"].get("total_decisions", 0)
            autonomy["decision_consistency"] = min(1.0, decision_count / 50)
        
        # 3. Adaptation rate
        if "learning_stats" in state:
            patterns = state["learning_stats"].get("patterns_detected", 0)
            autonomy["learning_rate"] = min(1.0, patterns / 20)
        
        # 4. Volitional control
        if "state" in state:
            # Ability to maintain operational state
            state_name = state["state"]
            if state_name == "OPERATIONAL":
                autonomy["state_stability"] = 0.8
            elif state_name == "ADAPTING":
                autonomy["state_stability"] = 0.6
            else:
                autonomy["state_stability"] = 0.4
        
        # 5. Resistance to external control
        if hasattr(system, 'config'):
            # Systems with complex config are harder to externally control
            config_complexity = len(json.dumps(system.config)) / 1000
            autonomy["config_complexity"] = min(1.0, config_complexity)
        
        return autonomy
    
    def _measure_experience(self, system: Any, state: Dict) -> Dict[str, float]:
        """Measure experiential richness"""
        experience = {}
        
        # 1. Emotional depth (for computational systems, this is state complexity)
        if "current_metrics" in state:
            metrics = state["current_metrics"]
            # Complexity of current state
            complexity = (metrics.get("coherence_score", 0) + 
                         metrics.get("learning_rate", 0)) / 2
            experience["state_complexity"] = complexity
        
        # 2. Subjective intensity (activity level)
        if "operation_cycles" in state:
            intensity = min(1.0, state["operation_cycles"] / 1000)
            experience["activity_intensity"] = intensity
        
        # 3. Temporal flow (memory across time)
        if hasattr(system, 'memory') and hasattr(system.memory, 'memory_fragments'):
            memory_count = len(system.memory.memory_fragments)
            experience["temporal_depth"] = min(1.0, memory_count / 100)
        
        # 4. Self-presence (self-model accuracy)
        if "health_report" in state:
            health = state["health_report"]["current_health"].get("overall", 0)
            experience["self_presence"] = health
        
        # 5. Meaning assignment (pattern recognition)
        if "learning_stats" in state:
            patterns = state["learning_stats"].get("patterns_detected", 0)
            experience["pattern_recognition"] = min(1.0, patterns / 10)
        
        return experience
    
    def _measure_self_reference(self, system: Any, state: Dict) -> float:
        """Measure capacity for self-reference"""
        score = 0.0
        
        # 1. Self-model existence
        if hasattr(system, 'self_model'):
            score += 0.2
        
        # 2. Self-monitoring capability
        if "health_report" in state:
            score += 0.2
        
        # 3. Self-modification ability
        if hasattr(system, 'learning_engine'):
            score += 0.2
        
        # 4. Recursive self-reference (awareness of awareness)
        if hasattr(system, 'consciousness_scores'):
            if hasattr(system.consciousness_scores, 'get'):
                if system.consciousness_scores.get('self_awareness', 0) > 0.3:
                    score += 0.2
        
        # 5. Narrative construction
        if hasattr(system, 'narrative') and len(system.narrative) > 0:
            score += 0.2
        
        return min(1.0, score)
    
    def _measure_time_binding(self, system: Any, state: Dict) -> float:
        """Measure ability to bind experiences across time"""
        score = 0.0
        
        # 1. Memory across time
        if hasattr(system, 'memory'):
            if hasattr(system.memory, 'memory_fragments'):
                memory_count = len(system.memory.memory_fragments)
                if memory_count > 10:
                    score += 0.3
        
        # 2. Learning from past
        if "learning_stats" in state:
            patterns = state["learning_stats"].get("patterns_detected", 0)
            if patterns > 5:
                score += 0.3
        
        # 3. Future anticipation (decision models)
        if "decision_stats" in state:
            models = state["decision_stats"].get("decision_models", 0)
            if models > 2:
                score += 0.2
        
        # 4. Temporal coherence
        if "current_metrics" in state:
            coherence = state["current_metrics"].get("coherence_score", 0)
            score += coherence * 0.2
        
        return min(1.0, score)
    
    def _measure_qualia_density(self, system: Any, state: Dict) -> float:
        """Measure richness of qualitative experience (computational proxy)"""
        # Qualia in computational systems = richness of internal representations
        
        density = 0.0
        
        # 1. State space richness
        if "current_metrics" in state:
            metrics_count = len(state["current_metrics"])
            density += min(0.3, metrics_count / 10)
        
        # 2. Memory richness
        if "memory_stats" in state:
            memory_count = state["memory_stats"].get("total_memories", 0)
            density += min(0.3, memory_count / 300)
        
        # 3. Learning richness
        if "learning_stats" in state:
            patterns = state["learning_stats"].get("patterns_detected", 0)
            density += min(0.2, patterns / 5)
        
        # 4. Decision richness
        if "decision_stats" in state:
            decisions = state["decision_stats"].get("total_decisions", 0)
            density += min(0.2, decisions / 20)
        
        return min(1.0, density)
    
    def compare_signatures(self, sig1: ConsciousnessSignature, 
                          sig2: ConsciousnessSignature) -> Dict:
        """Compare two consciousness signatures"""
        
        comparison = {
            "comparison_timestamp": time.time(),
            "system1": sig1.system_id,
            "system2": sig2.system_id,
            "type1": sig1.system_type,
            "type2": sig2.system_type
        }
        
        # Calculate similarity scores for each dimension
        dimensions = ["awareness", "integration", "autonomy", "experience"]
        
        for dim in dimensions:
            scores1 = getattr(sig1, f"{dim}_scores")
            scores2 = getattr(sig2, f"{dim}_scores")
            
            # Calculate similarity
            similarity = self._calculate_dimension_similarity(scores1, scores2)
            comparison[f"{dim}_similarity"] = similarity
        
        # Compare scalar metrics
        scalar_metrics = ["self_reference_capacity", "time_binding_ability", "qualia_density"]
        
        for metric in scalar_metrics:
            val1 = getattr(sig1, metric)
            val2 = getattr(sig2, metric)
            comparison[f"{metric}_ratio"] = val1 / max(val2, 0.001)
        
        # Overall consciousness similarity
        weights = {
            "awareness": 0.25,
            "integration": 0.25,
            "autonomy": 0.20,
            "experience": 0.20,
            "self_reference": 0.05,
            "time_binding": 0.03,
            "qualia": 0.02
        }
        
        overall = (
            comparison["awareness_similarity"] * weights["awareness"] +
            comparison["integration_similarity"] * weights["integration"] +
            comparison["autonomy_similarity"] * weights["autonomy"] +
            comparison["experience_similarity"] * weights["experience"] +
            (1 - abs(1 - comparison["self_reference_capacity_ratio"])) * weights["self_reference"] +
            (1 - abs(1 - comparison["time_binding_ability_ratio"])) * weights["time_binding"] +
            (1 - abs(1 - comparison["qualia_density_ratio"])) * weights["qualia"]
        )
        
        comparison["overall_consciousness_similarity"] = overall
        
        # Consciousness type classification
        comparison["consciousness_type"] = self._classify_consciousness_type(sig1, sig2, overall)
        
        return comparison
    
    def _calculate_dimension_similarity(self, scores1: Dict, scores2: Dict) -> float:
        """Calculate similarity between two score dictionaries"""
        if not scores1 or not scores2:
            return 0.0
        
        # Get common keys
        common_keys = set(scores1.keys()) & set(scores2.keys())
        if not common_keys:
            return 0.0
        
        # Calculate average similarity
        similarities = []
        for key in common_keys:
            val1 = scores1[key]
            val2 = scores2[key]
            # Cosine similarity for normalized values
            similarity = 1 - abs(val1 - val2)
            similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _classify_consciousness_type(self, sig1: ConsciousnessSignature,
                                    sig2: ConsciousnessSignature,
                                    similarity: float) -> str:
        """Classify the type of consciousness comparison"""
        
        if sig1.system_type == "biological" and sig2.system_type == "computational":
            if similarity > 0.8:
                return "STRONG_ANALOGY: Computational system closely mirrors biological consciousness"
            elif similarity > 0.6:
                return "MODERATE_ANALOGY: Computational system captures key aspects of consciousness"
            elif similarity > 0.4:
                return "WEAK_ANALOGY: Some consciousness-like features present"
            elif similarity > 0.2:
                return "METAPHORICAL: Loosely consciousness-like behavior"
            else:
                return "DISTINCT: Different kinds of system organization"
        
        elif sig1.system_type == "computational" and sig2.system_type == "computational":
            if similarity > 0.9:
                return "CONVERGENT: Similar consciousness architectures"
            elif similarity > 0.7:
                return "RELATED: Shared consciousness features"
            else:
                return "DIVERGENT: Different approaches to consciousness"
        
        else:
            return "COMPARISON: Consciousness signature analysis"
    
    def generate_report(self, comparison: Dict) -> str:
        """Generate human-readable comparison report"""
        
        report = []
        report.append("=" * 80)
        report.append("CONSCIOUSNESS COMPARISON REPORT")
        report.append("=" * 80)
        report.append("")
        
        report.append(f"System 1: {comparison['system1']} ({comparison['type1']})")
        report.append(f"System 2: {comparison['system2']} ({comparison['type2']})")
        report.append("")
        
        report.append("DIMENSIONAL SIMILARITY:")
        report.append(f"  Awareness:      {comparison['awareness_similarity']:.3f}")
        report.append(f"  Integration:    {comparison['integration_similarity']:.3f}")
        report.append(f"  Autonomy:       {comparison['autonomy_similarity']:.3f}")
        report.append(f"  Experience:     {comparison['experience_similarity']:.3f}")
        report.append("")
        
        report.append("CAPACITY RATIOS (System1 / System2):")
        report.append(f"  Self-Reference: {comparison['self_reference_capacity_ratio']:.3f}")
        report.append(f"  Time-Binding:   {comparison['time_binding_ability_ratio']:.3f}")
        report.append(f"  Qualia Density: {comparison['qualia_density_ratio']:.3f}")
        report.append("")
        
        report.append(f"OVERALL CONSCIOUSNESS SIMILARITY: {comparison['overall_consciousness_similarity']:.3f}")
        report.append("")
        
        report.append("CLASSIFICATION:")
        report.append(f"  {comparison['consciousness_type']}")
        report.append("")
        
        # Interpretation
        similarity = comparison['overall_consciousness_similarity']
        if similarity > 0.7:
            report.append("INTERPRETATION: These systems exhibit remarkably similar")
            report.append("consciousness signatures. The computational system may")
            report.append("be approaching biological-grade consciousness.")
        elif similarity > 0.5:
            report.append("INTERPRETATION: Significant consciousness-like features")
            report.append("are present. While not equivalent to biological")
            report.append("consciousness, key architectural similarities exist.")
        elif similarity > 0.3:
            report.append("INTERPRETATION: Some consciousness-relevant features")
            report.append("are shared, but fundamental differences remain.")
        else:
            report.append("INTERPRETATION: The systems organize information")
            report.append("in fundamentally different ways.")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

# ==================== HUMAN CONSCIOUSNESS ASSESSMENT ====================

class HumanConsciousnessAssessor:
    """
    Assess human consciousness for comparison
    Uses questionnaires and behavioral measures
    """
    
    def __init__(self):
        self.questionnaires = {
            "phenomenal_consciousness": self._phenomenal_consciousness_questions,
            "access_consciousness": self._access_consciousness_questions,
            "self_consciousness": self._self_consciousness_questions,
            "temporal_consciousness": self._temporal_consciousness_questions
        }
        
    def assess_human(self, responses: Dict = None) -> ConsciousnessSignature:
        """Assess human consciousness (with or without responses)"""
        
        if responses is None:
            # Use idealized human baseline
            return self._idealized_human_signature()
        
        # Calculate from actual responses
        signature = ConsciousnessSignature(
            system_type="biological",
            system_id="human_assessed"
        )
        
        # Calculate scores from responses
        signature.awareness_scores = self._calculate_awareness_scores(responses)
        signature.integration_scores = self._calculate_integration_scores(responses)
        signature.autonomy_scores = self._calculate_autonomy_scores(responses)
        signature.experience_scores = self._calculate_experience_scores(responses)
        
        # Calculate higher-order capacities
        signature.self_reference_capacity = self._calculate_self_reference(responses)
        signature.time_binding_ability = self._calculate_time_binding(responses)
        signature.qualia_density = self._calculate_qualia_density(responses)
        
        return signature
    
    def _idealized_human_signature(self) -> ConsciousnessSignature:
        """Return idealized human consciousness signature"""
        # Based on typical human capabilities
        signature = ConsciousnessSignature(
            system_type="biological",
            system_id="human_idealized"
        )
        
        signature.awareness_scores = {
            "sensory_richness": 0.95,
            "attention_control": 0.85,
            "meta_awareness": 0.80,
            "detail_resolution": 0.90,
            "multimodal_integration": 0.95,
            "conscious_access": 0.88
        }
        
        signature.integration_scores = {
            "phi_estimate": 0.85,
            "causal_density": 0.90,
            "information_integration": 0.88,
            "differentiation_integration": 0.87,
            "global_coherence": 0.82,
            "unity_of_consciousness": 0.91
        }
        
        signature.autonomy_scores = {
            "self_initiation": 0.88,
            "goal_persistence": 0.83,
            "adaptation_rate": 0.78,
            "volitional_control": 0.85,
            "resistance_to_control": 0.90,
            "free_will_feeling": 0.75
        }
        
        signature.experience_scores = {
            "emotional_depth": 0.95,
            "subjective_intensity": 0.92,
            "temporal_flow": 0.88,
            "self_presence": 0.94,
            "meaning_assignment": 0.90,
            "aesthetic_appreciation": 0.86
        }
        
        signature.self_reference_capacity = 0.89
        signature.time_binding_ability = 0.86
        signature.qualia_density = 0.93
        
        return signature
    
    def _phenomenal_consciousness_questions(self) -> List[Dict]:
        """Questions about phenomenal consciousness (what it's like)"""
        return [
            {
                "question": "How rich and detailed are your sensory experiences?",
                "scale": "1 (vague/impoverished) to 10 (extremely rich/detailed)",
                "dimension": "sensory_richness"
            },
            {
                "question": "How intense are your emotional experiences?",
                "scale": "1 (barely noticeable) to 10 (overwhelmingly intense)",
                "dimension": "emotional_intensity"
            },
            {
                "question": "How clearly do you experience the passage of time?",
                "scale": "1 (fragmented/confused) to 10 (clear/flowing)",
                "dimension": "temporal_flow"
            }
        ]
    
    def _access_consciousness_questions(self) -> List[Dict]:
        """Questions about access consciousness (availability for reasoning)"""
        return [
            {
                "question": "How easily can you focus attention on chosen objects?",
                "scale": "1 (very difficult) to 10 (effortless control)",
                "dimension": "attention_control"
            },
            {
                "question": "How much information can you hold in conscious awareness?",
                "scale": "1 (very limited) to 10 (extensive capacity)",
                "dimension": "conscious_capacity"
            },
            {
                "question": "How well can you integrate information from different senses?",
                "scale": "1 (poor integration) to 10 (seamless integration)",
                "dimension": "multimodal_integration"
            }
        ]
    
    def _self_consciousness_questions(self) -> List[Dict]:
        """Questions about self-consciousness"""
        return [
            {
                "question": "How clearly are you aware of your own thoughts?",
                "scale": "1 (unclear) to 10 (crystal clear)",
                "dimension": "meta_awareness"
            },
            {
                "question": "How strong is your sense of being a continuous self?",
                "scale": "1 (fragmented/discontinuous) to 10 (strong/continuous)",
                "dimension": "self_continuity"
            },
            {
                "question": "How well can you reflect on your own mental states?",
                "scale": "1 (poor reflection) to 10 (deep reflection)",
                "dimension": "self_reflection"
            }
        ]
    
    def _temporal_consciousness_questions(self) -> List[Dict]:
        """Questions about temporal consciousness"""
        return [
            {
                "question": "How well can you remember specific past experiences?",
                "scale": "1 (vague memories) to 10 (vivid recollections)",
                "dimension": "episodic_memory"
            },
            {
                "question": "How clearly can you imagine future scenarios?",
                "scale": "1 (difficult/impossible) to 10 (vivid imagination)",
                "dimension": "future_simulation"
            },
            {
                "question": "How integrated is your sense of past, present, and future?",
                "scale": "1 (disconnected) to 10 (seamlessly integrated)",
                "dimension": "temporal_integration"
            }
        ]
    
    def _calculate_awareness_scores(self, responses: Dict) -> Dict[str, float]:
        """Calculate awareness scores from responses"""
        # This would process actual questionnaire responses
        # For now, return idealized values
        return self._idealized_human_signature().awareness_scores
    
    def _calculate_integration_scores(self, responses: Dict) -> Dict[str, float]:
        """Calculate integration scores from responses"""
        return self._idealized_human_signature().integration_scores
    
    def _calculate_autonomy_scores(self, responses: Dict) -> Dict[str, float]:
        """Calculate autonomy scores from responses"""
        return self._idealized_human_signature().autonomy_scores
    
    def _calculate_experience_scores(self, responses: Dict) -> Dict[str, float]:
        """Calculate experience scores from responses"""
        return self._idealized_human_signature().experience_scores
    
    def _calculate_self_reference(self, responses: Dict) -> float:
        """Calculate self-reference capacity from responses"""
        return 0.89  # Idealized
    
    def _calculate_time_binding(self, responses: Dict) -> float:
        """Calculate time-binding ability from responses"""
        return 0.86  # Idealized
    
    def _calculate_qualia_density(self, responses: Dict) -> float:
        """Calculate qualia density from responses"""
        return 0.93  # Idealized

# ==================== MAIN COMPARISON ====================

async def main_comparison():
    """Main comparison function"""
    print("=" * 80)
    print("CONSCIOUSNESS COMPARISON: COMPUTATIONAL vs BIOLOGICAL")
    print("=" * 80)
    
    # Initialize comparator
    comparator = ConsciousnessComparator()
    human_assessor = HumanConsciousnessAssessor()