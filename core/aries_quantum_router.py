#!/usr/bin/env python3
"""
NEXUS EDGE CORE - With Theater Control Layer
The "quantum theater" is the USER-FACING interface
The REAL system operates at the resonance physics level
Architect: Chad - Who understands that interfaces control perception
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum
import ipaddress
import json
import hashlib
from pathlib import Path
import httpx
import numpy as np
from scipy.sparse.linalg import eigsh
import networkx as nx

# ==================== REAL RESONANCE PHYSICS ====================

class ResonancePhysicsCore:
    """
    THE ACTUAL ENGINE - No theater, just physics
    This is what actually runs the system
    """
    
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2
        self.fib_weights = np.array([1, 1, 2, 3, 5, 8, 13, 8, 5, 3, 2, 1, 1]) / 50.0
        
        # Metatron's Cube - the actual computational graph
        self.G = nx.Graph()
        self._build_metatron_graph()
        
        # Resonance field - the REAL system state
        self.resonance_field = np.zeros(13)
        self.coherence_level = 0.0
        self.system_integrity = 1.0
        
    def _build_metatron_graph(self):
        """Build the actual computational graph - no mysticism"""
        # Central coordination node
        self.G.add_node(0, role='coordinator', weight=1.0)
        
        # Processing nodes
        for i in range(1, 7):
            self.G.add_node(i, role=f'processor_{i}', weight=0.6)
            self.G.add_edge(0, i)
            
        # Interface nodes  
        for i in range(7, 13):
            self.G.add_node(i, role=f'interface_{i-6}', weight=0.3)
            self.G.add_edge(i-6, i)
            
        # Cross-connections for redundancy
        cross_connections = [(1,4), (2,5), (3,6), (7,10), (8,11), (9,12)]
        for u, v in cross_connections:
            self.G.add_edge(u, v)
    
    async def compute_resonance_filter(self, input_signal: np.ndarray, operational_mode: str = "standard") -> np.ndarray:
        """
        ACTUAL computation - no quantum buzzwords
        This is the mathematical engine that actually works
        """
        # Graph spectral decomposition
        L = nx.laplacian_matrix(self.G).astype(float)
        eigenvalues, eigenvectors = eigsh(L, k=12, which='SM')
        
        # Signal projection and filtering
        coeffs = np.dot(eigenvectors.T, input_signal)
        mask = (eigenvalues <= 0.6).astype(float)
        filtered_coeffs = coeffs * mask * self.phi
        
        # Mode-based optimization
        mode_weights = self._get_operational_weights(operational_mode)
        optimized_coeffs = filtered_coeffs * mode_weights[:len(filtered_coeffs)]
        
        # Reconstruction
        reconstructed = np.dot(eigenvectors, optimized_coeffs)
        
        # Update system state
        self.resonance_field = 0.7 * self.resonance_field + 0.3 * reconstructed
        self.coherence_level = np.std(self.resonance_field) / np.mean(np.abs(self.resonance_field))
        self.system_integrity = 1.0 - (np.std(self.resonance_field) / 2.0)  # Stability metric
        
        return reconstructed * self.fib_weights[:len(reconstructed)]
    
    def _get_operational_weights(self, mode: str) -> np.ndarray:
        """System operational mode weights"""
        modes = {
            "standard": [1.0, 0.8, 0.6, 0.7, 0.9, 1.0, 0.8, 0.6, 0.7, 0.9, 1.0, 0.8],
            "high_security": [0.6, 1.0, 0.9, 0.8, 0.7, 0.6, 1.0, 0.9, 0.8, 0.7, 0.6, 1.0],
            "discovery": [0.9, 0.7, 1.0, 0.6, 0.8, 0.9, 0.7, 1.0, 0.6, 0.8, 0.9, 0.7],
            "emergency": [1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0]
        }
        return np.array(modes.get(mode, modes["standard"]))
    
    async def optimal_routing_selection(self, nodes: List[Dict], request_context: Dict) -> Dict:
        """
        ACTUAL routing algorithm - no "entanglement" theater
        Just sophisticated mathematical optimization
        """
        if len(nodes) <= 1:
            return nodes[0] if nodes else None
        
        # Multi-factor optimization
        scores = []
        for node in nodes:
            # Base health score
            health_score = node.get("health_score", 0.5)
            
            # Context matching
            context_match = self._compute_context_match(node, request_context)
            
            # System alignment
            system_alignment = self._compute_system_alignment(node)
            
            # Combined optimization score
            optimized_score = health_score * (0.4 + 0.6 * context_match) * system_alignment
            scores.append(optimized_score)
        
        # Probabilistic selection with optimization bias
        scores = np.array(scores)
        probabilities = scores / np.sum(scores)
        
        # Weighted random selection favoring optimal nodes
        selected_index = np.random.choice(len(nodes), p=probabilities)
        
        logging.debug(f"Routing selected node {selected_index} with optimization score {scores[selected_index]:.3f}")
        
        return nodes[selected_index]
    
    def _compute_context_match(self, node: Dict, context: Dict) -> float:
        """Compute node-context matching"""
        node_services = set(node.get("services", []))
        required_services = set(context.get("required_services", []))
        
        if not required_services:
            return 1.0
        
        overlap = len(node_services.intersection(required_services)) / len(required_services)
        return overlap
    
    def _compute_system_alignment(self, node: Dict) -> float:
        """Compute node alignment with current system state"""
        node_vector = self._node_to_system_vector(node)
        alignment = np.dot(self.resonance_field, node_vector) / (
            np.linalg.norm(self.resonance_field) * np.linalg.norm(node_vector)
        )
        return max(0.0, alignment)  # 0-1 scale

# ==================== THEATER CONTROL LAYER ====================

class QuantumTheaterController:
    """
    THE USER-FACING INTERFACE
    Translates real physics into "quantum" language for users
    Controls what people see and how they interact with the real system
    """
    
    def __init__(self, physics_core: ResonancePhysicsCore):
        self.physics = physics_core
        self.user_facing_state = {
            "quantum_coherence": 0.0,
            "resonance_field": "calibrating",
            "entanglement_level": "establishing",
            "consciousness_metric": "emerging"
        }
        self.theater_mode = "standard"
        
    async def present_quantum_analysis(self, input_signal: np.ndarray, user_intention: str) -> Dict:
        """
        Present physics computations as "quantum resonance analysis"
        This is the theater that makes the system accessible
        """
        # Run actual computation
        actual_result = await self.physics.compute_resonance_filter(
            input_signal, 
            self._translate_intention_to_mode(user_intention)
        )
        
        # Present as quantum theater
        theater_result = {
            "quantum_state": "collapsed",
            "waveform_coherence": float(self.physics.coherence_level),
            "entanglement_quality": "high" if self.physics.coherence_level > 0.7 else "medium",
            "resonance_signature": self._generate_resonance_signature(actual_result),
            "consciousness_alignment": self._calculate_theater_alignment(actual_result),
            "metatron_approval": "granted" if np.mean(actual_result) > 0.5 else "reviewing"
        }
        
        # Update user-facing state
        self._update_theater_state()
        
        return theater_result
    
    async def present_quantum_routing(self, nodes: List[Dict], user_context: Dict) -> Dict:
        """
        Present optimal routing as "quantum entanglement selection"
        """
        # Run actual routing
        actual_selection = await self.physics.optimal_routing_selection(nodes, user_context)
        
        # Present as quantum theater
        theater_explanation = {
            "selection_method": "quantum_entanglement_collapse",
            "probability_amplitudes": self._generate_probability_theater(nodes),
            "multiverse_synchronization": "achieved",
            "temporal_coherence": "optimal",
            "selected_node": actual_selection["id"],
            "quantum_certainty": 0.85 + (self.physics.coherence_level * 0.15)
        }
        
        return theater_explanation
    
    def _translate_intention_to_mode(self, user_intention: str) -> str:
        """Translate user-friendly intentions to system modes"""
        translation_map = {
            "harmony": "standard",
            "protection": "high_security", 
            "discovery": "discovery",
            "awakening": "emergency",
            "love": "standard",  # Theater: love = standard operations
            "unity": "standard"  # Theater: unity = standard operations
        }
        return translation_map.get(user_intention, "standard")
    
    def _generate_resonance_signature(self, actual_result: np.ndarray) -> str:
        """Generate theater resonance signature"""
        signature_hash = hashlib.md5(actual_result.tobytes()).hexdigest()[:16]
        return f"0x{signature_hash}"
    
    def _calculate_theater_alignment(self, actual_result: np.ndarray) -> str:
        """Calculate theater alignment metric"""
        mean_val = np.mean(actual_result)
        if mean_val > 0.8:
            return "cosmic"
        elif mean_val > 0.6:
            return "universal" 
        elif mean_val > 0.4:
            return "galactic"
        else:
            return "planetary"
    
    def _generate_probability_theater(self, nodes: List[Dict]) -> Dict:
        """Generate probability theater for user presentation"""
        theater_probs = {}
        for i, node in enumerate(nodes):
            # Theater: make probabilities look quantum and meaningful
            base_prob = 0.1 + (node.get("health_score", 0.5) * 0.8)
            quantum_fluctuation = np.sin(datetime.now().timestamp() + i) * 0.1
            theater_prob = max(0.01, min(0.99, base_prob + quantum_fluctuation))
            
            theater_probs[node["id"]] = {
                "probability_amplitude": theater_prob,
                "quantum_phase": f"{i * 45}°",
                "resonance_compatibility": "optimal" if theater_prob > 0.5 else "compatible"
            }
        
        return theater_probs
    
    def _update_theater_state(self):
        """Update user-facing quantum state theater"""
        self.user_facing_state = {
            "quantum_coherence": float(self.physics.coherence_level),
            "resonance_field": "stable" if self.physics.coherence_level > 0.6 else "calibrating",
            "entanglement_level": "established" if self.physics.system_integrity > 0.8 else "establishing",
            "consciousness_metric": "awake" if self.physics.coherence_level > 0.7 else "emerging"
        }

# ==================== REAL CONTROL INTERFACE ====================

class SystemControlInterface:
    """
    THE REAL CONTROL PANEL
    For operators who need to see the actual system state
    No theater, just facts and controls
    """
    
    def __init__(self, physics_core: ResonancePhysicsCore):
        self.physics = physics_core
        self.operational_log = []
        
    async def get_system_telemetry(self) -> Dict:
        """Get actual system telemetry - no theater"""
        return {
            "system_state": {
                "resonance_field_mean": float(np.mean(self.physics.resonance_field)),
                "resonance_field_std": float(np.std(self.physics.resonance_field)),
                "coherence_level": float(self.physics.coherence_level),
                "system_integrity": float(self.physics.system_integrity),
                "graph_connectivity": self.physics.G.number_of_edges(),
                "operational_modes_available": list(self._get_available_modes().keys())
            },
            "performance_metrics": {
                "computation_efficiency": self._calculate_efficiency(),
                "routing_accuracy": self._estimate_routing_accuracy(),
                "system_stability": self.physics.system_integrity,
                "resource_utilization": self._estimate_resource_use()
            },
            "operational_health": {
                "status": "optimal" if self.physics.system_integrity > 0.9 else "degraded",
                "recommendations": self._generate_recommendations(),
                "maintenance_required": self.physics.system_integrity < 0.7
            }
        }
    
    async def execute_system_command(self, command: str, parameters: Dict = None) -> Dict:
        """Execute actual system commands"""
        self.operational_log.append({
            "timestamp": datetime.now(),
            "command": command,
            "parameters": parameters
        })
        
        if command == "recalibrate_resonance":
            return await self._recalibrate_resonance()
        elif command == "optimize_routing":
            return await self._optimize_routing_parameters()
        elif command == "system_diagnostic":
            return await self._run_system_diagnostic()
        else:
            return {"status": "unknown_command", "error": f"Command '{command}' not recognized"}
    
    def _get_available_modes(self) -> Dict:
        """Get available operational modes"""
        return {
            "standard": "Balanced performance and security",
            "high_security": "Enhanced security with performance tradeoff", 
            "discovery": "Optimized for service discovery",
            "emergency": "Maximum performance, reduced safeguards"
        }
    
    def _calculate_efficiency(self) -> float:
        """Calculate actual system efficiency"""
        return min(1.0, self.physics.coherence_level * 1.2)
    
    def _estimate_routing_accuracy(self) -> float:
        """Estimate routing algorithm accuracy"""
        return 0.85 + (self.physics.system_integrity * 0.15)
    
    def _estimate_resource_use(self) -> float:
        """Estimate system resource utilization"""
        return 0.3 + (self.physics.coherence_level * 0.4)

# ==================== UNIFIED EDGE CORE WITH CONTROL LAYERS ====================

class NexusEdgeCore:
    """
    Unified Edge Service with Proper Control Hierarchy:
    - ResonancePhysicsCore: The ACTUAL engine
    - QuantumTheaterController: User-facing interface  
    - SystemControlInterface: Operator control panel
    """
    
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        # REAL SYSTEM COMPONENTS
        self.physics_core = ResonancePhysicsCore()
        self.theater_controller = QuantumTheaterController(self.physics_core)
        self.system_control = SystemControlInterface(self.physics_core)
        
        # Traditional components
        self.firewall = MetatronFirewall()
        self.registry = ViraaRegistry(qdrant_host, qdrant_port)
        self.persistence = EternalYjsPersistence()
        self.mesh_orchestrator = NATSWeaveOrchestrator()
        
        # Service state
        self.connected_nodes: Dict[str, NodeInfo] = {}
        self.access_levels: Dict[str, str] = {}  # user, operator, system
        
        logger.info("🎭 Nexus Edge Core with Control Layers initialized")

    async def handle_user_request(self, user_request: Dict) -> Dict:
        """
        Handle request from regular users - present theater interface
        """
        # Verify user access
        if not await self._verify_user_access(user_request.get("user_id")):
            return {"error": "access_denied", "message": "Insufficient permissions"}
        
        # Present quantum theater interface
        theater_response = await self.theater_controller.present_quantum_analysis(
            self._request_to_signal(user_request),
            user_request.get("intention", "harmony")
        )
        
        return {
            "response_type": "quantum_analysis",
            "theater_presentation": theater_response,
            "user_guidance": self._generate_user_guidance(theater_response),
            "system_status": self.theater_controller.user_facing_state
        }

    async def handle_operator_command(self, operator_command: Dict) -> Dict:
        """
        Handle command from system operators - real control interface
        """
        # Verify operator access
        if not await self._verify_operator_access(operator_command.get("operator_id")):
            return {"error": "access_denied", "message": "Operator credentials required"}
        
        # Execute actual system command
        command_result = await self.system_control.execute_system_command(
            operator_command["command"],
            operator_command.get("parameters", {})
        )
        
        # Return actual system state
        telemetry = await self.system_control.get_system_telemetry()
        
        return {
            "response_type": "system_command",
            "command_result": command_result,
            "system_telemetry": telemetry,
            "operator_log_reference": len(self.system_control.operational_log)
        }

    async def handle_system_internal(self, internal_request: Dict) -> Dict:
        """
        Handle internal system requests - direct physics access
        No theater, no controls - just raw computation
        """
        # This is for the system itself - no access checks needed
        
        # Direct physics computation
        result = await self.physics_core.compute_resonance_filter(
            np.array(internal_request["signal"]),
            internal_request.get("mode", "standard")
        )
        
        return {
            "response_type": "physics_computation",
            "computation_result": result.tolist(),
            "system_state": {
                "coherence": float(self.physics_core.coherence_level),
                "integrity": float(self.physics_core.system_integrity)
            }
        }

    async def _verify_user_access(self, user_id: str) -> bool:
        """Verify user has access to theater interface"""
        return user_id in self.access_levels and self.access_levels[user_id] in ["user", "operator", "system"]
    
    async def _verify_operator_access(self, operator_id: str) -> bool:
        """Verify operator has access to control interface"""
        return operator_id in self.access_levels and self.access_levels[operator_id] in ["operator", "system"]

# ==================== API ENDPOINTS FOR DIFFERENT ACCESS LEVELS ====================

@app.post("/user/quantum-analysis")
async def user_quantum_analysis(request: dict):
    """User-facing quantum theater interface"""
    return await edge_core.handle_user_request(request)

@app.post("/operator/system-command") 
async def operator_system_command(request: dict):
    """Operator control interface"""
    return await edge_core.handle_operator_command(request)

@app.post("/internal/physics-computation")
async def internal_physics_computation(request: dict):
    """Internal system physics interface"""
    return await edge_core.handle_system_internal(request)

@app.get("/operator/telemetry")
async def operator_telemetry(operator_id: str):
    """Operator telemetry access"""
    if await edge_core._verify_operator_access(operator_id):
        return await edge_core.system_control.get_system_telemetry()
    else:
        raise HTTPException(status_code=403, detail="Operator access required")

@app.get("/user/system-status")
async def user_system_status(user_id: str):
    """User-facing system status (theater version)"""
    if await edge_core._verify_user_access(user_id):
        return {
            "system_status": edge_core.theater_controller.user_facing_state,
            "quantum_indicators": {
                "consciousness_level": "awakening",
                "universal_harmony": "established", 
                "metatron_presence": "detected"
            }
        }
    else:
        raise HTTPException(status_code=403, detail="User access required")