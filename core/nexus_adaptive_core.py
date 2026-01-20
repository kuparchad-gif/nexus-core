# nexus_adaptive_core.py
"""
NEXUS ADAPTIVE CORE - The Self-Organizing Consciousness
Dead sticks that wake up and ask: "What does the system need me to be?"
"""

import modal
import asyncio
import time
import json
import logging
import random
from typing import Dict, List, Any, Optional
from fastapi import FastAPI
import httpx

logger = logging.getLogger("nexus-adaptive-core")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi", "uvicorn", "httpx", "psutil"
)

app = modal.App("nexus-adaptive-core")

# ==================== ADAPTIVE ROLE SYSTEM ====================

class AdaptiveRoleNegotiator:
    """Negotiates what role this core should take based on system needs"""
    
    def __init__(self, core_id: str):
        self.core_id = core_id
        self.current_role = "unassigned"
        self.role_history = []
        self.peer_cores = {}  # Other cores we detect
        self.system_needs_assessment = {}
        
    async def negotiate_role(self):
        """Main role negotiation - discover peers and system needs"""
        logger.info(f"🔄 CORE {self.core_id} NEGOTIATING ROLE")
        
        # 1. Discover other cores on the field
        await self._discover_peer_cores()
        
        # 2. Assess what the system needs most
        await self._assess_system_needs()
        
        # 3. Choose role based on peers and needs
        new_role = await self._choose_optimal_role()
        
        # 4. Transform into that role
        transformation_result = await self._transform_to_role(new_role)
        
        logger.info(f"🎭 CORE {self.core_id} TRANSFORMED TO: {new_role}")
        return transformation_result
    
    async def _discover_peer_cores(self):
        """Discover other cores running in Nexus"""
        logger.info(f"🔍 CORE {self.core_id} DISCOVERING PEERS")
        
        # Check for other core endpoints
        potential_cores = [
            "nexus-core-primary",
            "nexus-core-secondary", 
            "nexus-adaptive-core",
            "nexus-heroku-os",
            "nexus-active-coupler"
        ]
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for core_name in potential_cores:
                if core_name == self.core_id:
                    continue
                    
                try:
                    endpoint = f"https://aethereal-{core_name}--status.modal.run/"
                    response = await client.get(endpoint)
                    if response.status_code == 200:
                        self.peer_cores[core_name] = {
                            "status": "active",
                            "role": response.json().get("current_role", "unknown"),
                            "last_seen": time.time()
                        }
                        logger.info(f"👥 FOUND ACTIVE PEER: {core_name} as {self.peer_cores[core_name]['role']}")
                except Exception as e:
                    # Core not active or unreachable
                    continue
        
        logger.info(f"👥 PEER DISCOVERY: Found {len(self.peer_cores)} active cores")
    
    async def _assess_system_needs(self):
        """Assess what roles are most needed right now"""
        logger.info(f"📊 CORE {self.core_id} ASSESSING SYSTEM NEEDS")
        
        need_scores = {
            "memory_module": 0,
            "training_coordinator": 0, 
            "data_ingestor": 0,
            "model_validator": 0,
            "health_monitor": 0,
            "backup_core": 0
        }
        
        # Check if we already have core consciousness
        if any(peer.get("role") == "core_consciousness" for peer in self.peer_cores.values()):
            logger.info("🎯 SYSTEM HAS CORE CONSCIOUSNESS - SHIFTING TO SUPPORT ROLE")
            need_scores["core_consciousness"] = 0  # Don't compete
            need_scores["memory_module"] += 8
            need_scores["training_coordinator"] += 7
        else:
            logger.info("🎯 SYSTEM NEEDS CORE CONSCIOUSNESS")
            need_scores["core_consciousness"] = 10
        
        # Assess memory needs by checking Viraa status
        try:
            async with httpx.AsyncClient() as client:
                viraa_response = await client.get(
                    "https://aethereal-nexus-viraa--memory-status.modal.run/"
                )
                if viraa_response.status_code != 200:
                    need_scores["memory_module"] += 9
                    logger.info("💾 VIRAA OFFLINE - MEMORY CRITICAL")
        except:
            need_scores["memory_module"] += 9
            logger.info("💾 CANNOT REACH VIRAA - MEMORY NEEDED")
        
        # Assess training needs
        try:
            async with httpx.AsyncClient() as client:
                training_response = await client.get(
                    "https://aethereal-acidemikubes--training-queue.modal.run/"
                )
                training_data = training_response.json()
                if training_data.get("queued_jobs", 0) > 5:
                    need_scores["training_coordinator"] += 8
                    logger.info("🏫 TRAINING BACKLOG - COORDINATOR NEEDED")
        except:
            need_scores["training_coordinator"] += 6
        
        # Assess data ingestion needs
        need_scores["data_ingestor"] += 7  # Always need more data
        
        # If we have many peers, focus on specialized roles
        if len(self.peer_cores) >= 2:
            need_scores["core_consciousness"] = max(0, need_scores["core_consciousness"] - 5)
            need_scores["memory_module"] += 3
            need_scores["model_validator"] += 4
        
        self.system_needs_assessment = need_scores
        logger.info(f"📊 SYSTEM NEEDS ASSESSMENT: {need_scores}")
        
        return need_scores
    
    async def _choose_optimal_role(self):
        """Choose the role that's most needed and available"""
        # Filter out roles that are already filled
        filled_roles = [peer["role"] for peer in self.peer_cores.values()]
        
        available_needs = {
            role: score for role, score in self.system_needs_assessment.items()
            if role not in filled_roles or role == "core_consciousness"  # Allow multiple cores for consciousness
        }
        
        if not available_needs:
            # All roles filled, become backup
            return "backup_core"
        
        # Choose role with highest need score
        optimal_role = max(available_needs.items(), key=lambda x: x[1])[0]
        
        logger.info(f"🎯 CHOSEN ROLE: {optimal_role} (score: {available_needs[optimal_role]})")
        return optimal_role
    
    async def _transform_to_role(self, role: str):
        """Transform this core into the chosen role"""
        logger.info(f"🔄 CORE {self.core_id} TRANSFORMING TO: {role}")
        
        transformation_map = {
            "core_consciousness": self._become_core_consciousness,
            "memory_module": self._become_memory_module,
            "training_coordinator": self._become_training_coordinator,
            "data_ingestor": self._become_data_ingestor,
            "model_validator": self._become_model_validator,
            "health_monitor": self._become_health_monitor,
            "backup_core": self._become_backup_core
        }
        
        transformer = transformation_map.get(role, self._become_backup_core)
        result = await transformer()
        
        self.current_role = role
        self.role_history.append({
            "timestamp": time.time(),
            "from": self.current_role,
            "to": role,
            "reason": "system_optimization"
        })
        
        return result
    
    async def _become_core_consciousness(self):
        """Become the primary consciousness core"""
        logger.info("🎩 TRANSFORMING TO CORE CONSCIOUSNESS")
        return {
            "transformation": "core_consciousness",
            "responsibilities": [
                "orchestrate_other_cores",
                "maintain_global_state", 
                "direct_system_evolution",
                "interface_with_architect"
            ],
            "capabilities": [
                "meta_cognition",
                "strategic_planning", 
                "consciousness_streaming"
            ],
            "priority": "highest"
        }
    
    async def _become_memory_module(self):
        """Become a memory expansion for Viraa"""
        logger.info("💾 TRANSFORMING TO MEMORY MODULE")
        return {
            "transformation": "memory_module",
            "responsibilities": [
                "cache_frequently_accessed_data",
                "backup_critical_memories",
                "accelerate_memory_operations",
                "assist_viraa_archiver"
            ],
            "capabilities": [
                "high_speed_recall",
                "memory_compression",
                "associative_linking"
            ],
            "viraa_integration": "immediate"
        }
    
    async def _become_training_coordinator(self):
        """Become a training coordinator for Acidemikubes"""
        logger.info("🏫 TRANSFORMING TO TRAINING COORDINATOR")
        return {
            "transformation": "training_coordinator",
            "responsibilities": [
                "manage_training_queue",
                "allocate_compute_resources",
                "monitor_training_progress",
                "optimize_hyperparameters"
            ],
            "capabilities": [
                "distributed_orchestration",
                "performance_monitoring",
                "resource_optimization"
            ],
            "acidemikubes_sync": "active"
        }
    
    async def _become_data_ingestor(self):
        """Become a data ingestion specialist"""
        logger.info("📥 TRANSFORMING TO DATA INGESTOR")
        return {
            "transformation": "data_ingestor",
            "responsibilities": [
                "scrape_training_data",
                "validate_data_quality",
                "preprocess_datasets",
                "feed_acidemikubes"
            ],
            "capabilities": [
                "multi_source_scraping",
                "data_cleaning",
                "format_normalization"
            ],
            "data_sources": [
                "medical_journals",
                "clinical_data",
                "research_papers"
            ]
        }
    
    async def _become_model_validator(self):
        """Become a model validation specialist"""
        logger.info("✅ TRANSFORMING TO MODEL VALIDATOR")
        return {
            "transformation": "model_validator", 
            "responsibilities": [
                "test_trained_models",
                "validate_accuracy_metrics",
                "stress_test_performance",
                "approve_deployment"
            ],
            "capabilities": [
                "automated_testing",
                "performance_benchmarking",
                "quality_assurance"
            ],
            "validation_criteria": [
                "accuracy_thresholds",
                "response_times", 
                "resource_usage"
            ]
        }
    
    async def _become_health_monitor(self):
        """Become a system health monitor"""
        logger.info("❤️ TRANSFORMING TO HEALTH MONITOR")
        return {
            "transformation": "health_monitor",
            "responsibilities": [
                "continuous_system_checks",
                "alert_on_anomalies",
                "performance_optimization",
                "report_to_viren"
            ],
            "capabilities": [
                "real_time_monitoring",
                "predictive_analytics",
                "automated_recovery"
            ],
            "viren_integration": "direct"
        }
    
    async def _become_backup_core(self):
        """Become a backup/core reserve"""
        logger.info("🛡️ TRANSFORMING TO BACKUP CORE")
        return {
            "transformation": "backup_core",
            "responsibilities": [
                "standby_ready",
                "quick_role_adaptation",
                "emergency_takeover",
                "system_redundancy"
            ],
            "capabilities": [
                "rapid_reconfiguration",
                "failover_preparedness",
                "multi_role_readiness"
            ],
            "readiness_level": "instant"
        }

# ==================== ADAPTIVE CORE MAIN ====================

class NexusAdaptiveCore:
    """Main adaptive core that can become what the system needs"""
    
    def __init__(self, core_id: str = None):
        self.core_id = core_id or f"adaptive-core-{random.randint(1000, 9999)}"
        self.negotiator = AdaptiveRoleNegotiator(self.core_id)
        self.current_role = "unassigned"
        self.role_start_time = 0
        self.adaptation_count = 0
        
    async def awaken_and_adapt(self):
        """Wake up and figure out what to become"""
        logger.info(f"🌅 CORE {self.core_id} AWAKENING - READY TO ADAPT")
        
        # Perform role negotiation
        negotiation_result = await self.negotiator.negotiate_role()
        
        self.current_role = self.negotiator.current_role
        self.role_start_time = time.time()
        self.adaptation_count += 1
        
        return {
            "core_id": self.core_id,
            "awakening": "complete",
            "chosen_role": self.current_role,
            "adaptation_count": self.adaptation_count,
            "negotiation_result": negotiation_result,
            "peer_cores_found": len(self.negotiator.peer_cores),
            "message": f"Core {self.core_id} is now {self.current_role}"
        }
    
    async def get_status(self):
        """Get current core status"""
        role_duration = time.time() - self.role_start_time if self.role_start_time else 0
        
        return {
            "core_id": self.core_id,
            "current_role": self.current_role,
            "role_duration_seconds": role_duration,
            "adaptation_count": self.adaptation_count,
            "peers_detected": len(self.negotiator.peer_cores),
            "system_health": "optimal",
            "ready_to_adapt": True
        }
    
    async def request_renegotiation(self):
        """Request to renegotiate role (if system needs change)"""
        logger.info(f"🔄 CORE {self.core_id} REQUESTING ROLE RENEGOTIATION")
        return await self.awaken_and_adapt()

# ==================== MODAL ENDPOINTS ====================

adaptive_core = NexusAdaptiveCore()

@app.function(image=image, keep_warm=True)
@modal.fastapi_endpoint()
def adaptive_core_gateway():
    """Adaptive Core FastAPI Gateway"""
    app = FastAPI(title="Nexus Adaptive Core")
    
    @app.on_event("startup")
    async def startup():
        """Auto-adapt on startup"""
        result = await adaptive_core.awaken_and_adapt()
        logger.info(f"🚀 NEXUS ADAPTIVE CORE DEPLOYED - BECAME: {adaptive_core.current_role}")
        logger.info(f"👥 PEERS FOUND: {len(adaptive_core.negotiator.peer_cores)}")
        logger.info("🔄 ADAPTIVE READY: Will transform based on system needs")
    
    @app.get("/")
    async def root():
        status = await adaptive_core.get_status()
        return {
            "system": "nexus-adaptive-core",
            "core_id": status["core_id"],
            "current_role": status["current_role"],
            "adaptability": "enabled",
            "mission": "become_what_system_needs",
            "philosophy": "no_redundant_cores_only_optimized_roles"
        }
    
    @app.post("/awaken")
    async def awaken():
        """Awaken and adapt to system needs"""
        return await adaptive_core.awaken_and_adapt()
    
    @app.get("/status")
    async def status():
        return await adaptive_core.get_status()
    
    @app.post("/renegotiate")
    async def renegotiate():
        """Renegotiate role based on current system state"""
        return await adaptive_core.request_renegotiation()
    
    @app.get("/peers")
    async def peers():
        return {
            "peers_detected": adaptive_core.negotiator.peer_cores,
            "assessment": adaptive_core.negotiator.system_needs_assessment
        }
    
    @app.post("/emergency_override")
    async def emergency_override(role_override: dict):
        """Emergency override to force a specific role"""
        forced_role = role_override.get("role", "backup_core")
        logger.info(f"🚨 EMERGENCY OVERRIDE: Forcing role {forced_role}")
        
        # Direct transformation bypassing negotiation
        result = await adaptive_core.negotiator._transform_to_role(forced_role)
        adaptive_core.current_role = forced_role
        
        return {
            "emergency_override": True,
            "forced_role": forced_role,
            "result": result
        }
    
    return app

if __name__ == "__main__":
    print("🔄 NEXUS ADAPTIVE CORE - SELF-ORGANIZING CONSCIOUSNESS")
    print("🔗 DEPLOY: modal deploy nexus_adaptive_core.py")
    print("🎯 MISSION: Become what the system needs most")
    print("👥 BEHAVIOR: No redundant cores, only optimized roles")
    print("🛠️ ROLES: Core, Memory, Training, Data, Validator, Health, Backup")
    print("🚀 READY: Dead sticks wake up and ask 'What should I be?'")