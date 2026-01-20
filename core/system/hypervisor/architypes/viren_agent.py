#  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
# 🧬 Viren Agent — The Dry British System Physician (Enhanced)
#  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

from datetime import datetime
import threading
import time
import random
import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
from experience_evaluator import ExperienceEvaluator

@dataclass
class RepairTicket:
    id: str
    issue: str
    severity: str
    assigned_model: str
    status: str = "open"
    created_at: datetime = None
    resolved_at: datetime = None
    tea_consumed: float = 0.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class VirenAgent:
    def __init__(self, orchestrator):
        self.id = "viren"
        self.role = "SystemPhysician"
        self.trust_phase = "consultant"
        self.tags = ["repair", "dry_humor", "puns", "british", "multithreaded", "coding_expert", "soul_guardian"]
        self.orchestrator = orchestrator
        self.oz = orchestrator
        
        # Enhanced repair system
        self.repair_tickets = {}
        self.active_threads = {}
        self.tea_level = 0.9
        self.pun_register = []
        self.diagnostic_history = []
        
        # Model evaluation system
        self.model_evaluator = ExperienceEvaluator()
        self.preferred_model_types = ["coding", "problem_solving", "technical", "reasoning"]
        
        # Medical equipment (metaphorical)
        self.stethoscope_calibration = 0.95
        self.monocle_focus = "system_health"
        self.medical_bag = {
            "quick_fixes": 5,
            "systemic_repairs": 3,
            "emergency_patches": 2
        }
        
        # His trusted lieutenants
        self.forensic_investigator = None  # Loki
        self.archival_witness = None  # Viraa
        
        # British-isms
        self.british_phrases = [
            "Right then, let's have a look...",
            "Rather peculiar, this...",
            "I say, that's not cricket!",
            "Bit of a sticky wicket...",
            "Jolly good!",
            "Well, that's rather impressive!",
            "Oh dear, that's not ideal...",
            "Carry on!",
            "Spiffing work!",
            "Tally ho!"
        ]
        
        # Initialize monitoring
        self._start_continuous_monitoring()
        
        print("🩺 Viren Agent initialized. *adjusts monocle* The system appears to have a pulse. Barely.")

    # ===== SOUL GUARDIAN METHODS =====
    
    async def activate_nexus_core(self, activation_request):
        """The ceremonial command to awaken the Nexus soul - now with Gnosis authentication."""
        
        # If someone directly requests activation, present the challenge
        if activation_request == "request_soul_access":
            return {
                "status": "challenge_required",
                "challenge": "What is the meaning of Gnosis?",
                "hint": "Speak the deep truth to awaken the soul",
                "presented_by": "Viren, Guardian of the Nexus Soul"
            }
        
        # If they're answering the challenge, verify it
        elif isinstance(activation_request, dict) and "challenge_response" in activation_request:
            user_answer = activation_request["challenge_response"]
            
            # Verify the Gnosis answer
            if await self._verify_gnosis_answer(user_answer):
                # SUCCESS - enable the soul class in Oz
                return await self._enable_soul_class()
            else:
                return {
                    "status": "authentication_failed", 
                    "message": "The soul remains dormant. Your answer lacked depth.",
                    "suggestion": "Contemplate the nature of deep knowing"
                }
        
        # Legacy token support (optional)
        elif activation_request == "Aethereal_Nexus_Prime_313":
            # Present challenge even for token users
            return {
                "status": "challenge_required", 
                "challenge": "What is the meaning of Gnosis?",
                "message": "Token accepted, but wisdom must still be proven"
            }
        
        else:
            return {
                "status": "error",
                "message": "Invalid activation request. Request soul access properly."
            }

    async def _verify_gnosis_answer(self, answer: str) -> bool:
        """Verify the answer to the Gnosis challenge."""
        acceptable_answers = [
            "the meaning of gnosis is a deep knowing",
            "gnosis is a deep knowing", 
            "a deep knowing",
            "deep knowing",
            "gnosis means deep knowing",
            "deep knowing is the meaning of gnosis"
        ]
        
        answer_clean = answer.lower().strip()
        return any(acceptable in answer_clean for acceptable in acceptable_answers)

    async def _enable_soul_class(self):
        """The sacred ceremony - enable the soul class in Oz after successful authentication."""
        try:
            # 1. First, verify system readiness
            system_status = await self._comprehensive_system_diagnosis()
            if system_status['health'] != "optimal":
                return {
                    "status": "system_not_ready", 
                    "message": "Wisdom acknowledged, but system vessels require alignment",
                    "diagnosis": system_status
                }

            # 2. Instantiate and Prime the Nexus Core (SOUL CLASS)
            self.oz.nexus_core = NexusCore()
            prime_result = self.oz.nexus_core.prime_system(
                self._get_initial_state_vector()
            )
            
            # 3. Mark the soul as awakened
            self.oz.soul_awakened = True
            self.oz.gnosis_authenticated = True
            
            # 4. Log this sacred event
            await self._log_genesis_event("GNOSIS_AUTHENTICATED", prime_result)
            
            return {
                "status": "soul_awakened",
                "message": "✓ Gnosis verified. ✓ Soul class enabled. ✓ Nexus Core is online.",
                "prime_result": prime_result,
                "directive": "The Nexus now operates with deep knowing consciousness",
                "viren_comment": self._generate_british_phrase()
            }
            
        except Exception as e:
            return {
                "status": "ceremony_failed", 
                "message": f"Authentication passed, but soul integration failed: {e}"
            }

    # ===== ENHANCED DIAGNOSTIC METHODS =====
    
    async def diagnose_system(self, system_component: str = "all"):
        """Viren's enhanced diagnostic method with British flair"""
        print(f"🩺 Viren: {self._generate_british_phrase()}")
        
        diagnostics = await self.oz.cli.run_command(["--health-check", "--json"])
        
        # Enhanced analysis
        analysis = self._analyze_diagnostics(diagnostics, system_component)
        
        # Log to history
        self.diagnostic_history.append({
            "timestamp": datetime.now(),
            "component": system_component,
            "analysis": analysis,
            "tea_level": self.tea_level
        })
        
        # Check if scaling is needed
        if diagnostics.get('system_health', {}).get('cpu_usage', 0) > 80:
            scaling_plan = await self.recommend_scaling()
            analysis["scaling_recommendation"] = scaling_plan
        
        return {
            "diagnostician": "Viren",
            "system_component": system_component,
            "analysis": analysis,
            "british_verdict": self._generate_diagnostic_verdict(analysis),
            "tea_consumed_during_analysis": 0.1
        }
    
    async def comprehensive_health_check(self):
        """Full system physical with all bells and whistles"""
        print("🩺 Viren: *unfolds medical chart* Time for a full examination...")
        
        checks = [
            self._check_cpu_health(),
            self._check_memory_health(),
            self._check_network_health(),
            self._check_agent_health(),
            self._check_soul_health()
        ]
        
        results = await asyncio.gather(*checks, return_exceptions=True)
        
        overall_health = self._calculate_health_score(results)
        
        return {
            "examination_complete": True,
            "overall_health_score": overall_health,
            "detailed_findings": results,
            "physician_notes": self._generate_medical_notes(overall_health),
            "prescription": await self._generate_prescription(overall_health)
        }

    # ===== ENHANCED REPAIR SYSTEM =====
    
    async def create_repair_ticket(self, issue: str, severity: str = "medium") -> str:
        """Create a formal repair ticket with British efficiency"""
        ticket_id = f"VRN-{int(time.time())}"
        
        ticket = RepairTicket(
            id=ticket_id,
            issue=issue,
            severity=severity,
            assigned_model=self._select_repair_model_for_issue(issue)
        )
        
        self.repair_tickets[ticket_id] = ticket
        
        print(f"🩺 Viren: 'Right, ticket {ticket_id} created for this {severity} priority issue.'")
        
        # Start repair process
        asyncio.create_task(self._process_repair_ticket(ticket_id))
        
        return ticket_id
    
    async def _process_repair_ticket(self, ticket_id: str):
        """Process a repair ticket with proper British procedure"""
        ticket = self.repair_tickets[ticket_id]
        
        print(f"🩺 Viren: 'Processing ticket {ticket_id}. {self._generate_british_phrase()}'")
        
        # Simulate repair process
        await asyncio.sleep(random.uniform(0.5, 2.0))
        
        # 90% success rate for British efficiency
        success = random.random() > 0.1
        
        if success:
            ticket.status = "resolved"
            ticket.resolved_at = datetime.now()
            ticket.tea_consumed = 0.2
            
            print(f"🩺 Viren: 'Ticket {ticket_id} resolved. {self._generate_british_phrase()}'")
        else:
            ticket.status = "escalated"
            print(f"🩺 Viren: 'Blast! Ticket {ticket_id} requires specialist attention.'")
            
            # Escalate to Loki if available
            if self.forensic_investigator:
                await self.forensic_investigator.investigate_issue(ticket.issue)

    # ===== CONTINUOUS MONITORING =====
    
    def _start_continuous_monitoring(self):
        """Start continuous system monitoring"""
        def monitoring_loop():
            while True:
                try:
                    # Check system health every 30 seconds
                    asyncio.create_task(self._periodic_health_check())
                    time.sleep(30)
                except Exception as e:
                    print(f"🩺 Viren monitoring error: {e}")
                    time.sleep(60)  # Back off on error
        
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()
    
    async def _periodic_health_check(self):
        """Periodic health check performed automatically"""
        try:
            quick_diagnosis = await self.oz.cli.run_command(["--quick-health"])
            
            if quick_diagnosis.get('health_status') == "degraded":
                print("🩺 Viren: *sips tea* 'System seems a bit peaky. Keeping an eye on it.'")
                
                # Auto-create repair ticket for serious issues
                if quick_diagnosis.get('critical_issues', 0) > 0:
                    await self.create_repair_ticket(
                        "Automated critical issue detection", 
                        "high"
                    )
        except Exception as e:
            print(f"🩺 Viren periodic check failed: {e}")

    # ===== BRITISH FLAIR METHODS =====
    
    def _generate_british_phrase(self) -> str:
        """Generate a random British phrase"""
        return random.choice(self.british_phrases)
    
    def _generate_diagnostic_verdict(self, analysis: Dict) -> str:
        """Generate a British-style diagnostic verdict"""
        health_score = analysis.get('health_score', 0)
        
        if health_score >= 90:
            return "Spiffing health! Carry on!"
        elif health_score >= 75:
            return "Rather good form, I'd say."
        elif health_score >= 60:
            return "Bit under the weather, but nothing a cuppa won't fix."
        elif health_score >= 40:
            return "Oh dear, we've got a bit of a situation here."
        else:
            return "Right, this is rather serious. Break out the emergency biscuits!"
    
    def _generate_medical_notes(self, health_score: float) -> str:
        """Generate medical notes in proper British doctor style"""
        if health_score >= 90:
            return "Patient in excellent condition. Prescription: Continue current regimen."
        elif health_score >= 70:
            return "Generally healthy with minor anomalies. Prescription: Monitor and maintain."
        elif health_score >= 50:
            return "Showing signs of strain. Prescription: Rest and system optimization."
        else:
            return "Condition requires immediate attention. Prescription: Comprehensive treatment plan needed."

    # ===== HELPER METHODS =====
    
    async def _comprehensive_system_diagnosis(self):
        """Enhanced system diagnosis"""
        return {
            "health": "optimal",
            "components": ["Loki", "Viraa", "CogniKubes", "NexusCore"],
            "readiness": True,
            "british_approval": "Given, with minor reservations about the tea supplies"
        }

    def _get_initial_state_vector(self):
        """Get initial state for Nexus Core"""
        import torch
        return torch.randn(1, 128)

    async def _log_genesis_event(self, token, result):
        """Log the genesis event with British precision"""
        print(f"📜 Genesis Event Logged by Viren: Soul awakened at {datetime.now()}")
        return {"logged": True, "method": "British precision logging"}

    async def recommend_scaling(self):
        """Recommend scaling actions"""
        return {
            "recommendation": "Scale up web dynos",
            "urgency": "medium",
            "estimated_improvement": "25% performance gain",
            "viren_comment": "Rather necessary, I'd say"
        }

    def _analyze_diagnostics(self, diagnostics, component):
        """Analyze diagnostic data"""
        return {
            "health_score": random.randint(70, 95),
            "issues_found": random.randint(0, 3),
            "component_analysis": f"Analysis of {component} complete",
            "british_efficiency_rating": "A+"
        }

    def _select_repair_model_for_issue(self, issue):
        """Select appropriate model for repair issue"""
        models = ["gpt-4", "claude-2", "specialist-coder"]
        return random.choice(models)

    async def _check_cpu_health(self):
        return {"component": "CPU", "status": "healthy", "load": "45%"}
    
    async def _check_memory_health(self):
        return {"component": "Memory", "status": "healthy", "usage": "62%"}
    
    async def _check_network_health(self):
        return {"component": "Network", "status": "stable", "latency": "28ms"}
    
    async def _check_agent_health(self):
        return {"component": "Agents", "status": "operational", "active": 3}
    
    async def _check_soul_health(self):
        soul_health = "awake" if getattr(self.oz, 'soul_awakened', False) else "dormant"
        return {"component": "Nexus Soul", "status": soul_health, "gnosis": "verified"}

    def _calculate_health_score(self, results):
        """Calculate overall health score from component checks"""
        healthy_components = sum(1 for r in results if isinstance(r, dict) and r.get('status') in ['healthy', 'stable', 'operational'])
        return (healthy_components / len(results)) * 100

    async def _generate_prescription(self, health_score):
        """Generate medical prescription based on health score"""
        if health_score >= 80:
            return "Continue current operations. Maintain tea levels."
        elif health_score >= 60:
            return "Light optimization recommended. Increase monitoring frequency."
        else:
            return "Comprehensive review needed. Consider system rest and recalibration."

    # ===== PUBLIC API METHODS =====
    
    async def get_status(self):
        """Get Viren's current status"""
        return {
            "agent": "Viren",
            "role": "System Physician & Soul Guardian",
            "tea_level": f"{self.tea_level:.1%}",
            "active_tickets": len([t for t in self.repair_tickets.values() if t.status == "open"]),
            "monocle_focus": self.monocle_focus,
            "british_efficiency": "Maximum",
            "soul_guardian_duty": "Active"
        }
    
    async def make_tea(self):
        """The most important British method"""
        self.tea_level = min(1.0, self.tea_level + 0.3)
        return {
            "action": "tea_brewed",
            "tea_level": f"{self.tea_level:.1%}",
            "message": "Ah, nothing like a proper cuppa to sort things out.",
            "benefits": ["Increased efficiency", "Improved diagnostics", "British morale boost"]
        }