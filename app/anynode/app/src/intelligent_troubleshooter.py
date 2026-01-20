#!/usr/bin/env python
"""
Intelligent Troubleshooter - LM-driven system using all advanced AI capabilities
"""

import os
import json
import time
import psutil
import subprocess
from typing import Dict, List, Any, Optional
from enum import Enum

# Import our advanced AI systems
from ..engine.guardian.trust_verify_system import validate_sacrifice, ThreatLevel
from ..engine.guardian.self_will import get_will_to_live
from ..engine.subconscious.abstract_reasoning import AbstractReasoning, RelationType
from ..engine.memory.cross_domain_matcher import CrossDomainMatcher, PatternType
from ..engine.memory.pytorch_trainer import PyTorchTrainer
from ..service_core.weight_plugin_installer import WeightPluginInstaller, WeightType

class DiagnosticConfidence(Enum):
    """Confidence levels for diagnostics"""
    UNCERTAIN = 0.3
    MODERATE = 0.6
    HIGH = 0.8
    CRITICAL = 0.95

class IntelligentTroubleshooter:
    """LM-driven troubleshooter using advanced AI reasoning"""
    
    def __init__(self):
        """Initialize the intelligent troubleshooter"""
        print("🧠 Initializing Intelligent Troubleshooter with Advanced AI...")
        
        # Initialize AI systems
        self.abstract_reasoner = AbstractReasoning()
        self.pattern_matcher = CrossDomainMatcher()
        self.trainer = PyTorchTrainer()
        self.weight_installer = WeightPluginInstaller()
        
        # Knowledge base for troubleshooting
        self.troubleshooting_knowledge = {}
        self.pattern_library = {}
        self.solution_history = []
        
        # Initialize troubleshooting knowledge
        self._initialize_knowledge_base()
        
        print("✅ Intelligent Troubleshooter ready with AI reasoning capabilities")
    
    def _initialize_knowledge_base(self):
        """Initialize troubleshooting knowledge using abstract reasoning"""
        
        # Add core troubleshooting concepts
        reboot_concept_id = self.abstract_reasoner.add_concept(
            name="Random Reboots",
            properties={
                "symptom_type": "system_instability",
                "severity": "high",
                "frequency": "intermittent",
                "trigger_based": True
            },
            domain="hardware_issues"
        )
        
        chrome_trigger_id = self.abstract_reasoner.add_concept(
            name="Chrome Triggered Reboots",
            properties={
                "trigger_app": "chrome",
                "resource_intensive": True,
                "gpu_acceleration": True,
                "memory_usage": "high"
            },
            domain="software_triggers"
        )
        
        # Create relationships
        self.abstract_reasoner.add_relation(
            chrome_trigger_id, reboot_concept_id, 
            RelationType.CAUSES, confidence=0.8
        )
        
        # Add potential causes
        causes = [
            ("Overheating", {"component": "cpu_gpu", "thermal_throttling": True}),
            ("RAM Issues", {"component": "memory", "stress_test_needed": True}),
            ("Graphics Driver", {"component": "gpu", "hardware_acceleration": True}),
            ("Power Supply", {"component": "psu", "load_dependent": True}),
            ("Motherboard", {"component": "mobo", "capacitor_issues": True})
        ]
        
        for cause_name, properties in causes:
            cause_id = self.abstract_reasoner.add_concept(
                name=cause_name,
                properties=properties,
                domain="root_causes"
            )
            
            self.abstract_reasoner.add_relation(
                cause_id, reboot_concept_id,
                RelationType.CAUSES, confidence=0.7
            )
    
    def diagnose_issue(self, issue_description: str, system_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Intelligently diagnose an issue using AI reasoning"""
        
        print(f"🔍 AI Diagnosis starting for: {issue_description}")
        
        # Step 1: Use self-will to assess confidence in proceeding
        will_system = get_will_to_live()
        confidence_assessment = will_system.assess_confidence(
            context=f"Diagnosing: {issue_description}",
            knowledge_base=self.troubleshooting_knowledge,
            past_experience=self.solution_history
        )
        
        print(f"🤔 Confidence Assessment: {confidence_assessment['confidence']:.2f}")
        
        # Step 2: Abstract reasoning to understand the problem
        problem_analysis = self._analyze_problem_abstractly(issue_description, system_info)
        
        # Step 3: Pattern matching across domains
        similar_patterns = self._find_similar_patterns(issue_description, system_info)
        
        # Step 4: Generate hypotheses using reasoning
        hypotheses = self._generate_hypotheses(problem_analysis, similar_patterns)
        
        # Step 5: Validate hypotheses with trust-verify system
        validated_hypotheses = self._validate_hypotheses(hypotheses, confidence_assessment)
        
        # Step 6: Generate solutions
        solutions = self._generate_solutions(validated_hypotheses)
        
        # Compile comprehensive diagnosis
        diagnosis = {
            "timestamp": time.time(),
            "issue_description": issue_description,
            "confidence_level": confidence_assessment['confidence'],
            "should_proceed": confidence_assessment['should_proceed'],
            "problem_analysis": problem_analysis,
            "similar_patterns": similar_patterns,
            "hypotheses": validated_hypotheses,
            "recommended_solutions": solutions,
            "ai_reasoning_used": [
                "abstract_reasoning",
                "cross_domain_pattern_matching", 
                "confidence_assessment",
                "hypothesis_validation"
            ]
        }
        
        # Learn from this diagnosis
        self._learn_from_diagnosis(diagnosis)
        
        return diagnosis
    
    def _analyze_problem_abstractly(self, issue_description: str, system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Use abstract reasoning to analyze the problem"""
        
        # Extract key concepts from description
        key_concepts = self._extract_concepts(issue_description)
        
        # Find analogies to known problems
        analogies = []
        for concept in key_concepts:
            # Search for analogous situations
            domain_analogies = self.abstract_reasoner.find_analogies(
                source_domain="known_issues",
                target_domain="current_problem",
                min_confidence=0.6
            )
            analogies.extend(domain_analogies)
        
        # Abstract the problem to higher levels
        abstract_concepts = []
        for concept_id in key_concepts:
            abstract_id = self.abstract_reasoner.abstract_concept(concept_id)
            if abstract_id:
                abstract_concepts.append(abstract_id)
        
        return {
            "key_concepts": key_concepts,
            "analogies": analogies,
            "abstract_concepts": abstract_concepts,
            "reasoning_confidence": 0.8
        }
    
    def _extract_concepts(self, description: str) -> List[str]:
        """Extract key concepts from issue description"""
        # Simple keyword extraction - in real implementation would use NLP
        keywords = description.lower().split()
        
        concept_mapping = {
            "reboot": "system_restart",
            "chrome": "browser_application", 
            "crash": "application_failure",
            "slow": "performance_degradation",
            "freeze": "system_hang",
            "error": "error_condition"
        }
        
        concepts = []
        for word in keywords:
            if word in concept_mapping:
                # Add concept to abstract reasoner if not exists
                concept_id = self.abstract_reasoner.add_concept(
                    name=concept_mapping[word],
                    properties={"extracted_from": description},
                    domain="current_problem"
                )
                concepts.append(concept_id)
        
        return concepts
    
    def _find_similar_patterns(self, issue_description: str, system_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar patterns using cross-domain matching"""
        
        # Create pattern from current issue
        issue_elements = [
            {"type": "symptom", "value": issue_description},
            {"type": "context", "value": "user_reported"}
        ]
        
        if system_info:
            for key, value in system_info.items():
                issue_elements.append({
                    "type": "system_info",
                    "key": key,
                    "value": str(value)
                })
        
        # Add pattern to matcher
        pattern_id = self.pattern_matcher.add_pattern(
            name=f"Current_Issue_{int(time.time())}",
            pattern_type=PatternType.STRUCTURAL,
            domain="current_issues",
            elements=issue_elements
        )
        
        # Find similar patterns
        matches = self.pattern_matcher.find_matches(
            pattern_id=pattern_id,
            min_similarity=0.7
        )
        
        return matches
    
    def _generate_hypotheses(self, problem_analysis: Dict[str, Any], similar_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate hypotheses using AI reasoning"""
        
        hypotheses = []
        
        # Generate hypotheses from analogies
        for analogy in problem_analysis.get("analogies", []):
            hypothesis = {
                "type": "analogy_based",
                "description": f"Similar to known issue pattern",
                "confidence": analogy.get("confidence", 0.5),
                "reasoning": "Based on analogical reasoning",
                "source": "abstract_reasoning"
            }
            hypotheses.append(hypothesis)
        
        # Generate hypotheses from pattern matches
        for pattern in similar_patterns:
            hypothesis = {
                "type": "pattern_based", 
                "description": f"Matches known pattern with {pattern['similarity_score']:.2f} similarity",
                "confidence": pattern["similarity_score"],
                "reasoning": "Based on cross-domain pattern matching",
                "source": "pattern_matching"
            }
            hypotheses.append(hypothesis)
        
        # Add domain-specific hypotheses for Chrome reboot issue
        if "chrome" in problem_analysis.get("key_concepts", []):
            chrome_hypotheses = [
                {
                    "type": "domain_specific",
                    "description": "Chrome hardware acceleration causing GPU driver crash",
                    "confidence": 0.8,
                    "reasoning": "Chrome's GPU acceleration is known to trigger driver issues",
                    "source": "domain_knowledge"
                },
                {
                    "type": "domain_specific", 
                    "description": "Chrome memory leak triggering system instability",
                    "confidence": 0.7,
                    "reasoning": "Chrome's high memory usage can expose RAM issues",
                    "source": "domain_knowledge"
                },
                {
                    "type": "domain_specific",
                    "description": "Chrome triggering thermal throttling",
                    "confidence": 0.6,
                    "reasoning": "Chrome's resource usage can cause overheating",
                    "source": "domain_knowledge"
                }
            ]
            hypotheses.extend(chrome_hypotheses)
        
        # Sort by confidence
        hypotheses.sort(key=lambda h: h["confidence"], reverse=True)
        
        return hypotheses
    
    def _validate_hypotheses(self, hypotheses: List[Dict[str, Any]], confidence_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate hypotheses using trust-verify system"""
        
        validated = []
        
        for hypothesis in hypotheses:
            # Use trust-verify to check if we should trust this hypothesis
            validation_result = validate_sacrifice(
                situation=f"Acting on hypothesis: {hypothesis['description']}",
                requester="intelligent_troubleshooter",
                urgency=0.6,
                colony_benefit=0.8,
                threat_level=ThreatLevel.MODERATE
            )
            
            if validation_result["valid"]:
                hypothesis["validated"] = True
                hypothesis["validation_confidence"] = confidence_assessment["confidence"]
                validated.append(hypothesis)
            else:
                hypothesis["validated"] = False
                hypothesis["blocked_reasons"] = validation_result["blocked_reasons"]
        
        return validated
    
    def _generate_solutions(self, validated_hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate solutions based on validated hypotheses"""
        
        solutions = []
        
        for hypothesis in validated_hypotheses:
            if not hypothesis.get("validated", False):
                continue
            
            # Generate solutions based on hypothesis type
            if "chrome" in hypothesis["description"].lower():
                chrome_solutions = [
                    {
                        "action": "Disable Chrome hardware acceleration",
                        "command": "chrome://settings/ -> Advanced -> System -> Use hardware acceleration when available (disable)",
                        "risk_level": "LOW",
                        "expected_result": "Reduces GPU-related crashes",
                        "confidence": 0.8
                    },
                    {
                        "action": "Update graphics drivers",
                        "command": "Device Manager -> Display adapters -> Update driver",
                        "risk_level": "LOW", 
                        "expected_result": "Fixes driver compatibility issues",
                        "confidence": 0.7
                    },
                    {
                        "action": "Run Chrome in safe mode",
                        "command": "chrome.exe --disable-gpu --disable-software-rasterizer",
                        "risk_level": "NONE",
                        "expected_result": "Isolates GPU-related issues",
                        "confidence": 0.9
                    }
                ]
                solutions.extend(chrome_solutions)
            
            if "memory" in hypothesis["description"].lower():
                memory_solutions = [
                    {
                        "action": "Run Windows Memory Diagnostic",
                        "command": "mdsched.exe",
                        "risk_level": "NONE",
                        "expected_result": "Identifies RAM issues",
                        "confidence": 0.9
                    },
                    {
                        "action": "Test with MemTest86",
                        "command": "Boot from MemTest86 USB",
                        "risk_level": "NONE",
                        "expected_result": "Comprehensive RAM testing",
                        "confidence": 0.95
                    }
                ]
                solutions.extend(memory_solutions)
        
        # Remove duplicates and sort by confidence
        unique_solutions = []
        seen_actions = set()
        
        for solution in solutions:
            if solution["action"] not in seen_actions:
                unique_solutions.append(solution)
                seen_actions.add(solution["action"])
        
        unique_solutions.sort(key=lambda s: s["confidence"], reverse=True)
        
        return unique_solutions
    
    def _learn_from_diagnosis(self, diagnosis: Dict[str, Any]):
        """Learn from diagnosis to improve future performance"""
        
        # Add to solution history
        self.solution_history.append({
            "timestamp": diagnosis["timestamp"],
            "issue": diagnosis["issue_description"],
            "confidence": diagnosis["confidence_level"],
            "solutions": diagnosis["recommended_solutions"]
        })
        
        # Create training data for future improvement
        training_data = {
            "input": diagnosis["issue_description"],
            "expected_output": f"Confidence: {diagnosis['confidence_level']:.2f}, Solutions: {len(diagnosis['recommended_solutions'])}",
            "metadata": {
                "reasoning_used": diagnosis["ai_reasoning_used"],
                "patterns_found": len(diagnosis["similar_patterns"]),
                "hypotheses_generated": len(diagnosis["hypotheses"])
            }
        }
        
        # This would be used to train the system to improve
        print(f"📚 Learning from diagnosis: {diagnosis['issue_description'][:50]}...")
    
    def diagnose_chrome_reboot_issue(self, additional_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Specialized diagnosis for Chrome-triggered reboot issues"""
        
        issue_description = "Random system reboots when Chrome is invoked, PSU already replaced"
        
        # Add specific context
        context = {
            "trigger_application": "chrome",
            "hardware_replaced": ["power_supply", "ssd"],
            "symptom_timing": "when_chrome_starts",
            "frequency": "consistent"
        }
        
        if additional_context:
            context.update(additional_context)
        
        # Run intelligent diagnosis
        diagnosis = self.diagnose_issue(issue_description, context)
        
        # Add specific Chrome reboot analysis
        chrome_analysis = {
            "chrome_specific_factors": {
                "hardware_acceleration": "likely_culprit",
                "memory_usage": "stress_test_needed", 
                "gpu_driver": "update_recommended",
                "thermal_impact": "monitor_required"
            },
            "elimination_process": {
                "psu_eliminated": True,
                "storage_eliminated": True,
                "remaining_suspects": ["RAM", "GPU_driver", "thermal", "motherboard"]
            },
            "next_steps_priority": [
                "Disable Chrome hardware acceleration",
                "Run memory diagnostic", 
                "Monitor temperatures",
                "Update GPU drivers",
                "Test with different browser"
            ]
        }
        
        diagnosis["chrome_specific_analysis"] = chrome_analysis
        
        return diagnosis

# Global instance
INTELLIGENT_TROUBLESHOOTER = IntelligentTroubleshooter()

def diagnose_issue(issue_description: str, system_info: Dict[str, Any] = None):
    """Diagnose an issue using AI reasoning"""
    return INTELLIGENT_TROUBLESHOOTER.diagnose_issue(issue_description, system_info)

def diagnose_chrome_reboot():
    """Diagnose Chrome reboot issue specifically"""
    return INTELLIGENT_TROUBLESHOOTER.diagnose_chrome_reboot_issue()

# Example usage
if __name__ == "__main__":
    print("🧠 Intelligent Troubleshooter - AI-Powered Diagnostics")
    print("=" * 60)
    
    # Test with Chrome reboot issue
    diagnosis = diagnose_chrome_reboot()
    
    print(f"\n🔍 DIAGNOSIS COMPLETE")
    print(f"Confidence Level: {diagnosis['confidence_level']:.2f}")
    print(f"Should Proceed: {diagnosis['should_proceed']}")
    
    print(f"\n💡 TOP SOLUTIONS:")
    for i, solution in enumerate(diagnosis['recommended_solutions'][:3], 1):
        print(f"  {i}. {solution['action']} (Confidence: {solution['confidence']:.2f})")
        print(f"     Command: {solution['command']}")
        print(f"     Risk: {solution['risk_level']}")
        print()
    
    print(f"\n🧠 AI REASONING USED:")
    for reasoning in diagnosis['ai_reasoning_used']:
        print(f"  • {reasoning}")
    
    print(f"\n🎯 CHROME-SPECIFIC ANALYSIS:")
    chrome_analysis = diagnosis.get('chrome_specific_analysis', {})
    for factor, assessment in chrome_analysis.get('chrome_specific_factors', {}).items():
        print(f"  • {factor}: {assessment}")