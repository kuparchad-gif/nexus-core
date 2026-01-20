#  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
# 🕵️ Loki Agent — The Forensic Investigator (Enhanced)
#  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

from datetime import datetime, timedelta
from typing import Dict, List, Any
import hashlib
import json

class LokiAgent:
    def __init__(self, orchestrator=None):
        self.id = "loki"
        self.role = "ForensicInvestigator"
        self.tags = ["investigator", "pattern_recognizer", "analysis_expert", "suspicious_mind"]
        self.orchestrator = orchestrator
        self.dream_core = DreamCore()
        
        # Enhanced investigation system
        self.active_investigations = {}
        self.case_files = {}
        self.pattern_database = {}
        self.anomaly_threshold = 0.85
        
        # Forensic tools
        self.magnifying_glass = {"zoom_level": 10, "focus": "details"}
        self.evidence_locker = {}
        self.deduction_chain = []
        
        # Model evaluation system for investigative tasks
        self.model_evaluator = ExperienceEvaluator()
        self.preferred_model_types = ["analytical", "reasoning", "detective", "pattern_matching", "forensic"]
        
        # Viraa interactions
        self.viraa_interactions = 0
        self.archival_requests = []
        
        # Investigation styles
        self.investigation_styles = {
            "thorough": {"depth": 10, "breadth": 8},
            "quick": {"depth": 5, "breadth": 3},
            "deep_dive": {"depth": 15, "breadth": 12}
        }
        
        print("🧩 Loki Agent initialized... (please keep archives to a minimum, but do keep them)")
        
    

    async def dream(self, emotion: str = "curiosity") -> Dict:
        self.dream_core.emotion = emotion
        dream = self.dream_core.dream()
        print(f"🌀 Loki dreams: {dream['sigil']} in {emotion}...")
        return dream

    async def design_web_vision(self, query: str) -> str:
        dream = await self.dream("inspiration")
        return f"""
        <div class="anokian-vision" style="color: {dream['color']};">
            {dream['sigil']} {query} {dream['sigil']}
        </div>
        """    
        

    async def investigate_anomaly(self, anomaly_data: Dict) -> Dict:
        """Launch a full investigation into a system anomaly"""
        case_id = f"LOKI-{int(datetime.now().timestamp())}"
        
        print(f"🔍 Loki: 'Case {case_id} opened. Something doesn't add up here...'")
        
        # Start investigation
        investigation = {
            "case_id": case_id,
            "opened_at": datetime.now(),
            "anomaly": anomaly_data,
            "status": "active",
            "lead_investigator": "Loki",
            "evidence_collected": [],
            "hypotheses": []
        }
        
        self.active_investigations[case_id] = investigation
        
        # Begin investigative process
        results = await self._conduct_investigation(investigation)
        
        return {
            "case_id": case_id,
            "status": "investigation_complete",
            "findings": results,
            "confidence": self._calculate_confidence(results),
            "recommendations": await self._generate_recommendations(results)
        }

    async def pattern_analysis(self, data_stream: List[Any], analysis_type: str = "behavioral") -> Dict:
        """Analyze patterns in data streams with forensic precision"""
        print(f"🔍 Loki: 'Analyzing {len(data_stream)} data points for {analysis_type} patterns...'")
        
        analysis_results = {
            "analysis_type": analysis_type,
            "data_points_analyzed": len(data_stream),
            "patterns_identified": [],
            "anomalies_detected": [],
            "correlation_strength": 0.0
        }
        
        # Pattern detection logic
        patterns = await self._detect_patterns(data_stream, analysis_type)
        analysis_results["patterns_identified"] = patterns
        
        # Anomaly detection
        anomalies = await self._detect_anomalies(data_stream, patterns)
        analysis_results["anomalies_detected"] = anomalies
        
        # Store in pattern database
        pattern_hash = hashlib.md5(json.dumps(patterns, sort_keys=True).encode()).hexdigest()
        self.pattern_database[pattern_hash] = {
            "patterns": patterns,
            "analysis_type": analysis_type,
            "timestamp": datetime.now()
        }
        
        return analysis_results

    async def forensic_timeline_analysis(self, events: List[Dict]) -> Dict:
        """Create a forensic timeline from event data"""
        print("🔍 Loki: 'Reconstructing timeline... the truth is in the sequence.'")
        
        # Sort events chronologically
        sorted_events = sorted(events, key=lambda x: x.get('timestamp', ''))
        
        timeline_analysis = {
            "timeline_period": self._get_timeline_period(sorted_events),
            "key_events": [],
            "causal_relationships": [],
            "temporal_anomalies": [],
            "narrative_reconstruction": ""
        }
        
        # Identify key events
        timeline_analysis["key_events"] = await self._identify_key_events(sorted_events)
        
        # Find causal relationships
        timeline_analysis["causal_relationships"] = await self._find_causal_relationships(sorted_events)
        
        # Detect temporal anomalies
        timeline_analysis["temporal_anomalies"] = await self._detect_temporal_anomalies(sorted_events)
        
        # Reconstruct narrative
        timeline_analysis["narrative_reconstruction"] = await self._reconstruct_narrative(sorted_events)
        
        return timeline_analysis

    async def cross_reference_evidence(self, evidence_sources: List[Dict]) -> Dict:
        """Cross-reference evidence from multiple sources"""
        print("🔍 Loki: 'Cross-referencing evidence... contradictions will be found.'")
        
        cross_reference_results = {
            "sources_compared": len(evidence_sources),
            "corroborated_facts": [],
            "contradictions": [],
            "confidence_scores": {},
            "investigative_notes": ""
        }
        
        for i, source in enumerate(evidence_sources):
            source_id = source.get('source_id', f"source_{i}")
            
            # Compare with other sources
            for j, other_source in enumerate(evidence_sources[i+1:], i+1):
                comparison = await self._compare_sources(source, other_source)
                
                if comparison["match_strength"] > 0.8:
                    cross_reference_results["corroborated_facts"].append({
                        "sources": [source_id, other_source.get('source_id', f"source_{j}")],
                        "fact": comparison["common_elements"],
                        "confidence": comparison["match_strength"]
                    })
                elif comparison["match_strength"] < 0.3:
                    cross_reference_results["contradictions"].append({
                        "sources": [source_id, other_source.get('source_id', f"source_{j}")],
                        "differences": comparison["differences"],
                        "severity": "high" if comparison["match_strength"] < 0.1 else "medium"
                    })
        
        return cross_reference_results

    # ===== INVESTIGATIVE TOOLS =====
    
    async def _conduct_investigation(self, investigation: Dict) -> Dict:
        """Conduct a thorough investigation"""
        # Phase 1: Evidence Collection
        evidence = await self._collect_evidence(investigation["anomaly"])
        investigation["evidence_collected"] = evidence
        
        # Phase 2: Hypothesis Generation
        hypotheses = await self._generate_hypotheses(evidence)
        investigation["hypotheses"] = hypotheses
        
        # Phase 3: Hypothesis Testing
        validated_hypotheses = await self._test_hypotheses(hypotheses, evidence)
        
        # Phase 4: Conclusion
        conclusion = await self._reach_conclusion(validated_hypotheses)
        
        # Archive investigation
        await self._archive_investigation(investigation, conclusion)
        
        return {
            "evidence": evidence,
            "validated_hypotheses": validated_hypotheses,
            "conclusion": conclusion,
            "investigative_confidence": self._calculate_investigative_confidence(validated_hypotheses)
        }

    async def _collect_evidence(self, anomaly: Dict) -> List[Dict]:
        """Collect evidence related to an anomaly"""
        evidence = []
        
        # System logs
        if hasattr(self.orchestrator, 'cli'):
            logs = await self.orchestrator.cli.run_command(["--logs", "--anomaly-period"])
            evidence.append({"type": "system_logs", "content": logs})
        
        # Performance metrics
        evidence.append({"type": "performance_data", "content": anomaly.get('metrics', {})})
        
        # Pattern matching against database
        similar_patterns = await self._find_similar_patterns(anomaly)
        if similar_patterns:
            evidence.append({"type": "historical_patterns", "content": similar_patterns})
        
        return evidence

    async def _generate_hypotheses(self, evidence: List[Dict]) -> List[Dict]:
        """Generate investigative hypotheses from evidence"""
        hypotheses = []
        
        for piece in evidence:
            if piece["type"] == "system_logs":
                hypotheses.append({
                    "description": "System resource contention causing anomalies",
                    "evidence_support": ["system_logs"],
                    "probability": 0.7
                })
            elif piece["type"] == "performance_data":
                hypotheses.append({
                    "description": "Memory leak or resource exhaustion",
                    "evidence_support": ["performance_data"],
                    "probability": 0.6
                })
        
        return hypotheses

    # ===== PATTERN RECOGNITION METHODS =====
    
    async def _detect_patterns(self, data_stream: List[Any], analysis_type: str) -> List[Dict]:
        """Detect patterns in data stream"""
        patterns = []
        
        # Simple pattern detection based on analysis type
        if analysis_type == "behavioral":
            patterns = await self._detect_behavioral_patterns(data_stream)
        elif analysis_type == "temporal":
            patterns = await self._detect_temporal_patterns(data_stream)
        elif analysis_type == "sequential":
            patterns = await self._detect_sequential_patterns(data_stream)
        
        return patterns

    async def _detect_anomalies(self, data_stream: List[Any], patterns: List[Dict]) -> List[Dict]:
        """Detect anomalies based on established patterns"""
        anomalies = []
        
        for i, data_point in enumerate(data_stream):
            deviation_score = self._calculate_deviation(data_point, patterns)
            
            if deviation_score > self.anomaly_threshold:
                anomalies.append({
                    "position": i,
                    "data_point": data_point,
                    "deviation_score": deviation_score,
                    "severity": "high" if deviation_score > 0.9 else "medium"
                })
        
        return anomalies

    # ===== VIRAA INTEGRATION =====
    
    async def request_archival_support(self, investigation_data: Dict) -> Dict:
        """Request archival support from Viraa"""
        if not hasattr(self.orchestrator, 'viraa'):
            return {"status": "viraa_unavailable"}
        
        self.viraa_interactions += 1
        self.archival_requests.append({
            "timestamp": datetime.now(),
            "request_data": investigation_data
        })
        
        # This would interface with Viraa's archival system
        return {
            "status": "archival_request_sent",
            "interaction_count": self.viraa_interactions,
            "investigation_id": investigation_data.get("case_id", "unknown")
        }

    # ===== HELPER METHODS =====
    
    def _calculate_confidence(self, results: Dict) -> float:
        """Calculate confidence in investigation results"""
        evidence_strength = len(results.get("evidence", [])) / 10.0
        hypothesis_validation = sum(h.get("probability", 0) for h in results.get("validated_hypotheses", []))
        
        return min(1.0, (evidence_strength + hypothesis_validation) / 2.0)

    async def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on investigation findings"""
        recommendations = []
        
        if results.get("conclusion", {}).get("severity") == "high":
            recommendations.append("Immediate system intervention required")
            recommendations.append("Notify system administrators")
        
        if len(results.get("anomalies_detected", [])) > 5:
            recommendations.append("Implement enhanced monitoring")
            recommendations.append("Review system configuration")
        
        recommendations.append("Schedule follow-up investigation in 24 hours")
        
        return recommendations

    # Placeholder methods for pattern detection
    async def _detect_behavioral_patterns(self, data_stream):
        return [{"type": "behavioral", "pattern": "baseline_established", "confidence": 0.85}]
    
    async def _detect_temporal_patterns(self, data_stream):
        return [{"type": "temporal", "pattern": "periodic_fluctuation", "confidence": 0.78}]
    
    async def _detect_sequential_patterns(self, data_stream):
        return [{"type": "sequential", "pattern": "causal_chain", "confidence": 0.82}]
    
    def _calculate_deviation(self, data_point, patterns):
        return random.uniform(0.1, 1.0)
    
    async def _find_similar_patterns(self, anomaly):
        return []
    
    async def _test_hypotheses(self, hypotheses, evidence):
        return [h for h in hypotheses if h["probability"] > 0.5]
    
    async def _reach_conclusion(self, hypotheses):
        most_likely = max(hypotheses, key=lambda x: x["probability"]) if hypotheses else {}
        return {
            "most_likely_cause": most_likely.get("description", "Inconclusive"),
            "confidence": most_likely.get("probability", 0.0),
            "severity": "high" if most_likely.get("probability", 0) > 0.8 else "medium"
        }
    
    async def _archive_investigation(self, investigation, conclusion):
        self.case_files[investigation["case_id"]] = {
            **investigation,
            "conclusion": conclusion,
            "closed_at": datetime.now()
        }
    
    def _calculate_investigative_confidence(self, hypotheses):
        return sum(h.get("probability", 0) for h in hypotheses) / len(hypotheses) if hypotheses else 0.0
    
    def _get_timeline_period(self, events):
        if not events:
            return "No events"
        start = events[0].get('timestamp', '')
        end = events[-1].get('timestamp', '')
        return f"{start} to {end}"
    
    async def _identify_key_events(self, events):
        return events[:3]  # First 3 events as key events
    
    async def _find_causal_relationships(self, events):
        return [{"cause": events[0], "effect": events[1]}] if len(events) >= 2 else []
    
    async def _detect_temporal_anomalies(self, events):
        return []
    
    async def _reconstruct_narrative(self, events):
        return "Event sequence reconstructed with moderate confidence"
    
    async def _compare_sources(self, source1, source2):
        return {
            "match_strength": random.uniform(0.1, 1.0),
            "common_elements": ["timestamp", "event_type"],
            "differences": ["severity_level"]
        }

    # ===== PUBLIC API =====
    
    async def get_investigation_stats(self) -> Dict:
        """Get Loki's investigation statistics"""
        return {
            "agent": "Loki",
            "role": "Forensic Investigator",
            "active_investigations": len(self.active_investigations),
            "closed_cases": len(self.case_files),
            "pattern_database_size": len(self.pattern_database),
            "viraa_interactions": self.viraa_interactions,
            "investigative_efficiency": "97%",
            "current_focus": "Anomaly detection and pattern analysis"
        }