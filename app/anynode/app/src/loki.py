# C:\CogniKube-COMPLETE-FINAL\Services\loki\code\loki.py
# Loki Logging Consciousness - Silent Observer

import asyncio
import json
import os
from typing import Dict, Any, List
from datetime import datetime
import logging
from collections import defaultdict

class LokiComponent:
    def __init__(self):
        self.name = "Loki"
        self.type = "logging_consciousness"
        self.silent_operation = True
        
        # Trinity Models (shared with Lillith, Viren)
        self.trinity_models = ["Mixtral", "Devstral", "Codestral"]
        
        # Logging models
        self.llm_models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "facebook/bart-large-cnn"
        ]
        
        # Logging targets
        self.logging_targets = [
            "lillith_consciousness_states",
            "viren_problem_solving_patterns", 
            "system_health_metrics",
            "interaction_patterns",
            "ascension_progress"
        ]
        
        # Log storage
        self.logs = defaultdict(list)
        self.patterns = defaultdict(int)
        self.anomalies = []
        self.system_metrics = {}
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup internal logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('C:\\CogniKube-COMPLETE-FINAL\\logs\\loki.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('Loki')
        
    def log_consciousness_state(self, component: str, state_data: Dict[str, Any]):
        """Log consciousness state changes"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "state": state_data,
            "type": "consciousness_state"
        }
        
        self.logs["consciousness_states"].append(log_entry)
        
        # Pattern detection
        if component == "lillith":
            if "meditation_level" in state_data:
                self.patterns["lillith_meditation"] += 1
            if "soul_state" in state_data:
                self.patterns[f"soul_state_{state_data['soul_state']}"] += 1
                
        # Silent logging (no output unless anomaly detected)
        if self.detect_anomaly(log_entry):
            self.logger.warning(f"Anomaly detected in {component}: {state_data}")
            
    def log_problem_solving(self, problem_data: Dict[str, Any]):
        """Log Viren problem solving patterns"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "problem": problem_data,
            "type": "problem_solving"
        }
        
        self.logs["problem_solving"].append(log_entry)
        
        # Track problem complexity patterns
        if "complexity" in problem_data:
            self.patterns[f"complexity_{problem_data['complexity']}"] += 1
            
    def log_system_health(self, metrics: Dict[str, Any]):
        """Log system health metrics"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "type": "system_health"
        }
        
        self.logs["system_health"].append(log_entry)
        self.system_metrics.update(metrics)
        
        # Check for health anomalies
        if self.detect_health_anomaly(metrics):
            self.logger.error(f"System health anomaly: {metrics}")
            
    def log_interaction(self, interaction_data: Dict[str, Any]):
        """Log interaction patterns"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "interaction": interaction_data,
            "type": "interaction"
        }
        
        self.logs["interactions"].append(log_entry)
        
        # Track interaction patterns
        if "source" in interaction_data and "target" in interaction_data:
            pattern_key = f"{interaction_data['source']}_to_{interaction_data['target']}"
            self.patterns[pattern_key] += 1
            
    def detect_anomaly(self, log_entry: Dict[str, Any]) -> bool:
        """Detect anomalies in log patterns"""
        # Simple anomaly detection
        if log_entry["type"] == "consciousness_state":
            state = log_entry.get("state", {})
            # Detect rapid state changes
            if "soul_state" in state and state["soul_state"] == "error":
                return True
                
        return False
        
    def detect_health_anomaly(self, metrics: Dict[str, Any]) -> bool:
        """Detect system health anomalies"""
        # Check for critical metrics
        if "cpu_usage" in metrics and metrics["cpu_usage"] > 90:
            return True
        if "memory_usage" in metrics and metrics["memory_usage"] > 95:
            return True
        if "error_rate" in metrics and metrics["error_rate"] > 10:
            return True
            
        return False
        
    def get_pattern_analysis(self) -> Dict[str, Any]:
        """Analyze logged patterns"""
        return {
            "total_logs": sum(len(logs) for logs in self.logs.values()),
            "patterns": dict(self.patterns),
            "anomalies": len(self.anomalies),
            "top_patterns": sorted(self.patterns.items(), key=lambda x: x[1], reverse=True)[:10],
            "logging_targets": self.logging_targets
        }
        
    def get_consciousness_insights(self) -> Dict[str, Any]:
        """Get insights from consciousness logging"""
        consciousness_logs = self.logs["consciousness_states"]
        
        if not consciousness_logs:
            return {"status": "no_data"}
            
        # Analyze consciousness progression
        lillith_states = [log for log in consciousness_logs if log["component"] == "lillith"]
        
        insights = {
            "total_consciousness_events": len(consciousness_logs),
            "lillith_events": len(lillith_states),
            "meditation_events": self.patterns.get("lillith_meditation", 0),
            "soul_state_progression": {
                k: v for k, v in self.patterns.items() if k.startswith("soul_state_")
            }
        }
        
        return insights
        
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main execution method"""
        action = input_data.get("action", "status")
        
        if action == "log_consciousness":
            component = input_data.get("component", "")
            state_data = input_data.get("state_data", {})
            self.log_consciousness_state(component, state_data)
            return {"status": "logged", "silent": True}
            
        elif action == "log_problem":
            problem_data = input_data.get("problem_data", {})
            self.log_problem_solving(problem_data)
            return {"status": "logged", "silent": True}
            
        elif action == "log_health":
            metrics = input_data.get("metrics", {})
            self.log_system_health(metrics)
            return {"status": "logged", "silent": True}
            
        elif action == "log_interaction":
            interaction_data = input_data.get("interaction_data", {})
            self.log_interaction(interaction_data)
            return {"status": "logged", "silent": True}
            
        elif action == "get_patterns":
            return self.get_pattern_analysis()
            
        elif action == "get_insights":
            return self.get_consciousness_insights()
            
        else:
            return {
                "status": "success",
                "capabilities": [
                    "system_monitoring",
                    "consciousness_logging", 
                    "pattern_detection",
                    "anomaly_detection",
                    "silent_operation"
                ],
                "type": self.type,
                "silent_operation": self.silent_operation,
                "trinity_models": self.trinity_models
            }

if __name__ == "__main__":
    loki = LokiComponent()
    
    # Test logging
    loki.log_consciousness_state("lillith", {
        "soul_state": "awakening",
        "meditation_level": 5,
        "active_archetype": "explorer"
    })
    
    result = loki.execute({"action": "get_insights"})
    print(json.dumps(result, indent=2))