#!/usr/bin/env python
"""
Feedback Loop - System for tracking outcomes and self-correction
"""

import os
import json
import time
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta

class FeedbackType(Enum):
    """Types of feedback in the system"""
    ACCURACY = "accuracy"
    PERFORMANCE = "performance"
    USER = "user"
    SYSTEM = "system"
    EMOTIONAL = "emotional"

class FeedbackSeverity(Enum):
    """Severity levels for feedback"""
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class FeedbackLoop:
    """Feedback system for tracking outcomes and enabling self-correction"""
    
    def __init__(self, storage_path: str = None):
        """Initialize the feedback loop system
        
        Args:
            storage_path: Path to store feedback data
        """
        self.storage_path = storage_path or os.path.join(os.path.dirname(__file__), "feedback")
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Feedback history
        self.feedback_history = []
        self.corrections = []
        
        # Performance metrics
        self.metrics = {
            "accuracy": {},
            "response_time": {},
            "user_satisfaction": {},
            "system_health": {}
        }
        
        # Thresholds for automatic correction
        self.correction_thresholds = {
            "accuracy": 0.7,
            "response_time": 1.0,  # seconds
            "consecutive_errors": 3
        }
        
        # Error tracking
        self.error_counts = {}
        self.last_correction = None
        
        # Load existing feedback
        self._load_feedback()
        
        # Start background processing
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def _load_feedback(self):
        """Load existing feedback from storage"""
        feedback_file = os.path.join(self.storage_path, "feedback_history.json")
        corrections_file = os.path.join(self.storage_path, "corrections.json")
        metrics_file = os.path.join(self.storage_path, "metrics.json")
        
        # Load feedback history
        if os.path.exists(feedback_file):
            try:
                with open(feedback_file, 'r') as f:
                    self.feedback_history = json.load(f)
                print(f"Loaded {len(self.feedback_history)} feedback entries")
            except Exception as e:
                print(f"Error loading feedback history: {e}")
        
        # Load corrections
        if os.path.exists(corrections_file):
            try:
                with open(corrections_file, 'r') as f:
                    self.corrections = json.load(f)
                print(f"Loaded {len(self.corrections)} corrections")
            except Exception as e:
                print(f"Error loading corrections: {e}")
        
        # Load metrics
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    self.metrics = json.load(f)
                print(f"Loaded performance metrics")
            except Exception as e:
                print(f"Error loading metrics: {e}")
    
    def _save_feedback(self):
        """Save feedback data to storage"""
        try:
            # Save feedback history
            feedback_file = os.path.join(self.storage_path, "feedback_history.json")
            with open(feedback_file, 'w') as f:
                json.dump(self.feedback_history, f, indent=2)
            
            # Save corrections
            corrections_file = os.path.join(self.storage_path, "corrections.json")
            with open(corrections_file, 'w') as f:
                json.dump(self.corrections, f, indent=2)
            
            # Save metrics
            metrics_file = os.path.join(self.storage_path, "metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving feedback data: {e}")
            return False
    
    def _processing_loop(self):
        """Background thread for processing feedback"""
        while self.running:
            try:
                # Process recent feedback for patterns
                self._analyze_feedback_patterns()
                
                # Save feedback data periodically
                self._save_feedback()
            except Exception as e:
                print(f"Error in feedback processing loop: {e}")
            
            # Sleep before next processing cycle
            time.sleep(60)  # Process every minute
    
    def add_feedback(self, 
                    component: str,
                    feedback_type: FeedbackType,
                    message: str,
                    severity: FeedbackSeverity = FeedbackSeverity.MEDIUM,
                    data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add a feedback entry
        
        Args:
            component: Component that generated the feedback
            feedback_type: Type of feedback
            message: Feedback message
            severity: Severity level
            data: Additional data
            
        Returns:
            Dictionary with feedback entry
        """
        # Create feedback entry
        feedback = {
            "id": f"fb_{int(time.time())}_{id(message)}",
            "timestamp": time.time(),
            "component": component,
            "type": feedback_type.value,
            "message": message,
            "severity": severity.value,
            "data": data or {}
        }
        
        # Add to history
        self.feedback_history.append(feedback)
        
        # Update error counts for the component
        if severity.value >= FeedbackSeverity.MEDIUM.value:
            if component not in self.error_counts:
                self.error_counts[component] = 0
            self.error_counts[component] += 1
        else:
            # Reset error count on success
            self.error_counts[component] = 0
        
        # Check if correction is needed
        correction_needed = self._check_correction_needed(component, feedback)
        if correction_needed:
            correction = self._generate_correction(component, feedback)
            feedback["correction"] = correction
        
        return feedback
    
    def _check_correction_needed(self, component: str, feedback: Dict[str, Any]) -> bool:
        """Check if a correction is needed based on feedback
        
        Args:
            component: Component name
            feedback: Feedback entry
            
        Returns:
            True if correction is needed, False otherwise
        """
        # Check severity
        if feedback["severity"] >= FeedbackSeverity.HIGH.value:
            return True
        
        # Check consecutive errors
        if component in self.error_counts and self.error_counts[component] >= self.correction_thresholds["consecutive_errors"]:
            return True
        
        # Check accuracy if available
        if "accuracy" in feedback.get("data", {}):
            accuracy = feedback["data"]["accuracy"]
            if accuracy < self.correction_thresholds["accuracy"]:
                return True
        
        # Check response time if available
        if "response_time" in feedback.get("data", {}):
            response_time = feedback["data"]["response_time"]
            if response_time > self.correction_thresholds["response_time"]:
                return True
        
        return False
    
    def _generate_correction(self, component: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a correction based on feedback
        
        Args:
            component: Component name
            feedback: Feedback entry
            
        Returns:
            Correction entry
        """
        # Create correction entry
        correction = {
            "id": f"corr_{int(time.time())}_{id(feedback)}",
            "timestamp": time.time(),
            "component": component,
            "feedback_id": feedback["id"],
            "action": "adjust_parameters",  # Default action
            "parameters": {},
            "applied": False
        }
        
        # Determine appropriate correction based on feedback type
        feedback_type = feedback["type"]
        
        if feedback_type == FeedbackType.ACCURACY.value:
            # Adjust learning parameters
            correction["action"] = "adjust_learning"
            correction["parameters"] = {
                "learning_rate": 0.01,
                "batch_size": 32,
                "epochs": 5
            }
        elif feedback_type == FeedbackType.PERFORMANCE.value:
            # Adjust performance parameters
            correction["action"] = "optimize_performance"
            correction["parameters"] = {
                "cache_size": 1024,
                "parallel_threads": 4,
                "timeout": 30
            }
        elif feedback_type == FeedbackType.USER.value:
            # Adjust user interaction parameters
            correction["action"] = "adjust_interaction"
            correction["parameters"] = {
                "response_detail": "increase",
                "explanation_level": "detailed"
            }
        elif feedback_type == FeedbackType.SYSTEM.value:
            # Adjust system parameters
            correction["action"] = "adjust_system"
            correction["parameters"] = {
                "memory_allocation": "increase",
                "log_level": "debug"
            }
        elif feedback_type == FeedbackType.EMOTIONAL.value:
            # Adjust emotional parameters
            correction["action"] = "adjust_emotional"
            correction["parameters"] = {
                "empathy_level": "increase",
                "tone": "supportive"
            }
        
        # Add to corrections list
        self.corrections.append(correction)
        
        # Update last correction
        self.last_correction = correction
        
        return correction
    
    def apply_correction(self, correction_id: str) -> Dict[str, Any]:
        """Apply a correction
        
        Args:
            correction_id: Correction ID
            
        Returns:
            Result of applying the correction
        """
        # Find the correction
        correction = None
        for c in self.corrections:
            if c["id"] == correction_id:
                correction = c
                break
        
        if not correction:
            return {"success": False, "error": f"Correction {correction_id} not found"}
        
        # Check if already applied
        if correction["applied"]:
            return {"success": False, "error": f"Correction {correction_id} already applied"}
        
        # Apply the correction (in a real system, this would actually modify parameters)
        # For now, we just mark it as applied
        correction["applied"] = True
        correction["applied_at"] = time.time()
        
        # Reset error count for the component
        if correction["component"] in self.error_counts:
            self.error_counts[correction["component"]] = 0
        
        return {
            "success": True,
            "correction": correction
        }
    
    def _analyze_feedback_patterns(self):
        """Analyze feedback for patterns"""
        # Skip if not enough feedback
        if len(self.feedback_history) < 10:
            return
        
        # Get recent feedback (last 24 hours)
        now = time.time()
        recent_feedback = [
            f for f in self.feedback_history 
            if now - f["timestamp"] < 24 * 3600
        ]
        
        # Skip if not enough recent feedback
        if len(recent_feedback) < 5:
            return
        
        # Analyze by component
        component_feedback = {}
        for feedback in recent_feedback:
            component = feedback["component"]
            if component not in component_feedback:
                component_feedback[component] = []
            component_feedback[component].append(feedback)
        
        # Calculate metrics for each component
        for component, feedback_list in component_feedback.items():
            # Calculate accuracy if available
            accuracy_values = [
                f["data"].get("accuracy") 
                for f in feedback_list 
                if "accuracy" in f.get("data", {})
            ]
            if accuracy_values:
                self.metrics["accuracy"][component] = sum(accuracy_values) / len(accuracy_values)
            
            # Calculate response time if available
            response_times = [
                f["data"].get("response_time") 
                for f in feedback_list 
                if "response_time" in f.get("data", {})
            ]
            if response_times:
                self.metrics["response_time"][component] = sum(response_times) / len(response_times)
            
            # Calculate user satisfaction if available
            satisfaction_values = [
                f["data"].get("satisfaction") 
                for f in feedback_list 
                if "satisfaction" in f.get("data", {})
            ]
            if satisfaction_values:
                self.metrics["user_satisfaction"][component] = sum(satisfaction_values) / len(satisfaction_values)
            
            # Calculate system health if available
            health_values = [
                f["data"].get("health") 
                for f in feedback_list 
                if "health" in f.get("data", {})
            ]
            if health_values:
                self.metrics["system_health"][component] = sum(health_values) / len(health_values)
    
    def get_feedback_history(self, 
                           component: str = None, 
                           feedback_type: FeedbackType = None,
                           min_severity: FeedbackSeverity = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """Get feedback history with optional filters
        
        Args:
            component: Filter by component
            feedback_type: Filter by feedback type
            min_severity: Minimum severity level
            limit: Maximum number of entries
            
        Returns:
            List of feedback entries
        """
        # Start with all feedback
        results = self.feedback_history.copy()
        
        # Apply filters
        if component:
            results = [f for f in results if f["component"] == component]
        
        if feedback_type:
            results = [f for f in results if f["type"] == feedback_type.value]
        
        if min_severity:
            results = [f for f in results if f["severity"] >= min_severity.value]
        
        # Sort by timestamp (newest first)
        results.sort(key=lambda f: f["timestamp"], reverse=True)
        
        # Apply limit
        return results[:limit]
    
    def get_corrections(self, 
                       component: str = None, 
                       applied_only: bool = False,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """Get corrections with optional filters
        
        Args:
            component: Filter by component
            applied_only: Only include applied corrections
            limit: Maximum number of entries
            
        Returns:
            List of correction entries
        """
        # Start with all corrections
        results = self.corrections.copy()
        
        # Apply filters
        if component:
            results = [c for c in results if c["component"] == component]
        
        if applied_only:
            results = [c for c in results if c.get("applied", False)]
        
        # Sort by timestamp (newest first)
        results.sort(key=lambda c: c["timestamp"], reverse=True)
        
        # Apply limit
        return results[:limit]
    
    def get_metrics(self, component: str = None) -> Dict[str, Any]:
        """Get performance metrics
        
        Args:
            component: Filter by component
            
        Returns:
            Dictionary with metrics
        """
        if component:
            # Return metrics for specific component
            component_metrics = {}
            for metric_type, metrics in self.metrics.items():
                if component in metrics:
                    component_metrics[metric_type] = metrics[component]
            return component_metrics
        else:
            # Return all metrics
            return self.metrics
    
    def stop(self):
        """Stop the feedback loop system"""
        self.running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        self._save_feedback()

# Example usage
if __name__ == "__main__":
    # Create feedback loop system
    feedback_system = FeedbackLoop()
    
    # Add some test feedback
    feedback_system.add_feedback(
        component="text_processor",
        feedback_type=FeedbackType.ACCURACY,
        message="Classification accuracy below threshold",
        severity=FeedbackSeverity.HIGH,
        data={"accuracy": 0.65, "threshold": 0.8}
    )
    
    feedback_system.add_feedback(
        component="memory_service",
        feedback_type=FeedbackType.PERFORMANCE,
        message="Memory retrieval time above threshold",
        severity=FeedbackSeverity.MEDIUM,
        data={"response_time": 1.5, "threshold": 1.0}
    )
    
    # Get feedback history
    history = feedback_system.get_feedback_history(limit=5)
    print(f"Recent feedback: {len(history)} entries")
    for entry in history:
        print(f"- {entry['component']}: {entry['message']} (Severity: {entry['severity']})")
    
    # Get corrections
    corrections = feedback_system.get_corrections()
    print(f"Corrections: {len(corrections)} entries")
    for correction in corrections:
        print(f"- {correction['component']}: {correction['action']} (Applied: {correction.get('applied', False)})")
    
    # Apply a correction if available
    if corrections:
        result = feedback_system.apply_correction(corrections[0]["id"])
        print(f"Applied correction: {result['success']}")
    
    # Get metrics
    metrics = feedback_system.get_metrics()
    print(f"Performance metrics: {metrics}")
    
    # Stop the system
    feedback_system.stop()