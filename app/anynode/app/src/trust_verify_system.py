#!/usr/bin/env python
"""
Trust but Verify System - Confidence-based prediction with failure learning
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

class ConfidenceLevel(Enum):
    """Confidence levels for predictions"""
    LOW = 0.0
    MEDIUM = 0.5
    HIGH = 0.75
    CRITICAL = 0.88

class PredictionResult:
    """Result of a prediction attempt"""
    
    def __init__(self, 
                query: str,
                prediction: str,
                confidence: float,
                actual_result: str = None,
                success: bool = None):
        """Initialize prediction result"""
        self.id = f"pred_{int(time.time())}_{id(query)}"
        self.query = query
        self.prediction = prediction
        self.confidence = confidence
        self.actual_result = actual_result
        self.success = success
        self.timestamp = time.time()
        self.trained = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "query": self.query,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "actual_result": self.actual_result,
            "success": self.success,
            "timestamp": self.timestamp,
            "trained": self.trained
        }

class TrustVerifySystem:
    """System for confidence-based predictions with failure learning"""
    
    def __init__(self, storage_path: str = None, confidence_threshold: float = 0.88):
        """Initialize trust-verify system"""
        self.storage_path = storage_path or os.path.join(os.path.dirname(__file__), "trust_verify")
        self.confidence_threshold = confidence_threshold
        
        # Create storage directories
        self.predictions_path = os.path.join(self.storage_path, "predictions")
        self.failures_path = os.path.join(self.storage_path, "failures")
        self.training_path = os.path.join(self.storage_path, "training")
        
        os.makedirs(self.predictions_path, exist_ok=True)
        os.makedirs(self.failures_path, exist_ok=True)
        os.makedirs(self.training_path, exist_ok=True)
        
        # In-memory stores
        self.predictions = {}  # prediction_id -> PredictionResult
        self.failure_patterns = {}  # pattern -> count
        self.accuracy_stats = {"correct": 0, "incorrect": 0, "total": 0}
        
        # Load existing data
        self._load_data()
    
    def _load_data(self):
        """Load predictions and failures from storage"""
        # Load predictions
        pred_files = [f for f in os.listdir(self.predictions_path) if f.endswith('.json')]
        for file_name in pred_files:
            try:
                with open(os.path.join(self.predictions_path, file_name), 'r') as f:
                    data = json.load(f)
                    pred = PredictionResult(
                        query=data["query"],
                        prediction=data["prediction"],
                        confidence=data["confidence"],
                        actual_result=data["actual_result"],
                        success=data["success"]
                    )
                    pred.id = data["id"]
                    pred.timestamp = data["timestamp"]
                    pred.trained = data["trained"]
                    self.predictions[pred.id] = pred
                    
                    # Update stats
                    if pred.success is not None:
                        self.accuracy_stats["total"] += 1
                        if pred.success:
                            self.accuracy_stats["correct"] += 1
                        else:
                            self.accuracy_stats["incorrect"] += 1
            except Exception as e:
                print(f"Error loading prediction {file_name}: {e}")
        
        print(f"Loaded {len(self.predictions)} predictions")
    
    def _save_prediction(self, prediction: PredictionResult) -> bool:
        """Save prediction to storage"""
        try:
            file_path = os.path.join(self.predictions_path, f"{prediction.id}.json")
            with open(file_path, 'w') as f:
                json.dump(prediction.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving prediction {prediction.id}: {e}")
            return False
    
    def should_predict(self, confidence: float) -> bool:
        """Determine if prediction should be made based on confidence"""
        return confidence >= self.confidence_threshold
    
    def make_prediction(self, query: str, prediction_func, confidence_func) -> Dict[str, Any]:
        """Make a prediction with confidence checking"""
        # Calculate confidence
        confidence = confidence_func(query)
        
        # Check if we should predict
        if self.should_predict(confidence):
            # Make prediction
            prediction = prediction_func(query)
            
            # Create prediction result
            pred_result = PredictionResult(
                query=query,
                prediction=prediction,
                confidence=confidence
            )
            
            # Store prediction
            self.predictions[pred_result.id] = pred_result
            self._save_prediction(pred_result)
            
            return {
                "predicted": True,
                "prediction": prediction,
                "confidence": confidence,
                "prediction_id": pred_result.id
            }
        else:
            # Don't predict, ask for verification
            return {
                "predicted": False,
                "confidence": confidence,
                "message": f"Confidence {confidence:.2f} below threshold {self.confidence_threshold}"
            }
    
    def verify_prediction(self, prediction_id: str, actual_result: str) -> Dict[str, Any]:
        """Verify a prediction result"""
        if prediction_id not in self.predictions:
            return {"success": False, "error": "Prediction not found"}
        
        prediction = self.predictions[prediction_id]
        
        # Set actual result
        prediction.actual_result = actual_result
        prediction.success = prediction.prediction.lower().strip() == actual_result.lower().strip()
        
        # Update accuracy stats
        self.accuracy_stats["total"] += 1
        if prediction.success:
            self.accuracy_stats["correct"] += 1
        else:
            self.accuracy_stats["incorrect"] += 1
            # Log failure for training
            self._log_failure(prediction)
        
        # Save updated prediction
        self._save_prediction(prediction)
        
        return {
            "success": True,
            "correct": prediction.success,
            "accuracy": self.get_accuracy()
        }
    
    def _log_failure(self, prediction: PredictionResult):
        """Log a failed prediction for training"""
        failure_data = {
            "prediction_id": prediction.id,
            "query": prediction.query,
            "predicted": prediction.prediction,
            "actual": prediction.actual_result,
            "confidence": prediction.confidence,
            "timestamp": time.time(),
            "pattern": self._extract_pattern(prediction.query)
        }
        
        # Save failure
        failure_file = os.path.join(self.failures_path, f"failure_{prediction.id}.json")
        try:
            with open(failure_file, 'w') as f:
                json.dump(failure_data, f, indent=2)
        except Exception as e:
            print(f"Error saving failure: {e}")
        
        # Update failure patterns
        pattern = failure_data["pattern"]
        if pattern not in self.failure_patterns:
            self.failure_patterns[pattern] = 0
        self.failure_patterns[pattern] += 1
    
    def _extract_pattern(self, query: str) -> str:
        """Extract pattern from query for failure analysis"""
        # Simple pattern extraction - first 3 words
        words = query.lower().split()
        return " ".join(words[:3]) if len(words) >= 3 else query.lower()
    
    def get_accuracy(self) -> float:
        """Get current accuracy rate"""
        if self.accuracy_stats["total"] == 0:
            return 0.0
        return self.accuracy_stats["correct"] / self.accuracy_stats["total"]
    
    def get_training_data(self) -> List[Dict[str, Any]]:
        """Get failed predictions for training"""
        training_data = []
        
        for prediction in self.predictions.values():
            if prediction.success is False and not prediction.trained:
                training_data.append({
                    "input": prediction.query,
                    "expected_output": prediction.actual_result,
                    "failed_prediction": prediction.prediction,
                    "confidence": prediction.confidence,
                    "pattern": self._extract_pattern(prediction.query)
                })
        
        return training_data
    
    def mark_trained(self, prediction_ids: List[str]) -> int:
        """Mark predictions as trained"""
        count = 0
        for pred_id in prediction_ids:
            if pred_id in self.predictions:
                self.predictions[pred_id].trained = True
                self._save_prediction(self.predictions[pred_id])
                count += 1
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "accuracy": self.get_accuracy(),
            "total_predictions": self.accuracy_stats["total"],
            "correct_predictions": self.accuracy_stats["correct"],
            "failed_predictions": self.accuracy_stats["incorrect"],
            "confidence_threshold": self.confidence_threshold,
            "failure_patterns": dict(sorted(self.failure_patterns.items(), key=lambda x: x[1], reverse=True)[:10])
        }

# Example usage
if __name__ == "__main__":
    # Create trust-verify system
    system = TrustVerifySystem(confidence_threshold=0.88)
    
    # Example prediction function
    def simple_prediction(query):
        return f"Predicted answer for: {query}"
    
    # Example confidence function
    def simple_confidence(query):
        # Simple confidence based on query length
        return min(0.95, len(query) / 100.0)
    
    # Test prediction
    result = system.make_prediction(
        "What is the capital of France?",
        simple_prediction,
        simple_confidence
    )
    
    print(f"Prediction result: {result}")
    
    # Verify prediction if made
    if result["predicted"]:
        verification = system.verify_prediction(
            result["prediction_id"],
            "Paris"
        )
        print(f"Verification result: {verification}")
    
    # Get training data
    training_data = system.get_training_data()
    print(f"Training data: {len(training_data)} items")
    
    # Get stats
    stats = system.get_stats()
    print(f"System stats: {stats}")