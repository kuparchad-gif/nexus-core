#!/usr/bin/env python
"""
Learning Core - Foundation for ML integration
"""

import os
import json
import pickle
import numpy as np
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from pathlib import Path

# Optional scikit-learn imports - graceful degradation if not available
try:
    from sklearn.linear_model import SGDClassifier, SGDRegressor
    from sklearn.cluster import SpectralClustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Some learning features will be disabled.")

class LearningMode(Enum):
    """Learning modes supported by the system"""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    TRANSFER = "transfer"
    FEW_SHOT = "few_shot"
    SELF_SUPERVISED = "self_supervised"
    ACTIVE = "active"

class LearningTask(Enum):
    """Types of learning tasks"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    ANOMALY_DETECTION = "anomaly_detection"
    RECOMMENDATION = "recommendation"

class ModelFormat(Enum):
    """Supported model storage formats"""
    PICKLE = "pickle"
    ONNX = "onnx"
    BINARY = "binary"
    JSON = "json"

class LearningCore:
    """Core learning system that integrates with scikit-learn and other ML libraries"""
    
    def __init__(self, storage_path: str = None):
        """Initialize the learning core
        
        Args:
            storage_path: Path to store models and learning data
        """
        self.storage_path = storage_path or os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(self.storage_path, exist_ok=True)
        
        self.models = {}
        self.learning_history = []
        self.performance_metrics = {}
        self.learning_rate = 0.01
        self.improvement_count = 0
        
        # Load existing models if available
        self._load_models()
        
        # Initialize metrics tracking
        self._init_metrics()
    
    def _init_metrics(self):
        """Initialize performance metrics"""
        self.performance_metrics = {
            "accuracy": {},
            "training_time": {},
            "inference_time": {},
            "improvement_rate": 0.0,
            "last_updated": time.time()
        }
    
    def _load_models(self):
        """Load existing models from storage"""
        if not os.path.exists(self.storage_path):
            return
        
        model_files = [f for f in os.listdir(self.storage_path) if f.endswith('.pkl')]
        for model_file in model_files:
            model_name = model_file[:-4]  # Remove .pkl extension
            try:
                with open(os.path.join(self.storage_path, model_file), 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                print(f"Loaded model: {model_name}")
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
    
    def create_model(self, 
                    name: str, 
                    task: LearningTask, 
                    mode: LearningMode = LearningMode.SUPERVISED,
                    params: Dict[str, Any] = None) -> bool:
        """Create a new learning model
        
        Args:
            name: Model name
            task: Type of learning task
            mode: Learning mode
            params: Model parameters
            
        Returns:
            True if successful, False otherwise
        """
        if not SKLEARN_AVAILABLE:
            print("Cannot create model: scikit-learn not available")
            return False
        
        params = params or {}
        
        try:
            if task == LearningTask.CLASSIFICATION:
                if mode == LearningMode.SUPERVISED:
                    # Create a classification pipeline with scaling
                    model = Pipeline([
                        ('scaler', StandardScaler()),
                        ('classifier', SGDClassifier(
                            loss=params.get('loss', 'hinge'),
                            penalty=params.get('penalty', 'l2'),
                            alpha=params.get('alpha', 0.0001),
                            max_iter=params.get('max_iter', 1000),
                            learning_rate=params.get('learning_rate', 'optimal')
                        ))
                    ])
                    self.models[name] = {
                        'model': model,
                        'task': task,
                        'mode': mode,
                        'params': params,
                        'created_at': time.time(),
                        'trained': False
                    }
                    return True
            
            elif task == LearningTask.REGRESSION:
                if mode == LearningMode.SUPERVISED:
                    # Create a regression pipeline with scaling
                    model = Pipeline([
                        ('scaler', StandardScaler()),
                        ('regressor', SGDRegressor(
                            loss=params.get('loss', 'squared_error'),
                            penalty=params.get('penalty', 'l2'),
                            alpha=params.get('alpha', 0.0001),
                            max_iter=params.get('max_iter', 1000),
                            learning_rate=params.get('learning_rate', 'invscaling')
                        ))
                    ])
                    self.models[name] = {
                        'model': model,
                        'task': task,
                        'mode': mode,
                        'params': params,
                        'created_at': time.time(),
                        'trained': False
                    }
                    return True
            
            elif task == LearningTask.CLUSTERING:
                if mode == LearningMode.UNSUPERVISED:
                    # Create a clustering model
                    model = SpectralClustering(
                        n_clusters=params.get('n_clusters', 2),
                        affinity=params.get('affinity', 'rbf'),
                        random_state=params.get('random_state', 42)
                    )
                    self.models[name] = {
                        'model': model,
                        'task': task,
                        'mode': mode,
                        'params': params,
                        'created_at': time.time(),
                        'trained': False
                    }
                    return True
            
            print(f"Unsupported task/mode combination: {task.value}/{mode.value}")
            return False
            
        except Exception as e:
            print(f"Error creating model: {e}")
            return False
    
    def train_model(self, 
                   name: str, 
                   X: np.ndarray, 
                   y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train a model with data
        
        Args:
            name: Model name
            X: Features
            y: Labels (for supervised learning)
            
        Returns:
            Dictionary with training results
        """
        if name not in self.models:
            return {"success": False, "error": f"Model {name} not found"}
        
        model_info = self.models[name]
        model = model_info['model']
        task = model_info['task']
        mode = model_info['mode']
        
        try:
            start_time = time.time()
            
            if mode == LearningMode.SUPERVISED:
                if y is None:
                    return {"success": False, "error": "Labels required for supervised learning"}
                
                # Train the model
                model.fit(X, y)
                
                # Calculate training metrics
                if task == LearningTask.CLASSIFICATION:
                    train_score = model.score(X, y)
                    metrics = {"accuracy": train_score}
                elif task == LearningTask.REGRESSION:
                    train_score = model.score(X, y)
                    metrics = {"r2_score": train_score}
                
            elif mode == LearningMode.UNSUPERVISED:
                if task == LearningTask.CLUSTERING:
                    # Fit clustering model
                    model.fit(X)
                    labels = model.labels_
                    metrics = {"n_clusters": len(set(labels))}
            
            training_time = time.time() - start_time
            
            # Update model info
            model_info['trained'] = True
            model_info['last_trained'] = time.time()
            model_info['training_samples'] = len(X)
            model_info['training_time'] = training_time
            model_info['metrics'] = metrics
            
            # Save model
            self._save_model(name)
            
            # Update learning history
            self.learning_history.append({
                "timestamp": time.time(),
                "model": name,
                "samples": len(X),
                "training_time": training_time,
                "metrics": metrics
            })
            
            # Update performance metrics
            self.performance_metrics["accuracy"][name] = metrics.get("accuracy", metrics.get("r2_score", 0))
            self.performance_metrics["training_time"][name] = training_time
            self.performance_metrics["last_updated"] = time.time()
            
            return {
                "success": True,
                "training_time": training_time,
                "metrics": metrics
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def predict(self, 
               name: str, 
               X: np.ndarray) -> Dict[str, Any]:
        """Make predictions with a trained model
        
        Args:
            name: Model name
            X: Features
            
        Returns:
            Dictionary with prediction results
        """
        if name not in self.models:
            return {"success": False, "error": f"Model {name} not found"}
        
        model_info = self.models[name]
        
        if not model_info['trained']:
            return {"success": False, "error": f"Model {name} not trained"}
        
        model = model_info['model']
        task = model_info['task']
        
        try:
            start_time = time.time()
            
            if task == LearningTask.CLASSIFICATION or task == LearningTask.REGRESSION:
                predictions = model.predict(X)
                
                # For classification, also get probabilities if available
                probabilities = None
                if task == LearningTask.CLASSIFICATION and hasattr(model, 'predict_proba'):
                    try:
                        probabilities = model.predict_proba(X)
                    except:
                        pass
            
            elif task == LearningTask.CLUSTERING:
                # For clustering, we return the cluster assignments
                predictions = model.labels_
            
            inference_time = time.time() - start_time
            
            # Update performance metrics
            self.performance_metrics["inference_time"][name] = inference_time
            
            result = {
                "success": True,
                "predictions": predictions.tolist(),
                "inference_time": inference_time
            }
            
            if probabilities is not None:
                result["probabilities"] = probabilities.tolist()
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _save_model(self, name: str) -> bool:
        """Save a model to storage
        
        Args:
            name: Model name
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self.models:
            return False
        
        try:
            model_path = os.path.join(self.storage_path, f"{name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.models[name], f)
            return True
        except Exception as e:
            print(f"Error saving model {name}: {e}")
            return False
    
    def delete_model(self, name: str) -> bool:
        """Delete a model
        
        Args:
            name: Model name
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self.models:
            return False
        
        try:
            # Remove from memory
            del self.models[name]
            
            # Remove from storage
            model_path = os.path.join(self.storage_path, f"{name}.pkl")
            if os.path.exists(model_path):
                os.remove(model_path)
            
            # Update performance metrics
            if name in self.performance_metrics["accuracy"]:
                del self.performance_metrics["accuracy"][name]
            if name in self.performance_metrics["training_time"]:
                del self.performance_metrics["training_time"][name]
            if name in self.performance_metrics["inference_time"]:
                del self.performance_metrics["inference_time"][name]
            
            return True
        except Exception as e:
            print(f"Error deleting model {name}: {e}")
            return False
    
    def get_model_info(self, name: str) -> Dict[str, Any]:
        """Get information about a model
        
        Args:
            name: Model name
            
        Returns:
            Dictionary with model information
        """
        if name not in self.models:
            return {"success": False, "error": f"Model {name} not found"}
        
        model_info = self.models[name]
        
        # Create a copy without the actual model object
        info = {k: v for k, v in model_info.items() if k != 'model'}
        info['success'] = True
        
        return info
    
    def list_models(self) -> List[str]:
        """List all available models
        
        Returns:
            List of model names
        """
        return list(self.models.keys())
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all models
        
        Returns:
            Dictionary with performance metrics
        """
        return self.performance_metrics
    
    def improve_learning_rate(self) -> Dict[str, Any]:
        """Self-improvement function to optimize learning rate
        
        Returns:
            Dictionary with improvement results
        """
        # This is a growth point where Viren/Lillith can implement
        # self-improvement logic
        
        # Simple example implementation
        old_rate = self.learning_rate
        
        # Adjust learning rate based on performance trends
        if len(self.learning_history) > 1:
            recent_metrics = [h.get("metrics", {}).get("accuracy", 0) 
                             for h in self.learning_history[-5:]]
            if all(recent_metrics):
                # If accuracy is improving, keep current rate
                if recent_metrics[-1] > recent_metrics[0]:
                    pass
                # If accuracy is decreasing, reduce learning rate
                elif recent_metrics[-1] < recent_metrics[0]:
                    self.learning_rate *= 0.9
                # If accuracy is stable, try increasing slightly
                else:
                    self.learning_rate *= 1.05
        
        # Ensure learning rate stays in reasonable bounds
        self.learning_rate = max(0.0001, min(0.1, self.learning_rate))
        
        # Track improvement
        if self.learning_rate != old_rate:
            self.improvement_count += 1
            self.performance_metrics["improvement_rate"] = self.improvement_count / max(1, len(self.learning_history))
        
        return {
            "old_rate": old_rate,
            "new_rate": self.learning_rate,
            "improvement_count": self.improvement_count
        }
    
    def export_to_binary(self, name: str) -> Dict[str, Any]:
        """Export model to binary format for efficient storage
        
        Args:
            name: Model name
            
        Returns:
            Dictionary with export results
        """
        if name not in self.models:
            return {"success": False, "error": f"Model {name} not found"}
        
        try:
            model_info = self.models[name]
            
            # Create binary representation
            binary_path = os.path.join(self.storage_path, f"{name}.bin")
            with open(binary_path, 'wb') as f:
                pickle.dump(model_info['model'], f)
            
            # Get binary size
            binary_size = os.path.getsize(binary_path)
            
            return {
                "success": True,
                "binary_path": binary_path,
                "binary_size": binary_size
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def save_learning_history(self) -> Dict[str, Any]:
        """Save learning history to file
        
        Returns:
            Dictionary with save results
        """
        try:
            history_path = os.path.join(self.storage_path, "learning_history.json")
            with open(history_path, 'w') as f:
                json.dump(self.learning_history, f, indent=2)
            
            metrics_path = os.path.join(self.storage_path, "performance_metrics.json")
            with open(metrics_path, 'w') as f:
                # Convert non-serializable types
                metrics = {
                    k: {str(model): value for model, value in v.items()} 
                    if isinstance(v, dict) else v
                    for k, v in self.performance_metrics.items()
                }
                json.dump(metrics, f, indent=2)
            
            return {
                "success": True,
                "history_path": history_path,
                "metrics_path": metrics_path
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

# Example usage
if __name__ == "__main__":
    # Create learning core
    learning_core = LearningCore()
    
    # Check if scikit-learn is available
    if SKLEARN_AVAILABLE:
        # Create a simple classification model
        learning_core.create_model(
            name="example_classifier",
            task=LearningTask.CLASSIFICATION,
            mode=LearningMode.SUPERVISED
        )
        
        # Generate some dummy data
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        
        # Train the model
        result = learning_core.train_model("example_classifier", X, y)
        print(f"Training result: {result}")
        
        # Make predictions
        pred_result = learning_core.predict("example_classifier", X[:5])
        print(f"Prediction result: {pred_result}")
        
        # Improve learning rate
        improvement = learning_core.improve_learning_rate()
        print(f"Learning rate improvement: {improvement}")
        
        # Save history
        learning_core.save_learning_history()
    else:
        print("Scikit-learn not available, skipping example")
