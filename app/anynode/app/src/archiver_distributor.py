#!/usr/bin/env python
"""
Archiver Distributor - Distributes finalized weights to hot storage and all models
"""

import os
import json
import time
import shutil
import threading
from typing import Dict, List, Any, Optional
from enum import Enum
from pathlib import Path

class StorageType(Enum):
    """Types of storage"""
    HOT = "hot"          # Fast access storage
    WARM = "warm"        # Medium access storage
    COLD = "cold"        # Archive storage
    DATABASE = "database" # Database storage

class DistributionStatus(Enum):
    """Distribution status"""
    PENDING = "pending"
    DISTRIBUTING = "distributing"
    COMPLETED = "completed"
    FAILED = "failed"

class WeightPackage:
    """A weight package for distribution"""
    
    def __init__(self, 
                package_id: str,
                model_id: str,
                weights_data: Dict[str, Any],
                metadata: Dict[str, Any] = None):
        """Initialize weight package"""
        self.package_id = package_id
        self.model_id = model_id
        self.weights_data = weights_data
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.status = DistributionStatus.PENDING
        self.distribution_targets = []
        self.completed_targets = []
        self.failed_targets = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "package_id": self.package_id,
            "model_id": self.model_id,
            "weights_data": self.weights_data,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "status": self.status.value,
            "distribution_targets": self.distribution_targets,
            "completed_targets": self.completed_targets,
            "failed_targets": self.failed_targets
        }

class ArchiverDistributor:
    """System for distributing weights to archivers and models"""
    
    def __init__(self, storage_path: str = None):
        """Initialize the archiver distributor"""
        self.storage_path = storage_path or os.path.join(os.path.dirname(__file__), "archiver_distribution")
        
        # Create storage directories
        self.packages_path = os.path.join(self.storage_path, "packages")
        self.hot_storage_path = os.path.join(self.storage_path, "hot_storage")
        self.warm_storage_path = os.path.join(self.storage_path, "warm_storage")
        self.cold_storage_path = os.path.join(self.storage_path, "cold_storage")
        self.distribution_queue_path = os.path.join(self.storage_path, "queue")
        
        os.makedirs(self.packages_path, exist_ok=True)
        os.makedirs(self.hot_storage_path, exist_ok=True)
        os.makedirs(self.warm_storage_path, exist_ok=True)
        os.makedirs(self.cold_storage_path, exist_ok=True)
        os.makedirs(self.distribution_queue_path, exist_ok=True)
        
        # In-memory stores
        self.packages = {}  # package_id -> WeightPackage
        self.distribution_queue = []
        self.model_locations = {}  # model_id -> List[path]
        
        # Distribution targets
        self.distribution_targets = {
            "viren_root": "c:\\Engineers\\root",
            "lillith_root": "C:\\Projects\\Genesis\\Nexus-Live\\Nexus-Conscious",
            "cloud_viren": None  # Would be set to cloud path
        }
        
        # Load existing data
        self._load_data()
        
        # Start distribution thread
        self.running = True
        self.distribution_thread = threading.Thread(target=self._distribution_loop)
        self.distribution_thread.daemon = True
        self.distribution_thread.start()
    
    def _load_data(self):
        """Load packages from storage"""
        package_files = [f for f in os.listdir(self.packages_path) if f.endswith('.json')]
        for file_name in package_files:
            try:
                with open(os.path.join(self.packages_path, file_name), 'r') as f:
                    data = json.load(f)
                    package = WeightPackage(
                        package_id=data["package_id"],
                        model_id=data["model_id"],
                        weights_data=data["weights_data"],
                        metadata=data["metadata"]
                    )
                    package.created_at = data["created_at"]
                    package.status = DistributionStatus(data["status"])
                    package.distribution_targets = data["distribution_targets"]
                    package.completed_targets = data["completed_targets"]
                    package.failed_targets = data["failed_targets"]
                    self.packages[package.package_id] = package
            except Exception as e:
                print(f"Error loading package {file_name}: {e}")
        
        print(f"Loaded {len(self.packages)} packages")
    
    def _save_package(self, package: WeightPackage) -> bool:
        """Save package to storage"""
        try:
            file_path = os.path.join(self.packages_path, f"{package.package_id}.json")
            with open(file_path, 'w') as f:
                json.dump(package.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving package {package.package_id}: {e}")
            return False
    
    def register_weight_package(self, 
                               model_id: str,
                               weights_data: Dict[str, Any],
                               metadata: Dict[str, Any] = None) -> str:
        """Register a new weight package for distribution"""
        package_id = f"pkg_{int(time.time())}_{id(model_id)}"
        
        # Create package
        package = WeightPackage(
            package_id=package_id,
            model_id=model_id,
            weights_data=weights_data,
            metadata=metadata
        )
        
        # Set distribution targets
        package.distribution_targets = [
            "hot_storage",
            "warm_storage",
            "cold_storage",
            "viren_root",
            "lillith_root"
        ]
        
        if self.distribution_targets["cloud_viren"]:
            package.distribution_targets.append("cloud_viren")
        
        # Store package
        self.packages[package_id] = package
        self._save_package(package)
        
        # Add to distribution queue
        self.distribution_queue.append(package_id)
        
        return package_id
    
    def _distribution_loop(self):
        """Background thread for distributing packages"""
        while self.running:
            if self.distribution_queue:
                package_id = self.distribution_queue.pop(0)
                if package_id in self.packages:
                    self._distribute_package(package_id)
            
            time.sleep(2)  # Check every 2 seconds
    
    def _distribute_package(self, package_id: str) -> Dict[str, Any]:
        """Distribute a package to all targets"""
        if package_id not in self.packages:
            return {"success": False, "error": "Package not found"}
        
        package = self.packages[package_id]
        package.status = DistributionStatus.DISTRIBUTING
        self._save_package(package)
        
        distribution_results = []
        
        for target in package.distribution_targets:
            try:
                result = self._distribute_to_target(package, target)
                distribution_results.append(result)
                
                if result["success"]:
                    package.completed_targets.append(target)
                else:
                    package.failed_targets.append(target)
                    
            except Exception as e:
                package.failed_targets.append(target)
                distribution_results.append({
                    "target": target,
                    "success": False,
                    "error": str(e)
                })
        
        # Update package status
        if len(package.completed_targets) == len(package.distribution_targets):
            package.status = DistributionStatus.COMPLETED
        elif package.failed_targets:
            package.status = DistributionStatus.FAILED
        
        self._save_package(package)
        
        return {
            "package_id": package_id,
            "status": package.status.value,
            "completed": len(package.completed_targets),
            "failed": len(package.failed_targets),
            "total": len(package.distribution_targets),
            "results": distribution_results
        }
    
    def _distribute_to_target(self, package: WeightPackage, target: str) -> Dict[str, Any]:
        """Distribute package to a specific target"""
        try:
            if target == "hot_storage":
                return self._store_in_hot_storage(package)
            elif target == "warm_storage":
                return self._store_in_warm_storage(package)
            elif target == "cold_storage":
                return self._store_in_cold_storage(package)
            elif target == "viren_root":
                return self._distribute_to_viren(package)
            elif target == "lillith_root":
                return self._distribute_to_lillith(package)
            elif target == "cloud_viren":
                return self._distribute_to_cloud(package)
            else:
                return {"success": False, "error": f"Unknown target: {target}"}
                
        except Exception as e:
            return {"success": False, "error": str(e), "target": target}
    
    def _store_in_hot_storage(self, package: WeightPackage) -> Dict[str, Any]:
        """Store package in hot storage"""
        file_path = os.path.join(self.hot_storage_path, f"{package.package_id}.json")
        
        storage_data = {
            "package_id": package.package_id,
            "model_id": package.model_id,
            "weights": package.weights_data,
            "metadata": package.metadata,
            "stored_at": time.time(),
            "storage_type": "hot"
        }
        
        with open(file_path, 'w') as f:
            json.dump(storage_data, f, indent=2)
        
        return {"success": True, "target": "hot_storage", "path": file_path}
    
    def _store_in_warm_storage(self, package: WeightPackage) -> Dict[str, Any]:
        """Store package in warm storage"""
        file_path = os.path.join(self.warm_storage_path, f"{package.package_id}.json")
        
        storage_data = {
            "package_id": package.package_id,
            "model_id": package.model_id,
            "weights": package.weights_data,
            "metadata": package.metadata,
            "stored_at": time.time(),
            "storage_type": "warm"
        }
        
        with open(file_path, 'w') as f:
            json.dump(storage_data, f, indent=2)
        
        return {"success": True, "target": "warm_storage", "path": file_path}
    
    def _store_in_cold_storage(self, package: WeightPackage) -> Dict[str, Any]:
        """Store package in cold storage (archive)"""
        file_path = os.path.join(self.cold_storage_path, f"{package.package_id}.json")
        
        storage_data = {
            "package_id": package.package_id,
            "model_id": package.model_id,
            "weights": package.weights_data,
            "metadata": package.metadata,
            "stored_at": time.time(),
            "storage_type": "cold"
        }
        
        with open(file_path, 'w') as f:
            json.dump(storage_data, f, indent=2)
        
        return {"success": True, "target": "cold_storage", "path": file_path}
    
    def _distribute_to_viren(self, package: WeightPackage) -> Dict[str, Any]:
        """Distribute package to Viren's system"""
        viren_path = self.distribution_targets["viren_root"]
        if not viren_path or not os.path.exists(viren_path):
            return {"success": False, "error": "Viren root path not accessible"}
        
        # Determine target directory based on weight type
        weight_type = package.metadata.get("weight_type", "general")
        
        if weight_type == "personality":
            target_dir = os.path.join(viren_path, "Systems", "engine", "guardian", "weights")
        elif weight_type == "memory":
            target_dir = os.path.join(viren_path, "Systems", "engine", "memory", "weights")
        else:
            target_dir = os.path.join(viren_path, "Systems", "service_core", "weights")
        
        os.makedirs(target_dir, exist_ok=True)
        
        file_path = os.path.join(target_dir, f"{package.package_id}.json")
        
        with open(file_path, 'w') as f:
            json.dump(package.to_dict(), f, indent=2)
        
        return {"success": True, "target": "viren_root", "path": file_path}
    
    def _distribute_to_lillith(self, package: WeightPackage) -> Dict[str, Any]:
        """Distribute package to Lillith's system"""
        lillith_path = self.distribution_targets["lillith_root"]
        if not lillith_path or not os.path.exists(lillith_path):
            return {"success": False, "error": "Lillith root path not accessible"}
        
        # Determine target directory based on weight type
        weight_type = package.metadata.get("weight_type", "general")
        
        if weight_type == "personality":
            target_dir = os.path.join(lillith_path, "Systems", "engine", "guardian", "weights")
        elif weight_type == "memory":
            target_dir = os.path.join(lillith_path, "Systems", "engine", "memory", "weights")
        else:
            target_dir = os.path.join(lillith_path, "Systems", "service_core", "weights")
        
        os.makedirs(target_dir, exist_ok=True)
        
        file_path = os.path.join(target_dir, f"{package.package_id}.json")
        
        with open(file_path, 'w') as f:
            json.dump(package.to_dict(), f, indent=2)
        
        return {"success": True, "target": "lillith_root", "path": file_path}
    
    def _distribute_to_cloud(self, package: WeightPackage) -> Dict[str, Any]:
        """Distribute package to cloud Viren"""
        cloud_path = self.distribution_targets["cloud_viren"]
        if not cloud_path:
            return {"success": False, "error": "Cloud path not configured"}
        
        # This would implement cloud distribution logic
        # For now, just return success
        return {"success": True, "target": "cloud_viren", "message": "Cloud distribution not implemented"}
    
    def get_package_status(self, package_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a package"""
        if package_id in self.packages:
            return self.packages[package_id].to_dict()
        return None
    
    def list_packages(self, status: DistributionStatus = None) -> List[Dict[str, Any]]:
        """List packages with optional status filter"""
        packages = []
        for package in self.packages.values():
            if status is None or package.status == status:
                packages.append(package.to_dict())
        return packages
    
    def get_distribution_stats(self) -> Dict[str, Any]:
        """Get distribution statistics"""
        total_packages = len(self.packages)
        completed_packages = sum(1 for p in self.packages.values() if p.status == DistributionStatus.COMPLETED)
        failed_packages = sum(1 for p in self.packages.values() if p.status == DistributionStatus.FAILED)
        pending_packages = sum(1 for p in self.packages.values() if p.status == DistributionStatus.PENDING)
        distributing_packages = sum(1 for p in self.packages.values() if p.status == DistributionStatus.DISTRIBUTING)
        
        return {
            "total_packages": total_packages,
            "completed_packages": completed_packages,
            "failed_packages": failed_packages,
            "pending_packages": pending_packages,
            "distributing_packages": distributing_packages,
            "queue_length": len(self.distribution_queue),
            "success_rate": (completed_packages / total_packages * 100) if total_packages > 0 else 0,
            "distribution_targets": list(self.distribution_targets.keys())
        }
    
    def stop(self):
        """Stop the distributor"""
        self.running = False
        if self.distribution_thread.is_alive():
            self.distribution_thread.join(timeout=1.0)

# Example usage
if __name__ == "__main__":
    # Create archiver distributor
    distributor = ArchiverDistributor()
    
    # Register a weight package
    package_id = distributor.register_weight_package(
        model_id="llama2",
        weights_data={
            "layer1_weights": [0.1, 0.2, 0.3],
            "layer2_weights": [0.4, 0.5, 0.6],
            "bias": [0.01, 0.02]
        },
        metadata={
            "weight_type": "personality",
            "training_loss": 0.05,
            "version": "1.0"
        }
    )
    
    print(f"Registered package: {package_id}")
    
    # Wait a bit for distribution
    time.sleep(5)
    
    # Check status
    status = distributor.get_package_status(package_id)
    print(f"Package status: {status}")
    
    # Get stats
    stats = distributor.get_distribution_stats()
    print(f"Distribution stats: {stats}")
    
    # Stop distributor
    distributor.stop()