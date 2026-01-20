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
    HOT_STORAGE  =  "hot_storage"
    DATABASE  =  "database"
    CLOUD_BACKUP  =  "cloud_backup"
    LOCAL_CACHE  =  "local_cache"

class DistributionStatus(Enum):
    """Distribution status"""
    PENDING  =  "pending"
    IN_PROGRESS  =  "in_progress"
    COMPLETED  =  "completed"
    FAILED  =  "failed"

class WeightDistribution:
    """A weight distribution package"""

    def __init__(self,
                template_id: str,
                model_id: str,
                weights_data: Dict[str, Any],
                metadata: Dict[str, Any]  =  None):
        """Initialize weight distribution"""
        self.id  =  f"dist_{int(time.time())}_{id(template_id)}"
        self.template_id  =  template_id
        self.model_id  =  model_id
        self.weights_data  =  weights_data
        self.metadata  =  metadata or {}
        self.created_at  =  time.time()
        self.status  =  DistributionStatus.PENDING
        self.distribution_targets  =  []
        self.completed_targets  =  []
        self.failed_targets  =  []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "template_id": self.template_id,
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
    """System for distributing weights to archivers and all models"""

    def __init__(self, storage_path: str  =  None):
        """Initialize the archiver distributor"""
        self.storage_path  =  storage_path or os.path.join(os.path.dirname(__file__), "archiver_distribution")

        # Create storage directories
        self.distributions_path  =  os.path.join(self.storage_path, "distributions")
        self.hot_storage_path  =  os.path.join(self.storage_path, "hot_storage")
        self.database_path  =  os.path.join(self.storage_path, "database")
        self.queue_path  =  os.path.join(self.storage_path, "queue")

        os.makedirs(self.distributions_path, exist_ok = True)
        os.makedirs(self.hot_storage_path, exist_ok = True)
        os.makedirs(self.database_path, exist_ok = True)
        os.makedirs(self.queue_path, exist_ok = True)

        # In-memory stores
        self.distributions  =  {}  # distribution_id -> WeightDistribution
        self.distribution_queue  =  []
        self.processing  =  False

        # Model locations
        self.model_locations  =  {
            "viren": "c:\\Engineers\\root",
            "lillith": "C:\\Projects\\Genesis\\Nexus-Live\\Nexus-Conscious",
            "cloud": "cloud_storage_path"  # Would be configured
        }

        # Load existing data
        self._load_data()

        # Start distribution thread
        self.running  =  True
        self.distribution_thread  =  threading.Thread(target = self._distribution_loop)
        self.distribution_thread.daemon  =  True
        self.distribution_thread.start()

    def _load_data(self):
        """Load distributions from storage"""
        dist_files  =  [f for f in os.listdir(self.distributions_path) if f.endswith('.json')]
        for file_name in dist_files:
            try:
                with open(os.path.join(self.distributions_path, file_name), 'r') as f:
                    data  =  json.load(f)
                    dist  =  WeightDistribution(
                        template_id = data["template_id"],
                        model_id = data["model_id"],
                        weights_data = data["weights_data"],
                        metadata = data["metadata"]
                    )
                    dist.id  =  data["id"]
                    dist.created_at  =  data["created_at"]
                    dist.status  =  DistributionStatus(data["status"])
                    dist.distribution_targets  =  data["distribution_targets"]
                    dist.completed_targets  =  data["completed_targets"]
                    dist.failed_targets  =  data["failed_targets"]
                    self.distributions[dist.id]  =  dist
            except Exception as e:
                print(f"Error loading distribution {file_name}: {e}")

        print(f"Loaded {len(self.distributions)} distributions")

    def _save_distribution(self, distribution: WeightDistribution) -> bool:
        """Save distribution to storage"""
        try:
            file_path  =  os.path.join(self.distributions_path, f"{distribution.id}.json")
            with open(file_path, 'w') as f:
                json.dump(distribution.to_dict(), f, indent = 2)
            return True
        except Exception as e:
            print(f"Error saving distribution {distribution.id}: {e}")
            return False

    def create_distribution(self,
                          template_id: str,
                          model_id: str,
                          weights_data: Dict[str, Any],
                          metadata: Dict[str, Any]  =  None,
                          targets: List[str]  =  None) -> str:
        """Create a new weight distribution"""
        # Create distribution
        distribution  =  WeightDistribution(
            template_id = template_id,
            model_id = model_id,
            weights_data = weights_data,
            metadata = metadata
        )

        # Set distribution targets
        if targets:
            distribution.distribution_targets  =  targets
        else:
            # Default targets: hot storage, database, all models
            distribution.distribution_targets  =  [
                "hot_storage",
                "database",
                "viren",
                "lillith",
                "cloud"
            ]

        # Store distribution
        self.distributions[distribution.id]  =  distribution

        # Save to storage
        self._save_distribution(distribution)

        # Add to queue
        self.distribution_queue.append(distribution.id)

        return distribution.id

    def _distribution_loop(self):
        """Background thread for processing distributions"""
        while self.running:
            if self.distribution_queue and not self.processing:
                distribution_id  =  self.distribution_queue.pop(0)
                if distribution_id in self.distributions:
                    self._process_distribution(distribution_id)

            time.sleep(2)  # Check every 2 seconds

    def _process_distribution(self, distribution_id: str):
        """Process a distribution"""
        if distribution_id not in self.distributions:
            return

        distribution  =  self.distributions[distribution_id]

        if distribution.status != DistributionStatus.PENDING:
            return

        self.processing  =  True
        distribution.status  =  DistributionStatus.IN_PROGRESS
        self._save_distribution(distribution)

        try:
            # Process each target
            for target in distribution.distribution_targets:
                try:
                    if target == "hot_storage":
                        self._distribute_to_hot_storage(distribution)
                    elif target == "database":
                        self._distribute_to_database(distribution)
                    elif target in self.model_locations:
                        self._distribute_to_model(distribution, target)
                    else:
                        print(f"Unknown target: {target}")
                        distribution.failed_targets.append(target)
                        continue

                    distribution.completed_targets.append(target)

                except Exception as e:
                    print(f"Failed to distribute to {target}: {e}")
                    distribution.failed_targets.append(target)

            # Update status
            if len(distribution.completed_targets) == len(distribution.distribution_targets):
                distribution.status  =  DistributionStatus.COMPLETED
            elif distribution.completed_targets:
                distribution.status  =  DistributionStatus.COMPLETED  # Partial success still counts
            else:
                distribution.status  =  DistributionStatus.FAILED

            self._save_distribution(distribution)

        except Exception as e:
            distribution.status  =  DistributionStatus.FAILED
            self._save_distribution(distribution)
            print(f"Distribution {distribution_id} failed: {e}")

        finally:
            self.processing  =  False

    def _distribute_to_hot_storage(self, distribution: WeightDistribution):
        """Distribute weights to hot storage"""
        # Create hot storage file
        hot_storage_file  =  os.path.join(
            self.hot_storage_path,
            f"weights_{distribution.model_id}_{distribution.template_id}.json"
        )

        hot_storage_data  =  {
            "distribution_id": distribution.id,
            "model_id": distribution.model_id,
            "template_id": distribution.template_id,
            "weights": distribution.weights_data,
            "metadata": distribution.metadata,
            "stored_at": time.time(),
            "access_count": 0,
            "last_accessed": None
        }

        with open(hot_storage_file, 'w') as f:
            json.dump(hot_storage_data, f, indent = 2)

    def _distribute_to_database(self, distribution: WeightDistribution):
        """Distribute weights to database storage"""
        # Create database record
        db_file  =  os.path.join(
            self.database_path,
            f"db_record_{distribution.id}.json"
        )

        db_record  =  {
            "id": distribution.id,
            "model_id": distribution.model_id,
            "template_id": distribution.template_id,
            "weights_hash": hash(str(distribution.weights_data)),
            "metadata": distribution.metadata,
            "created_at": distribution.created_at,
            "stored_at": time.time(),
            "size_bytes": len(json.dumps(distribution.weights_data)),
            "indexed": True
        }

        with open(db_file, 'w') as f:
            json.dump(db_record, f, indent = 2)

        # Store actual weights separately for efficiency
        weights_file  =  os.path.join(
            self.database_path,
            f"weights_{distribution.id}.json"
        )

        with open(weights_file, 'w') as f:
            json.dump(distribution.weights_data, f, indent = 2)

    def _distribute_to_model(self, distribution: WeightDistribution, model_name: str):
        """Distribute weights to a specific model"""
        model_path  =  self.model_locations.get(model_name)

        if not model_path:
            raise ValueError(f"Unknown model: {model_name}")

        # Create model-specific weights directory
        model_weights_dir  =  os.path.join(model_path, "Systems", "weights")
        os.makedirs(model_weights_dir, exist_ok = True)

        # Create weight file for the model
        weight_file  =  os.path.join(
            model_weights_dir,
            f"{distribution.model_id}_{distribution.template_id}.json"
        )

        model_weight_data  =  {
            "model_id": distribution.model_id,
            "template_id": distribution.template_id,
            "weights": distribution.weights_data,
            "metadata": distribution.metadata,
            "distributed_to": model_name,
            "distributed_at": time.time(),
            "active": True
        }

        with open(weight_file, 'w') as f:
            json.dump(model_weight_data, f, indent = 2)

    def get_distribution(self, distribution_id: str) -> Optional[Dict[str, Any]]:
        """Get a distribution by ID"""
        if distribution_id in self.distributions:
            return self.distributions[distribution_id].to_dict()
        return None

    def list_distributions(self,
                          status: DistributionStatus  =  None,
                          model_id: str  =  None) -> List[Dict[str, Any]]:
        """List distributions with optional filters"""
        results  =  []

        for distribution in self.distributions.values():
            # Filter by status
            if status and distribution.status != status:
                continue

            # Filter by model ID
            if model_id and distribution.model_id != model_id:
                continue

            results.append(distribution.to_dict())

        return results

    def get_hot_storage_weights(self, model_id: str, template_id: str  =  None) -> List[Dict[str, Any]]:
        """Get weights from hot storage"""
        results  =  []

        for file_name in os.listdir(self.hot_storage_path):
            if file_name.endswith('.json') and model_id in file_name:
                if template_id and template_id not in file_name:
                    continue

                try:
                    with open(os.path.join(self.hot_storage_path, file_name), 'r') as f:
                        data  =  json.load(f)
                        # Update access count
                        data["access_count"]  =  data.get("access_count", 0) + 1
                        data["last_accessed"]  =  time.time()

                        # Save updated access info
                        with open(os.path.join(self.hot_storage_path, file_name), 'w') as f2:
                            json.dump(data, f2, indent = 2)

                        results.append(data)
                except Exception as e:
                    print(f"Error reading hot storage file {file_name}: {e}")

        return results

    def sync_all_models(self, template_id: str) -> Dict[str, Any]:
        """Sync a template to all models"""
        # Find distribution with this template
        distribution  =  None
        for dist in self.distributions.values():
            if dist.template_id == template_id:
                distribution  =  dist
                break

        if not distribution:
            return {"success": False, "error": "Template not found"}

        # Create new distribution for sync
        sync_dist_id  =  self.create_distribution(
            template_id = distribution.template_id,
            model_id = distribution.model_id,
            weights_data = distribution.weights_data,
            metadata = {**distribution.metadata, "sync_operation": True},
            targets = list(self.model_locations.keys())
        )

        return {
            "success": True,
            "sync_distribution_id": sync_dist_id,
            "targets": list(self.model_locations.keys())
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get distribution statistics"""
        total_distributions  =  len(self.distributions)
        completed_distributions = sum(1 for d in self.distributions.values() if d.status == DistributionStatus.COMPLETED)
        failed_distributions = sum(1 for d in self.distributions.values() if d.status == DistributionStatus.FAILED)
        pending_distributions = sum(1 for d in self.distributions.values() if d.status == DistributionStatus.PENDING)

        # Count hot storage files
        hot_storage_files  =  len([f for f in os.listdir(self.hot_storage_path) if f.endswith('.json')])

        # Count database records
        db_records  =  len([f for f in os.listdir(self.database_path) if f.startswith('db_record_')])

        return {
            "total_distributions": total_distributions,
            "completed_distributions": completed_distributions,
            "failed_distributions": failed_distributions,
            "pending_distributions": pending_distributions,
            "success_rate": (completed_distributions / total_distributions * 100) if total_distributions > 0 else 0,
            "queue_length": len(self.distribution_queue),
            "currently_processing": self.processing,
            "hot_storage_files": hot_storage_files,
            "database_records": db_records,
            "model_locations": list(self.model_locations.keys())
        }

    def stop(self):
        """Stop the distributor"""
        self.running  =  False
        if self.distribution_thread.is_alive():
            self.distribution_thread.join(timeout = 1.0)

# Example usage
if __name__ == "__main__":
    # Create archiver distributor
    distributor  =  ArchiverDistributor()

    # Example weight distribution
    dist_id  =  distributor.create_distribution(
        template_id = "template_001",
        model_id = "llama2",
        weights_data = {
            "empathy": 0.95,
            "compassion": 0.92,
            "healing_focus": 0.93
        },
        metadata = {
            "source": "hope_memory",
            "purpose": "healing_and_service",
            "version": "1.0"
        }
    )

    print(f"Created distribution: {dist_id}")

    # Wait for processing
    time.sleep(5)

    # Get stats
    stats  =  distributor.get_stats()
    print(f"Distribution stats: {stats}")

    # Get hot storage weights
    hot_weights  =  distributor.get_hot_storage_weights("llama2")
    print(f"Hot storage weights: {len(hot_weights)}")

    # Stop distributor
    distributor.stop()