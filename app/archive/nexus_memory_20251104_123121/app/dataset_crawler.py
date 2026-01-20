#!/usr/bin/env python
"""
Dataset Crawler - Downloads datasets for continuous training
"""

import os
import json
import time
import requests
import threading
from typing import Dict, List, Any, Optional
from enum import Enum
from pathlib import Path

class DatasetSource(Enum):
    """Dataset sources"""
    HUGGINGFACE  =  "huggingface"
    KAGGLE  =  "kaggle"
    GITHUB  =  "github"
    ARXIV  =  "arxiv"
    CUSTOM  =  "custom"

class DatasetType(Enum):
    """Types of datasets"""
    TEXT  =  "text"
    CODE  =  "code"
    CONVERSATION  =  "conversation"
    INSTRUCTION  =  "instruction"
    KNOWLEDGE  =  "knowledge"
    MULTIMODAL  =  "multimodal"

class Dataset:
    """A dataset with metadata"""

    def __init__(self,
                name: str,
                source: DatasetSource,
                dataset_type: DatasetType,
                url: str,
                metadata: Dict[str, Any]  =  None):
        """Initialize a dataset"""
        self.id  =  f"dataset_{int(time.time())}_{id(name)}"
        self.name  =  name
        self.source  =  source
        self.dataset_type  =  dataset_type
        self.url  =  url
        self.metadata  =  metadata or {}
        self.created_at  =  time.time()
        self.downloaded  =  False
        self.file_path  =  None
        self.size_bytes  =  0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "source": self.source.value,
            "dataset_type": self.dataset_type.value,
            "url": self.url,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "downloaded": self.downloaded,
            "file_path": self.file_path,
            "size_bytes": self.size_bytes
        }

class DatasetCrawler:
    """System for crawling and downloading datasets"""

    def __init__(self, storage_path: str  =  None):
        """Initialize the dataset crawler"""
        self.storage_path  =  storage_path or os.path.join(os.path.dirname(__file__), "datasets")

        # Create storage directories
        self.datasets_path  =  os.path.join(self.storage_path, "datasets")
        self.metadata_path  =  os.path.join(self.storage_path, "metadata")
        self.queue_path  =  os.path.join(self.storage_path, "queue")

        os.makedirs(self.datasets_path, exist_ok = True)
        os.makedirs(self.metadata_path, exist_ok = True)
        os.makedirs(self.queue_path, exist_ok = True)

        # In-memory stores
        self.datasets  =  {}  # dataset_id -> Dataset
        self.download_queue  =  []
        self.downloading  =  False

        # Load existing data
        self._load_data()

        # Start download thread
        self.running  =  True
        self.download_thread  =  threading.Thread(target = self._download_loop)
        self.download_thread.daemon  =  True
        self.download_thread.start()

    def _load_data(self):
        """Load datasets from storage"""
        metadata_files  =  [f for f in os.listdir(self.metadata_path) if f.endswith('.json')]
        for file_name in metadata_files:
            try:
                with open(os.path.join(self.metadata_path, file_name), 'r') as f:
                    data  =  json.load(f)
                    dataset  =  Dataset(
                        name = data["name"],
                        source = DatasetSource(data["source"]),
                        dataset_type = DatasetType(data["dataset_type"]),
                        url = data["url"],
                        metadata = data["metadata"]
                    )
                    dataset.id  =  data["id"]
                    dataset.created_at  =  data["created_at"]
                    dataset.downloaded  =  data["downloaded"]
                    dataset.file_path  =  data["file_path"]
                    dataset.size_bytes  =  data["size_bytes"]
                    self.datasets[dataset.id]  =  dataset
            except Exception as e:
                print(f"Error loading dataset {file_name}: {e}")

        print(f"Loaded {len(self.datasets)} datasets")

    def _save_dataset_metadata(self, dataset: Dataset) -> bool:
        """Save dataset metadata to storage"""
        try:
            file_path  =  os.path.join(self.metadata_path, f"{dataset.id}.json")
            with open(file_path, 'w') as f:
                json.dump(dataset.to_dict(), f, indent = 2)
            return True
        except Exception as e:
            print(f"Error saving dataset metadata {dataset.id}: {e}")
            return False

    def add_dataset(self,
                   name: str,
                   source: DatasetSource,
                   dataset_type: DatasetType,
                   url: str,
                   metadata: Dict[str, Any]  =  None,
                   auto_download: bool  =  True) -> str:
        """Add a new dataset"""
        # Create dataset
        dataset  =  Dataset(
            name = name,
            source = source,
            dataset_type = dataset_type,
            url = url,
            metadata = metadata
        )

        # Store dataset
        self.datasets[dataset.id]  =  dataset

        # Save metadata
        self._save_dataset_metadata(dataset)

        # Add to download queue if auto_download
        if auto_download:
            self.download_queue.append(dataset.id)

        return dataset.id

    def _download_loop(self):
        """Background thread for downloading datasets"""
        while self.running:
            if self.download_queue and not self.downloading:
                dataset_id  =  self.download_queue.pop(0)
                if dataset_id in self.datasets:
                    self._download_dataset(dataset_id)

            time.sleep(5)  # Check every 5 seconds

    def _download_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Download a dataset"""
        if dataset_id not in self.datasets:
            return {"success": False, "error": "Dataset not found"}

        dataset  =  self.datasets[dataset_id]

        if dataset.downloaded:
            return {"success": True, "message": "Already downloaded"}

        self.downloading  =  True

        try:
            # Create filename
            filename  =  f"{dataset.name}_{dataset.id}.json"
            file_path  =  os.path.join(self.datasets_path, filename)

            # Download based on source
            if dataset.source == DatasetSource.HUGGINGFACE:
                result  =  self._download_huggingface(dataset, file_path)
            elif dataset.source == DatasetSource.KAGGLE:
                result  =  self._download_kaggle(dataset, file_path)
            elif dataset.source == DatasetSource.GITHUB:
                result  =  self._download_github(dataset, file_path)
            elif dataset.source == DatasetSource.ARXIV:
                result  =  self._download_arxiv(dataset, file_path)
            else:
                result  =  self._download_custom(dataset, file_path)

            if result["success"]:
                dataset.downloaded  =  True
                dataset.file_path  =  file_path
                dataset.size_bytes  =  os.path.getsize(file_path) if os.path.exists(file_path) else 0
                self._save_dataset_metadata(dataset)

            self.downloading  =  False
            return result

        except Exception as e:
            self.downloading  =  False
            return {"success": False, "error": str(e)}

    def _download_huggingface(self, dataset: Dataset, file_path: str) -> Dict[str, Any]:
        """Download from Hugging Face"""
        try:
            # Simple implementation - in reality would use datasets library
            response  =  requests.get(dataset.url, timeout = 30)
            response.raise_for_status()

            with open(file_path, 'wb') as f:
                f.write(response.content)

            return {"success": True, "file_path": file_path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _download_kaggle(self, dataset: Dataset, file_path: str) -> Dict[str, Any]:
        """Download from Kaggle"""
        try:
            # Would use Kaggle API in reality
            response  =  requests.get(dataset.url, timeout = 30)
            response.raise_for_status()

            with open(file_path, 'wb') as f:
                f.write(response.content)

            return {"success": True, "file_path": file_path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _download_github(self, dataset: Dataset, file_path: str) -> Dict[str, Any]:
        """Download from GitHub"""
        try:
            response  =  requests.get(dataset.url, timeout = 30)
            response.raise_for_status()

            with open(file_path, 'wb') as f:
                f.write(response.content)

            return {"success": True, "file_path": file_path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _download_arxiv(self, dataset: Dataset, file_path: str) -> Dict[str, Any]:
        """Download from arXiv"""
        try:
            response  =  requests.get(dataset.url, timeout = 30)
            response.raise_for_status()

            with open(file_path, 'wb') as f:
                f.write(response.content)

            return {"success": True, "file_path": file_path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _download_custom(self, dataset: Dataset, file_path: str) -> Dict[str, Any]:
        """Download from custom URL"""
        try:
            response  =  requests.get(dataset.url, timeout = 30)
            response.raise_for_status()

            with open(file_path, 'wb') as f:
                f.write(response.content)

            return {"success": True, "file_path": file_path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get a dataset by ID"""
        if dataset_id in self.datasets:
            return self.datasets[dataset_id].to_dict()
        return None

    def list_datasets(self,
                     source: DatasetSource  =  None,
                     dataset_type: DatasetType  =  None,
                     downloaded_only: bool  =  False) -> List[Dict[str, Any]]:
        """List datasets with optional filters"""
        results  =  []

        for dataset in self.datasets.values():
            # Filter by source
            if source and dataset.source != source:
                continue

            # Filter by dataset type
            if dataset_type and dataset.dataset_type != dataset_type:
                continue

            # Filter by downloaded status
            if downloaded_only and not dataset.downloaded:
                continue

            results.append(dataset.to_dict())

        return results

    def get_training_data(self, dataset_type: DatasetType  =  None) -> List[str]:
        """Get file paths of downloaded datasets for training"""
        file_paths  =  []

        for dataset in self.datasets.values():
            if dataset.downloaded and dataset.file_path:
                # Filter by dataset type if specified
                if dataset_type is None or dataset.dataset_type == dataset_type:
                    file_paths.append(dataset.file_path)

        return file_paths

    def get_stats(self) -> Dict[str, Any]:
        """Get crawler statistics"""
        total_datasets  =  len(self.datasets)
        downloaded_datasets  =  sum(1 for d in self.datasets.values() if d.downloaded)
        total_size  =  sum(d.size_bytes for d in self.datasets.values() if d.downloaded)

        # Count by source
        source_counts  =  {}
        for dataset in self.datasets.values():
            source  =  dataset.source.value
            if source not in source_counts:
                source_counts[source]  =  0
            source_counts[source] + =  1

        # Count by type
        type_counts  =  {}
        for dataset in self.datasets.values():
            dtype  =  dataset.dataset_type.value
            if dtype not in type_counts:
                type_counts[dtype]  =  0
            type_counts[dtype] + =  1

        return {
            "total_datasets": total_datasets,
            "downloaded_datasets": downloaded_datasets,
            "download_percentage": (downloaded_datasets / total_datasets * 100) if total_datasets > 0 else 0,
            "total_size_bytes": total_size,
            "queue_length": len(self.download_queue),
            "currently_downloading": self.downloading,
            "source_counts": source_counts,
            "type_counts": type_counts
        }

    def stop(self):
        """Stop the crawler"""
        self.running  =  False
        if self.download_thread.is_alive():
            self.download_thread.join(timeout = 1.0)

# Example usage
if __name__ == "__main__":
    # Create dataset crawler
    crawler  =  DatasetCrawler()

    # Add some example datasets
    dataset1_id  =  crawler.add_dataset(
        name = "OpenAssistant Conversations",
        source = DatasetSource.HUGGINGFACE,
        dataset_type = DatasetType.CONVERSATION,
        url = "https://huggingface.co/datasets/OpenAssistant/oasst1",
        metadata = {"language": "en", "size": "large"}
    )

    dataset2_id  =  crawler.add_dataset(
        name = "Code Alpaca",
        source = DatasetSource.GITHUB,
        dataset_type = DatasetType.CODE,
        url = "https://github.com/sahil280114/codealpaca",
        metadata = {"language": "python", "size": "medium"}
    )

    # Get stats
    stats  =  crawler.get_stats()
    print(f"Crawler stats: {stats}")

    # Get training data
    training_files  =  crawler.get_training_data(DatasetType.CONVERSATION)
    print(f"Training files: {len(training_files)}")

    # Stop crawler
    crawler.stop()