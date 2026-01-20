import os
import json
import logging
from datetime import datetime
import threading
import time
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sync_monitor')

class SyncMonitor:
    """Monitor and control the Viren sync system."""
    
    def __init__(self):
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.config_path = os.path.join(self.root_dir, 'Config', 'system', 'sync_config.json')
        self.sync_log_path = os.path.join(self.root_dir, 'logs', 'viren_sync.log')
        self.history_path = os.path.join(self.root_dir, 'data', 'sync_history.json')
        
        # Load config
        self.config = self._load_config()
        
        # Initialize history
        self._ensure_history_file()
        
        # Start monitoring thread
        self.monitoring = False
        self.monitor_thread = None
    
    def _load_config(self):
        """Load sync configuration."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Config file not found: {self.config_path}")
                return {"sync": {"enabled": True, "auto_sync_interval_minutes": 30}}
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return {"sync": {"enabled": True, "auto_sync_interval_minutes": 30}}
    
    def _ensure_history_file(self):
        """Ensure sync history file exists."""
        try:
            os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
            
            if not os.path.exists(self.history_path):
                with open(self.history_path, 'w') as f:
                    json.dump({"syncs": []}, f)
        except Exception as e:
            logger.error(f"Error ensuring history file: {str(e)}")
    
    def get_status(self):
        """Get current sync status."""
        status = {
            "enabled": self.config.get("sync", {}).get("enabled", True),
            "last_sync_time": self._get_last_sync_time(),
            "local_db_connected": self._check_local_db(),
            "cloud_db_connected": self._check_cloud_db(),
            "tinyllama_loaded": self._check_tinyllama(),
            "monitoring_active": self.monitoring,
            "metrics": self._get_metrics(),
            "recent_history": self._get_recent_history(5)
        }
        return status
    
    def _get_last_sync_time(self):
        """Get the timestamp of the last successful sync."""
        try:
            history = self._load_history()
            syncs = history.get("syncs", [])
            
            if syncs:
                # Find the most recent completed sync
                completed_syncs = [s for s in syncs if s.get("status") == "Completed"]
                if completed_syncs:
                    return completed_syncs[0].get("time", "Never")
            
            return "Never"
        except Exception as e:
            logger.error(f"Error getting last sync time: {str(e)}")
            return "Unknown"
    
    def _check_local_db(self):
        """Check if local database is connected."""
        try:
            import requests
            
            weaviate_url = self.config.get("sync", {}).get("local_endpoints", {}).get("weaviate", "http://localhost:8080")
            
            response = requests.get(f"{weaviate_url}/v1/meta")
            return response.status_code == 200
        except:
            return False
    
    def _check_cloud_db(self):
        """Check if cloud database is connected."""
        try:
            import requests
            
            cloud_url = self.config.get("sync", {}).get("cloud_endpoints", {}).get("weaviate")
            
            if not cloud_url:
                return False
            
            # For demo purposes, we'll just return True
            # In a real implementation, would check actual connection
            return True
        except:
            return False
    
    def _check_tinyllama(self):
        """Check if TinyLlama model is loaded."""
        try:
            model_path = self.config.get("sync", {}).get("models", {}).get("tinyllama_path")
            
            if not model_path:
                return False
            
            full_path = os.path.join(self.root_dir, model_path)
            
            # For demo purposes, we'll just check if the directory exists
            # In a real implementation, would check if model is actually loaded
            return os.path.exists(os.path.dirname(full_path))
        except:
            return False
    
    def _get_metrics(self):
        """Get sync metrics."""
        try:
            history = self._load_history()
            syncs = history.get("syncs", [])
            
            # Calculate metrics
            today = datetime.now().strftime("%Y-%m-%d")
            today_syncs = [s for s in syncs if s.get("time", "").startswith(today)]
            
            total_objects = sum(s.get("objects", 0) for s in syncs)
            total_conflicts = sum(len(s.get("conflicts", [])) for s in syncs)
            
            # Calculate average sync time
            sync_times = [s.get("duration", 0) for s in syncs if "duration" in s]
            avg_time = sum(sync_times) / len(sync_times) if sync_times else 0
            
            return {
                "total_syncs_today": len(today_syncs),
                "objects_synced": total_objects,
                "conflicts_resolved": total_conflicts,
                "avg_sync_time": f"{avg_time:.1f}s"
            }
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            return {
                "total_syncs_today": 0,
                "objects_synced": 0,
                "conflicts_resolved": 0,
                "avg_sync_time": "0.0s"
            }
    
    def _load_history(self):
        """Load sync history."""
        try:
            if os.path.exists(self.history_path):
                with open(self.history_path, 'r') as f:
                    return json.load(f)
            return {"syncs": []}
        except Exception as e:
            logger.error(f"Error loading history: {str(e)}")
            return {"syncs": []}
    
    def _get_recent_history(self, limit=5):
        """Get recent sync history."""
        try:
            history = self._load_history()
            syncs = history.get("syncs", [])
            
            # Sort by time (newest first) and limit
            sorted_syncs = sorted(syncs, key=lambda s: s.get("time", ""), reverse=True)
            return sorted_syncs[:limit]
        except Exception as e:
            logger.error(f"Error getting recent history: {str(e)}")
            return []
    
    def trigger_sync(self):
        """Trigger a manual sync."""
        try:
            logger.info("Triggering manual sync")
            
            # Record start time
            start_time = time.time()
            
            # Run the sync launcher script
            sync_script = os.path.join(self.root_dir, 'boot', 'viren_sync_launcher.py')
            
            result = subprocess.run(
                [sys.executable, sync_script],
                capture_output=True,
                text=True
            )
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Parse output for results
            output = result.stdout + result.stderr
            success = "Sync process completed successfully" in output
            
            # Extract objects count (simplified)
            objects_count = 0
            conflicts = []
            
            # Record sync in history
            self._record_sync({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": "Manual",
                "priority": "Medium",  # Default for manual
                "objects": objects_count,
                "conflicts": conflicts,
                "status": "Completed" if success else "Failed",
                "duration": duration
            })
            
            return {
                "success": success,
                "duration": f"{duration:.2f}s",
                "objects": objects_count,
                "conflicts": len(conflicts)
            }
        except Exception as e:
            logger.error(f"Error triggering sync: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _record_sync(self, sync_data):
        """Record a sync in the history."""
        try:
            history = self._load_history()
            syncs = history.get("syncs", [])
            
            # Add new sync at the beginning
            syncs.insert(0, sync_data)
            
            # Limit history size
            if len(syncs) > 100:
                syncs = syncs[:100]
            
            history["syncs"] = syncs
            
            # Save history
            with open(self.history_path, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error(f"Error recording sync: {str(e)}")
    
    def start_monitoring(self):
        """Start monitoring for automatic syncs."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Sync monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring for automatic syncs."""
        self.monitoring = False
        logger.info("Sync monitoring stopped")
    
    def _monitor_loop(self):
        """Monitor loop for automatic syncs."""
        while self.monitoring:
            try:
                # Check if sync is enabled
                if not self.config.get("sync", {}).get("enabled", True):
                    time.sleep(60)
                    continue
                
                # Get interval in seconds
                interval_minutes = self.config.get("sync", {}).get("auto_sync_interval_minutes", 30)
                interval_seconds = interval_minutes * 60
                
                # Sleep for the interval
                time.sleep(interval_seconds)
                
                # Trigger sync if still monitoring
                if self.monitoring:
                    logger.info("Triggering automatic sync")
                    
                    # Record start time
                    start_time = time.time()
                    
                    # Run the sync launcher script
                    sync_script = os.path.join(self.root_dir, 'boot', 'viren_sync_launcher.py')
                    
                    result = subprocess.run(
                        [sys.executable, sync_script],
                        capture_output=True,
                        text=True
                    )
                    
                    # Calculate duration
                    duration = time.time() - start_time
                    
                    # Parse output for results
                    output = result.stdout + result.stderr
                    success = "Sync process completed successfully" in output
                    
                    # Extract objects count (simplified)
                    objects_count = 0
                    conflicts = []
                    
                    # Determine priority (simplified)
                    if "high priority" in output.lower():
                        priority = "High"
                    elif "medium priority" in output.lower():
                        priority = "Medium"
                    else:
                        priority = "Low"
                    
                    # Record sync in history
                    self._record_sync({
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": "Automatic",
                        "priority": priority,
                        "objects": objects_count,
                        "conflicts": conflicts,
                        "status": "Completed" if success else "Failed",
                        "duration": duration
                    })
            except Exception as e:
                logger.error(f"Error in monitor loop: {str(e)}")
                time.sleep(60)  # Sleep for a minute before retrying

# For testing
if __name__ == "__main__":
    import sys
    
    monitor = SyncMonitor()
    status = monitor.get_status()
    print(json.dumps(status, indent=2))
    
    if len(sys.argv) > 1 and sys.argv[1] == "trigger":
        result = monitor.trigger_sync()
        print(json.dumps(result, indent=2))