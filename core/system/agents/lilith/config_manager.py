# Add to your imports
import os
from typing import Optional

class ConfigurationManager:
    """Manage Lilith's configuration with validation and fallbacks"""
    
    def __init__(self):
        self.config = self._load_config()
        
    def _load_config(self):
        """Load configuration with environment variable overrides"""
        base_config = {
            "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
            "gabriel_ws_url": os.getenv("GABRIEL_WS_URL", "ws://localhost:8765"),
            "max_scraping_duration": int(os.getenv("MAX_SCRAPING_HOURS", "10")),
            "min_system_resources": {
                "cpu_threshold": float(os.getenv("CPU_THRESHOLD", "80.0")),
                "memory_threshold": float(os.getenv("MEMORY_THRESHOLD", "85.0"))
            },
            "retry_config": {
                "max_attempts": int(os.getenv("MAX_RETRIES", "3")),
                "base_delay": float(os.getenv("BASE_DELAY", "1.0"))
            }
        }
        
        # Validate critical configurations
        assert base_config["qdrant_url"], "QDRANT_URL must be set"
        assert base_config["gabriel_ws_url"], "GABRIEL_WS_URL must be set"
        
        return base_config

# Update MasterKube initialization
class ProductionMasterKube(ResilientMasterKube):
    def __init__(self, config: Optional[ConfigurationManager] = None):
        self.config = config or ConfigurationManager()
        super().__init__(
            qdrant_url=self.config.config["qdrant_url"],
            ws_url=self.config.config["gabriel_ws_url"],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )