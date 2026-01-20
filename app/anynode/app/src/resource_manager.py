import psutil
import asyncio
import requests
from typing import Dict, List, Optional
from master_db_manager import MasterDBManager
from vault_carrier import VaultCarrier
import json
from datetime import datetime

class ResourceManager:
    def __init__(self):
        self.db_manager = MasterDBManager()
        self.vault = VaultCarrier()
        self.neighbors = self._discover_neighbors()
        self.cpu_threshold = 80.0  # CPU usage threshold for load balancing
        self.memory_threshold = 85.0  # Memory usage threshold
        self.queue_threshold = 10  # Max queue size before delegation
        self.task_queue = []
        
    def _discover_neighbors(self) -> List[Dict]:
        """Discover neighbor nodes from Consul service registry"""
        try:
            consul_config = self.vault.retrieve_credentials('consul')
            neighbors = [
                {'id': 'aws-node', 'endpoint': 'https://aws-lambda-endpoint.com/process'},
                {'id': 'gcp-node', 'endpoint': 'https://gcp-function-endpoint.com/process'},
                {'id': 'azure-node', 'endpoint': 'https://azure-function-endpoint.com/process'}
            ]
            return neighbors
        except:
            return []
    
    def get_system_metrics(self) -> Dict:
        """Get current system resource usage"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'queue_size': len(self.task_queue),
            'timestamp': datetime.now().isoformat()
        }
    
    def is_overloaded(self) -> bool:
        """Check if current node is overloaded"""
        metrics = self.get_system_metrics()
        return (
            metrics['cpu_percent'] > self.cpu_threshold or
            metrics['memory_percent'] > self.memory_threshold or
            metrics['queue_size'] > self.queue_threshold
        )
    
    async def delegate_task(self, task: Dict) -> Dict:
        """Delegate task to least loaded neighbor"""
        if not self.is_overloaded():
            return await self._process_locally(task)
        
        # Find best neighbor
        best_neighbor = await self._find_best_neighbor()
        if not best_neighbor:
            # No neighbors available, process locally anyway
            return await self._process_locally(task)
        
        # Delegate to neighbor
        try:
            response = await self._send_to_neighbor(best_neighbor, task)
            self._log_delegation(task, best_neighbor, 'success')
            return response
        except Exception as e:
            self._log_delegation(task, best_neighbor, 'failed', str(e))
            # Fallback to local processing
            return await self._process_locally(task)
    
    async def _find_best_neighbor(self) -> Optional[Dict]:
        """Find neighbor with lowest resource usage"""
        neighbor_metrics = []
        
        for neighbor in self.neighbors:
            try:
                # Get neighbor's resource usage
                response = requests.get(f"{neighbor['endpoint']}/metrics", timeout=2)
                if response.status_code == 200:
                    metrics = response.json()
                    metrics['neighbor'] = neighbor
                    neighbor_metrics.append(metrics)
            except:
                continue
        
        if not neighbor_metrics:
            return None
        
        # Sort by combined load (CPU + Memory + Queue)
        neighbor_metrics.sort(key=lambda x: 
            x.get('cpu_percent', 100) + 
            x.get('memory_percent', 100) + 
            (x.get('queue_size', 0) * 5)  # Weight queue size more heavily
        )
        
        best = neighbor_metrics[0]
        # Only delegate if neighbor is significantly less loaded
        if (best.get('cpu_percent', 100) < self.cpu_threshold - 20 and
            best.get('memory_percent', 100) < self.memory_threshold - 15):
            return best['neighbor']
        
        return None
    
    async def _send_to_neighbor(self, neighbor: Dict, task: Dict) -> Dict:
        """Send task to neighbor node"""
        payload = {
            'task': task,
            'delegated_from': 'local-node',
            'timestamp': datetime.now().isoformat()
        }
        
        response = requests.post(
            neighbor['endpoint'],
            json=payload,
            timeout=30,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Neighbor returned {response.status_code}")
    
    async def _process_locally(self, task: Dict) -> Dict:
        """Process task on local node"""
        task_type = task.get('type', 'unknown')
        
        if task_type == 'design_ui':
            from design_cluster import DesignCluster
            dc = DesignCluster()
            result = dc.generate_ui(task['prompt'], task.get('page_type', 'dashboard'))
            
        elif task_type == 'design_asset':
            from design_cluster import DesignCluster
            dc = DesignCluster()
            result = dc.generate_game_asset(task['prompt'], task.get('asset_type', 'sprite'))
            
        elif task_type == 'ecommerce':
            from financial_utils import FinancialUtils
            fu = FinancialUtils()
            result = fu.auto_list_products(task['store_name'], task.get('num_products', 3))
            
        elif task_type == 'social_media':
            from social_media_manager import SocialMediaManager
            smm = SocialMediaManager()
            result = smm.auto_post_designs(task['store_name'])
            
        else:
            result = f"Processed {task_type} locally"
        
        # Store processing record
        self.db_manager.write_record('resource_usage', {
            'task_type': task_type,
            'processed_locally': True,
            'metrics': self.get_system_metrics(),
            'result_size': len(str(result))
        })
        
        return {'result': result, 'processed_by': 'local-node'}
    
    def _log_delegation(self, task: Dict, neighbor: Dict, status: str, error: str = None):
        """Log task delegation for monitoring"""
        log_entry = {
            'task_type': task.get('type'),
            'delegated_to': neighbor['id'],
            'status': status,
            'timestamp': datetime.now().isoformat()
        }
        
        if error:
            log_entry['error'] = error
        
        self.db_manager.write_record('delegations', log_entry)
    
    async def health_check_neighbors(self) -> Dict:
        """Check health of all neighbor nodes"""
        neighbor_health = {}
        
        for neighbor in self.neighbors:
            try:
                response = requests.get(f"{neighbor['endpoint']}/health", timeout=5)
                neighbor_health[neighbor['id']] = {
                    'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                    'response_time': response.elapsed.total_seconds(),
                    'last_check': datetime.now().isoformat()
                }
            except Exception as e:
                neighbor_health[neighbor['id']] = {
                    'status': 'unreachable',
                    'error': str(e),
                    'last_check': datetime.now().isoformat()
                }
        
        return neighbor_health
    
    def get_cluster_stats(self) -> Dict:
        """Get overall cluster statistics"""
        local_metrics = self.get_system_metrics()
        
        return {
            'local_node': {
                'metrics': local_metrics,
                'overloaded': self.is_overloaded(),
                'queue_size': len(self.task_queue)
            },
            'neighbors': len(self.neighbors),
            'delegation_enabled': len(self.neighbors) > 0,
            'load_balancing': 'active' if self.is_overloaded() else 'standby'
        }