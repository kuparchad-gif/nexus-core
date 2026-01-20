# Base Layer: Processing with BERT/TinyLlama for resource management
import os
import logging
from typing import List, Dict, Any
from transformers import pipeline, AutoModelForMaskedLM, AutoTokenizer, AutoModelForCausalLM

class BaseLayer:
    def __init__(self, model_choice: str  =  "bert"):
        """
        Initialize BaseLayer with specified model for resource management.

        Args:
            model_choice: Either "bert" or "tinyllama" for different optimization strategies
        """
        self.resource_pool: List[Dict[str, Any]]  =  []  # List of available PCs/nodes with metadata
        self.model_choice  =  model_choice
        self.logger  =  self._setup_logger()

        # Initialize the selected model
        try:
            if model_choice.lower() == "bert":
                self.model  =  pipeline(
                    'fill-mask',
                    model = 'bert-base-uncased',
                    tokenizer = 'bert-base-uncased'
                )
                self.logger.info("BERT model loaded successfully for resource optimization")
            elif model_choice.lower() == "tinyllama":
                self.model  =  pipeline(
                    'text-generation',
                    model = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',  # Updated model path
                    tokenizer = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                    torch_dtype = 'auto',
                    device_map = 'auto'
                )
                self.logger.info("TinyLlama model loaded successfully for resource optimization")
            else:
                raise ValueError(f"Unsupported model choice: {model_choice}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.model  =  None

    def _setup_logger(self):
        """Setup logging configuration"""
        logger  =  logging.getLogger('BaseLayer')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler  =  logging.StreamHandler()
            formatter  =  logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def add_resources(self, resources: List[Dict[str, Any]]):
        """
        Add computational resources to the pool.

        Args:
            resources: List of dictionaries containing resource information
                      Example: [{'id': 'pc1', 'cpu_cores': 8, 'memory_gb': 16, 'gpu': True}]
        """
        self.resource_pool.extend(resources)
        self.logger.info(f"Added {len(resources)} resources to pool. Total: {len(self.resource_pool)}")

    def consume_resources(self, num_pcs: int  =  1, task_description: str  =  None):
        """
        Consume PCs for compute tasks with intelligent allocation.

        Args:
            num_pcs: Number of PCs to allocate
            task_description: Optional description for task-aware allocation
        """
        if len(self.resource_pool) < num_pcs:
            self.logger.warning(f"Insufficient resources. Requested: {num_pcs}, Available: {len(self.resource_pool)}")
            return False

        # Select best resources based on optimization
        selected_resources  =  self._select_optimal_resources(num_pcs, task_description)

        self.logger.info(f'Consuming {num_pcs} PCs for processing task: {task_description}')
        self.logger.info(f'Selected resources: {[r["id"] for r in selected_resources]}')

        # Remove allocated resources from pool
        for resource in selected_resources:
            self.resource_pool.remove(resource)

        return selected_resources

    def _select_optimal_resources(self, num_pcs: int, task_description: str  =  None) -> List[Dict[str, Any]]:
        """
        Select optimal resources based on task requirements and available resources.
        """
        if task_description and self.model:
            # Use model to analyze task requirements and suggest optimal allocation
            optimization_suggestion  =  self._analyze_task_requirements(task_description)
            self.logger.info(f"Task analysis: {optimization_suggestion}")

        # Simple heuristic: prioritize resources with GPU, then more CPU cores and memory
        sorted_resources  =  sorted(
            self.resource_pool,
            key = lambda x: (
                x.get('gpu', False),
                x.get('cpu_cores', 0),
                x.get('memory_gb', 0)
            ),
            reverse = True
        )

        return sorted_resources[:num_pcs]

    def _analyze_task_requirements(self, task_description: str) -> str:
        """
        Use the loaded model to analyze task requirements and suggest resource allocation.
        """
        try:
            if self.model_choice == "bert":
                # BERT-style analysis for resource requirements
                analysis_prompt  =  f"The computational task '{task_description}' requires"
                # This is a simplified example - in practice you'd use more sophisticated prompting
                return "BERT analysis: Moderate compute requirements suggested."

            elif self.model_choice == "tinyllama":
                # TinyLlama for more complex reasoning about resource needs
                prompt  =  f"""
                Analyze this computational task and suggest optimal resource allocation:
                Task: {task_description}

                Considerations:
                - CPU-intensive vs GPU-intensive
                - Memory requirements
                - Expected runtime
                - Parallelization potential

                Recommendation:
                """
                # In a real implementation, you would call the model here
                return "TinyLlama analysis: GPU acceleration recommended for this task."

        except Exception as e:
            self.logger.error(f"Task analysis failed: {e}")
            return "Analysis unavailable"

    def optimize_resources(self) -> Dict[str, Any]:
        """
        Use models for optimization (predict load, suggest rebalancing).

        Returns:
            Dictionary containing optimization suggestions
        """
        optimization_report  =  {
            "total_resources": len(self.resource_pool),
            "gpu_resources": len([r for r in self.resource_pool if r.get('gpu', False)]),
            "suggestions": []
        }

        # Generate optimization suggestions based on current pool
        if len(self.resource_pool) > 10:
            optimization_report["suggestions"].append("Consider load balancing across clusters")

        if len([r for r in self.resource_pool if r.get('gpu', False)]) < 2:
            optimization_report["suggestions"].append("Add more GPU resources for parallel tasks")

        self.logger.info(f"Optimization report: {optimization_report}")
        return optimization_report

    def release_resources(self, resources: List[Dict[str, Any]]):
        """
        Release resources back to the pool.

        Args:
            resources: List of resources to return to the pool
        """
        self.resource_pool.extend(resources)
        self.logger.info(f"Released {len(resources)} resources back to pool")

    def get_pool_status(self) -> Dict[str, Any]:
        """Get current status of resource pool"""
        return {
            "total_resources": len(self.resource_pool),
            "resource_details": self.resource_pool,
            "gpu_count": len([r for r in self.resource_pool if r.get('gpu', False)]),
            "total_cpu_cores": sum(r.get('cpu_cores', 0) for r in self.resource_pool),
            "total_memory_gb": sum(r.get('memory_gb', 0) for r in self.resource_pool)
        }

if __name__ == '__main__':
    # Example usage
    base  =  BaseLayer(model_choice = "bert")

    # Add some sample resources
    sample_resources  =  [
        {'id': 'pc1', 'cpu_cores': 8, 'memory_gb': 16, 'gpu': True},
        {'id': 'pc2', 'cpu_cores': 4, 'memory_gb': 8, 'gpu': False},
        {'id': 'pc3', 'cpu_cores': 12, 'memory_gb': 32, 'gpu': True},
    ]
    base.add_resources(sample_resources)

    # Consume resources for a task
    allocated  =  base.consume_resources(2, "Neural network training batch")

    # Check optimization
    base.optimize_resources()

    # Check pool status
    print("Pool status:", base.get_pool_status())