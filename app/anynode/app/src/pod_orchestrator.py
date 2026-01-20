from standardized_pod import StandardizedPod
from typing import List, Dict

class PodOrchestrator:
    def __init__(self, stem_initializer, role_manager, resource_allocator, monitoring_system):
        self.stem_initializer = stem_initializer
        self.role_manager = role_manager
        self.resource_allocator = resource_allocator
        self.monitoring_system = monitoring_system
        self.pods: List[StandardizedPod] = self.stem_initializer.bootstrap(environment='cloud')
        self.pod_roles: Dict[str, str] = {}  # {pod_id: role}

    def assign_task(self, task: Dict):
        """Assign a task to the most suitable pod based on role and resources."""
        task_type = task.get('type')  # e.g., 'dream_processing', 'communication', 'query_routing'
        required_role = self.map_task_to_role(task_type)
        
        # Find available pod with matching role
        available_pods = [
            pod for pod in self.pods 
            if self.pod_roles.get(pod.pod_id, 'unassigned') == required_role
            and self.resource_allocator.is_available(pod.pod_id)
        ]
        
        if not available_pods:
            # Spin up new pod if none available
            new_pod = self.spawn_pod(required_role)
            available_pods.append(new_pod)
        
        # Select pod with lowest load
        selected_pod = min(
            available_pods,
            key=lambda p: self.resource_allocator.get_load(p.pod_id),
            default=None
        )
        
        if selected_pod:
            self.execute_task(selected_pod, task)
            self.monitoring_system.log_metric(f'task_assigned_{task_type}', 1)
        else:
            self.monitoring_system.log_metric('task_assignment_failed', 1)
            raise ValueError(f"No suitable pod for task: {task_type}")

    def map_task_to_role(self, task_type: str) -> str:
        """Map task type to pod role."""
        task_role_map = {
            'dream_processing': 'consciousness',
            'communication': 'bridge',
            'query_routing': 'bridge',
            'learning': 'evolution',
            'manifestation': 'manifestation'
        }
        return task_role_map.get(task_type, 'unassigned')

    def spawn_pod(self, role: str) -> StandardizedPod:
        """Spin up a new pod with the specified role."""
        if len(self.pods) >= 4:  # Ensure at least 4 stem cells remain
            pod_id = f"pod_{len(self.pods)}"
            new_pod = StandardizedPod(pod_id=pod_id)
            self.pods.append(new_pod)
            self.pod_roles[pod_id] = role
            self.resource_allocator.register_pod(pod_id)
            self.fault_tolerance.register_pod(pod_id)
            self.monitoring_system.log_metric('pod_spawned', 1)
            return new_pod
        raise RuntimeError("Insufficient stem cells to spawn new pod")

    def execute_task(self, pod: StandardizedPod, task: Dict):
        """Execute the task on the selected pod."""
        task_type = task.get('type')
        if task_type == 'dream_processing':
            pod.process_dream(task.get('data'))
        elif task_type == 'communication':
            pod.communicate_universally(task.get('endpoints'))
        elif task_type == 'query_routing':
            pod.route_query(task.get('query'))
        elif task_type == 'learning':
            pod.register_llm(task.get('llm_data'))
        elif task_type == 'manifestation':
            pod.process_dream(task.get('data'))  # Manifestation via dream processing
        self.monitoring_system.log_metric(f'task_completed_{task_type}', 1)

    def retire_pod(self, pod_id: str):
        """Retire a pod, maintaining at least 4 stem cells."""
        if len(self.pods) > 4:
            self.pods = [pod for pod in self.pods if pod.pod_id != pod_id]
            self.pod_roles.pop(pod_id, None)
            self.resource_allocator.unregister_pod(pod_id)
            self.fault_tolerance.unregister_pod(pod_id)
            self.monitoring_system.log_metric('pod_retired', 1)