import ray, os

def init():
    # Local Ray in-pod; expand to external later
    ray.init(ignore_reinit_error=True, include_dashboard=False, _temp_dir="/tmp/ray")
    return ray

