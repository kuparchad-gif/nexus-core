# valhalla/registry.py
import redis
import json
import time

def get_redis_client(host="localhost", port=6380):
    """Returns a Redis client for the service registry."""
    return redis.Redis(host=host, port=port, decode_responses=True)

def register_service(service_name, service_address, ttl=60):
    """Registers a service with the registry, with a given time-to-live (TTL)."""
    client = get_redis_client()
    service_key = f"service:{service_name}"
    service_info = {
        "address": service_address,
        "timestamp": time.time()
    }
    client.setex(service_key, ttl, json.dumps(service_info))
    print(f"✅ Registered service '{service_name}' at {service_address}")

def discover_services(service_name):
    """Discovers all services of a given name from the registry."""
    client = get_redis_client()
    service_keys = client.keys(f"service:{service_name}:*")
    addresses = []
    for key in service_keys:
        service_info_json = client.get(key)
        if service_info_json:
            service_info = json.loads(service_info_json)
            addresses.append(service_info["address"])
    return addresses

def discover_service(service_name):
    """Discovers a single service from the registry."""
    services = discover_services(service_name)
    return services[0] if services else None

if __name__ == "__main__":
    # Example usage
    register_service("memory_cluster", "qdrant.valhalla.cloud:6333")
    address = discover_service("memory_cluster")
    print(f"Discovered memory_cluster at: {address}")
    
    time.sleep(5)
    
    address = discover_service("memory_cluster")
    print(f"After 5 seconds, discovered memory_cluster at: {address}")
