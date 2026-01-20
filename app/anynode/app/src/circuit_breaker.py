# Path: nexus_platform/common/circuit_breaker.py
import pybreaker
from typing import Callable
from functools import wraps

class CircuitBreaker:
    def __init__(self, service_name: str):
        self.breaker = pybreaker.CircuitBreaker(
            fail_max=5,
            reset_timeout=60,
            name=f"{service_name}_breaker"
        )
        self.logger = setup_logger(f"{service_name}.breaker")

    def protect(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await self.breaker.call_async(func, *args, **kwargs)
            except pybreaker.CircuitBreakerError as e:
                self.logger.error({"action": "circuit_breaker_triggered", "error": str(e)})
                raise
        return wrapper