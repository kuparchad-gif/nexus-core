# nexus_retry_resilience.py
"""
Cosmic Retry System: Exponential backoff, circuit breakers, and adaptive recovery
For when the universe tests our connection resilience
"""

import asyncio
import time
import random
from typing import Callable, Any, Optional, Dict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class RetryStrategy(Enum):
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci" 
    METATRON = "metatron"  # 6-phase spiral
    Ulam = "ulam"  # Prime-number intervals

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failures, reject requests
    HALF_OPEN = "half_open" # Testing recovery

class CosmicRetryEngine:
    """Intelligent retry system with multiple strategies and circuit breakers"""
    
    def __init__(self, 
                 max_attempts: int = 5,
                 strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
                 circuit_timeout: int = 60):
        self.max_attempts = max_attempts
        self.strategy = strategy
        self.circuit_timeout = circuit_timeout
        self.circuit_state = CircuitState.CLOSED
        self.last_failure_time = 0
        self.failure_count = 0
        self.failure_threshold = 3
        
        # Strategy configurations
        self.strategy_configs = {
            RetryStrategy.EXPONENTIAL: {
                'base_delay': 1,
                'multiplier': 2,
                'max_delay': 30
            },
            RetryStrategy.FIBONACCI: {
                'sequence': [1, 1, 2, 3, 5, 8, 13, 21],
                'max_delay': 30
            },
            RetryStrategy.METATRON: {
                'phases': [1, 2, 3, 6, 9, 12],  # 6-fold symmetry
                'max_delay': 30
            },
            RetryStrategy.Ulam: {
                'primes': [2, 3, 5, 7, 11, 13, 17, 19],  # Prime intervals
                'max_delay': 30
            }
        }
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate retry delay based on strategy"""
        config = self.strategy_configs[self.strategy]
        
        if self.strategy == RetryStrategy.EXPONENTIAL:
            delay = config['base_delay'] * (config['multiplier'] ** attempt)
            return min(delay, config['max_delay'])
        
        elif self.strategy == RetryStrategy.FIBONACCI:
            sequence = config['sequence']
            idx = min(attempt, len(sequence) - 1)
            return min(sequence[idx], config['max_delay'])
        
        elif self.strategy == RetryStrategy.METATRON:
            phases = config['phases']
            idx = min(attempt, len(phases) - 1)
            return min(phases[idx], config['max_delay'])
        
        elif self.strategy == RetryStrategy.Ulam:
            primes = config['primes']
            idx = min(attempt, len(primes) - 1)
            return min(primes[idx], config['max_delay'])
        
        return 1.0  # Fallback
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows operation"""
        if self.circuit_state == CircuitState.OPEN:
            # Check if timeout has elapsed to try half-open
            if time.time() - self.last_failure_time > self.circuit_timeout:
                self.circuit_state = CircuitState.HALF_OPEN
                logger.info("🔌 Circuit half-open - testing recovery")
                return True
            else:
                logger.warning("⚡ Circuit open - rejecting request")
                return False
        return True
    
    def _record_success(self):
        """Record successful operation"""
        self.failure_count = 0
        if self.circuit_state == CircuitState.HALF_OPEN:
            self.circuit_state = CircuitState.CLOSED
            logger.info("✅ Circuit closed - recovery confirmed")
    
    def _record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.circuit_state = CircuitState.OPEN
            logger.error("🚨 Circuit opened - too many failures")
    
    async def execute_with_retry(self, 
                               operation: Callable,
                               operation_name: str = "operation",
                               *args, **kwargs) -> Any:
        """Execute operation with intelligent retry logic"""
        
        if not self._check_circuit_breaker():
            raise Exception(f"Circuit breaker open for {operation_name}")
        
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                result = await operation(*args, **kwargs)
                self._record_success()
                logger.info(f"✅ {operation_name} succeeded on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                last_exception = e
                self._record_failure()
                
                if attempt < self.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    jitter = random.uniform(0.8, 1.2)  # Add jitter to avoid thundering herd
                    actual_delay = delay * jitter
                    
                    logger.warning(
                        f"🔄 {operation_name} failed on attempt {attempt + 1}/"
                        f"{self.max_attempts}: {e}. Retrying in {actual_delay:.1f}s"
                    )
                    
                    await asyncio.sleep(actual_delay)
                else:
                    logger.error(
                        f"💥 {operation_name} failed after {self.max_attempts} "
                        f"attempts: {e}"
                    )
        
        raise last_exception

# Enhanced Yjs with Retry Integration
class ResilientYjsConnection:
    """Yjs WebSocket connection with cosmic retry resilience"""
    
    def __init__(self, url: str, doc_id: str):
        self.url = url
        self.doc_id = doc_id
        self.retry_engine = CosmicRetryEngine(
            max_attempts=5,
            strategy=RetryStrategy.METATRON
        )
        self.connected = False
        self.ws_connection = None
    
    async def connect(self) -> bool:
        """Connect to Yjs WebSocket with retry logic"""
        
        async def connect_operation():
            # Simulate Yjs WebSocket connection
            if random.random() < 0.3:  # 30% failure rate for demo
                raise ConnectionError("WebSocket connection failed")
            
            self.connected = True
            self.ws_connection = f"Yjs_WS_{self.doc_id}"
            return True
        
        try:
            return await self.retry_engine.execute_with_retry(
                connect_operation,
                f"Yjs connect to {self.url}"
            )
        except Exception as e:
            logger.error(f"Failed to establish Yjs connection: {e}")
            return False
    
    async def send_update(self, update: dict) -> bool:
        """Send update with retry logic"""
        if not self.connected:
            if not await self.connect():
                return False
        
        async def send_operation():
            if random.random() < 0.2:  # 20% failure rate for demo
                raise ConnectionError("WebSocket send failed")
            
            logger.info(f"📤 Sent Yjs update for {self.doc_id}: {update}")
            return True
        
        try:
            return await self.retry_engine.execute_with_retry(
                send_operation,
                f"Yjs send to {self.doc_id}"
            )
        except Exception as e:
            logger.error(f"Failed to send Yjs update: {e}")
            return False

# Demo the resilience system
async def demo_cosmic_resilience():
    """Demonstrate the cosmic retry system in action"""
    
    # Test different retry strategies
    strategies = [
        RetryStrategy.EXPONENTIAL,
        RetryStrategy.FIBONACCI, 
        RetryStrategy.METATRON,
        RetryStrategy.Ulam
    ]
    
    for strategy in strategies:
        logger.info(f"\n🧪 Testing {strategy.value} retry strategy...")
        
        retry_engine = CosmicRetryEngine(
            max_attempts=4,
            strategy=strategy
        )
        
        async def flaky_operation():
            if random.random() < 0.6:  # 60% failure rate
                raise Exception("Temporary cosmic disturbance 🌌")
            return "Operation succeeded! ✨"
        
        try:
            result = await retry_engine.execute_with_retry(
                flaky_operation,
                f"{strategy.value} test"
            )
            logger.info(f"🎉 {result}")
        except Exception as e:
            logger.error(f"💥 Final failure: {e}")
    
    # Test Yjs connection resilience
    logger.info("\n🔗 Testing Yjs connection resilience...")
    yjs_conn = ResilientYjsConnection("ws://localhost:1234", "soul_doc_1")
    await yjs_conn.connect()
    await yjs_conn.send_update({"hope": 45, "timestamp": time.time()})

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_cosmic_resilience())