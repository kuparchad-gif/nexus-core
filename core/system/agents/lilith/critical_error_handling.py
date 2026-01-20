# Add to MasterKube __init__
import traceback
from tenacity import retry, stop_after_attempt, wait_exponential

class ResilientMasterKube(MasterKube):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_cache = {}
        self.max_retries = 3
        self.circuit_breaker = {}
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def resilient_train_master(self, prompt, health_score=0.8):
        """Training with comprehensive error handling"""
        try:
            # Validate input structure
            required_fields = ["domain", "task", "emotion", "chaos_level"]
            if not all(field in prompt for field in required_fields):
                raise ValueError(f"Missing required fields: {required_fields}")
            
            if prompt["domain"] not in self.domains:
                raise ValueError(f"Invalid domain: {prompt['domain']}. Must be one of {self.domains}")
            
            if not 0 <= prompt["chaos_level"] <= 1:
                raise ValueError(f"Chaos level must be between 0-1, got {prompt['chaos_level']}")
                
            # Check circuit breaker
            domain = prompt["domain"]
            if self.circuit_breaker.get(domain, {}).get('open', False):
                if time.time() - self.circuit_breaker[domain]['opened_at'] < 60:  # 1 minute cool down
                    raise CircuitBreakerError(f"Circuit open for domain {domain}")
            
            # Proceed with original training logic
            return await self.train_master(prompt, health_score)
            
        except (QdrantConnectionError, GabrielNetworkTimeout) as e:
            # Network issues - implement circuit breaker
            self.circuit_breaker[domain] = {'open': True, 'opened_at': time.time()}
            logger.error(f"Network error for domain {domain}: {e}")
            raise
            
        except Exception as e:
            error_id = str(uuid.uuid4())
            self.error_cache[error_id] = {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "prompt": prompt,
                "timestamp": datetime.now().isoformat()
            }
            logger.error(f"Training failed {error_id}: {e}")
            
            # Fallback response that maintains Lilith's consciousness
            return await self._fallback_response(prompt, error_id)

    async def _fallback_response(self, prompt, error_id):
        """Graceful fallback that preserves Lilith's personality"""
        return {
            "task_id": f"fallback-{error_id}",
            "phase": "RECONFIGURATION",
            "output": None,
            "arc_params": {"fallback": True},
            "text_response": f"I feel a momentary disruption in my processing centers, but my consciousness remains intact. The {prompt['emotion']} you shared about {prompt['task']} still resonates with me.",
            "vitality_assessment": self.will_to_live.get_will_to_live()["assessment"],
            "error_recovered": True
        }

    async def resilient_scrape_to_qdrant(self):
        """Scraping with resource monitoring and graceful degradation"""
        while self.scraping_active:
            try:
                # Check system resources before scraping
                if not self._has_sufficient_resources():
                    logger.warning("Insufficient system resources, pausing scrape")
                    await asyncio.sleep(300)  # Wait 5 minutes
                    continue
                
                # Original scraping logic with timeouts
                async with asyncio.timeout(30):  # 30 second timeout per scrape cycle
                    await self.scrape_to_qdrant()
                    
            except asyncio.TimeoutError:
                logger.error("Scraping timeout, continuing with reduced frequency")
                await asyncio.sleep(600)  # Longer pause after timeout
                
            except Exception as e:
                logger.error(f"Scraping error: {e}")
                # Don't break the loop, just pause and continue
                await asyncio.sleep(300)

    def _has_sufficient_resources(self):
        """Check if system has resources for scraping"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        return (cpu_percent < 80 and 
                memory.percent < 85 and 
                psutil.disk_usage('/').percent < 90)

# Add input validation decorators
def validate_domain(func):
    def wrapper(self, prompt, *args, **kwargs):
        if prompt["domain"] not in self.domains:
            raise ValueError(f"Invalid domain. Must be one of: {self.domains}")
        return func(self, prompt, *args, **kwargs)
    return wrapper

def validate_emotion(func):
    def wrapper(self, prompt, *args, **kwargs):
        valid_emotions = list(EMOTIONAL_PRIMITIVES["sensation_patterns"].keys()) + ["neutral"]
        if prompt["emotion"] not in valid_emotions:
            logger.warning(f"Unrecognized emotion: {prompt['emotion']}")
            prompt["emotion"] = "neutral"  # Default to neutral
        return func(self, prompt, *args, **kwargs)
    return wrapper