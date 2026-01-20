class PollingOrchestrator:
    def __init__(self):
        self.polling_cycles = {
            "fast": 3,    # Quick health checks
            "medium": 6,  # Performance metrics  
            "deep": 9,    # Role optimization
            "full": 13    # Complete system analysis
        }
    
    async def start_staggered_polling(self):
        # Modules poll at different intervals
        asyncio.create_task(self._poll_cycle("fast", self._fast_health_check))
        asyncio.create_task(self._poll_cycle("medium", self._performance_check))
        asyncio.create_task(self._poll_cycle("deep", self._role_optimization))
        asyncio.create_task(self._poll_cycle("full", self._system_analysis))