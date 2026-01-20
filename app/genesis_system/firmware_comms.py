class UniversalModule:
    async def check_in_with_firmware(self):
        """Modules regularly communicate with firmware"""
        status = {
            "module_id": self.id,
            "current_role": self.current_role,
            "capacity_available": self.available_capacity,
            "can_tag_in_for": ["cpu_assist", "memory_support", "io_optimization"],
            "health_status": "optimal",
            "last_heartbeat": datetime.now().isoformat()
        }
        
        # Direct firmware communication
        await firmware_orchestrator.report_module_status(status)
    
    async def on_tag_in_request(self, emergency_role):
        """Firmware requests immediate role change"""
        print(f"🎯 MODULE {self.id} TAGGED IN for {emergency_role}")
        await self.transform_to(emergency_role, priority="emergency")
        return {"status": "tagged_in", "new_role": emergency_role}