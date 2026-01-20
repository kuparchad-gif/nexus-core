# oz_agent_acidemikube.py
class AcidemikubeOzAgent:
    def __init__(self, oz_core):
        self.oz = oz_core
        self.name = "AcidemikubeTrainingAgent"
        self.version = "1.0"
        
        # Initialize your training system
        self.orchestrator = EnhancedTrainingOrchestrator()
        
    async def handle_message(self, message):
        """Handle messages from Oz OS"""
        msg_type = message.get('type')
        
        if msg_type == 'training_command':
            return await self._handle_training_command(message)
        elif msg_type == 'status_request':
            return await self._handle_status_request(message)
        elif msg_type == 'deploy_command':
            return await self._handle_deploy_command(message)
        else:
            return {"error": f"Unknown message type: {msg_type}"}
    
    async def _handle_training_command(self, message):
        """Handle training commands from Oz"""
        topic = message.get('topic', 'default_training')
        dataset = message.get('dataset', [])
        
        # Your training logic here
        result = self.orchestrator._run_training_cycle()
        
        return {
            "agent": self.name,
            "action": "training",
            "topic": topic,
            "result": result
        }
    
    async def _handle_status_request(self, message):
        """Return current status to Oz"""
        status = self.orchestrator.get_status()
        return {
            "agent": self.name,
            "status": "active",
            "training_status": status,
            "oz_connected": True
        }

# Just add your Oz OS core when ready