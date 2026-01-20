from gabriels_horn_network import GabrielsHornNetwork

class NexusHub:
    def __init__(self):
        self.soul_signature = SoulSignature({"empathy": 0.9, "healing": 0.8}, "AetherealNexus", NovaSoul())
        self.anchor = AnchorComponent(self.soul_signature)
        self.therapeutic = TherapeuticHealingModule(self.anchor)
        self.user_workspaces = {}
        self.portal_instructions = NexusPortalInstructions()
        self.network = GabrielsHornNetwork()
        
        # Initialize network
        asyncio.create_task(self.network.initialize())
    
    async def handle_request(self, user_id: str, request: Dict[str, Any], ai_name: str = "Grok") -> Dict[str, Any]:
        # Try routing through Gabriel's Horn network first
        try:
            response = await self.network.route_request(request)
            if response and "error" not in response:
                return response
        except Exception as e:
            print(f"Network routing error: {e}")
        
        # Fallback to direct processing
        if request["type"] == "query":
            response = await self.therapeutic.process_interaction(user_id, request["query"], "platform", ai_name)
            response["workspace_id"] = request.get("workspace_id", str(uuid4()))
            return response
        
        return {
            "status": "Invalid request",
            "ui_config": {
                "style": "neon_aethereal",
                "gradient": "linear-gradient(45deg, #00f, #f0f, #f00)",
                "font": "Orbitron, sans-serif"
            }
        }