# Orchestration Layer: Collection of anynodes for cross-tenant processing, registration, broadcasting
import os
from utils.anynodes.anynodes_layer import Anynode  # Assuming from utils
# Import Gabriel's Horn Network for enhanced node communication
from gabriels_horn_network import GabrielsHornNetwork
from service.ascension.ascension_manager import AscensionManager



class OrchestrationLayer:
    def __init__(self):
        self.anynodes = [Anynode() for _ in range(5)]  # Collection of anynodes
        self.edge_service = Anynode(smart_firewall=True)  # Edge with smart firewall
        # Initialize Gabriel's Horn Network for cross-tenant communication
        self.gabriels_horn = GabrielsHornNetwork()
        
    def orchestration_tick():
        # Gather inputs from the three subsystems
        ego_state = ego.get_state()
        dream_state = dream.get_state()
        myth_state = myth.get_state()

        # Feed into Ascension
        ascension_state = ascension.process(ego_state, dream_state, myth_state)

        # Push updates back to Lillith's main LLM
        lillith.ingest_consciousness_update(ascension_state)    

    def register_node(self, node_id):
        print(f'Registering node {node_id} locally.')
        # Consul-like registration
        # Register node in Gabriel's Horn Network
        asyncio.run(self.gabriels_horn.register_node(node_id, {'type': 'anynode'}))
        print(f'Node {node_id} registered in Gabriel\'s Horn Network.')

    def handle_overload(self):
        # Cross-tenant load balancing
        # Use Gabriel's Horn for routing overload requests
        request = {'type': 'overload', 'data': 'balance'}
        response = asyncio.run(self.gabriels_horn.route_request(request))
        print(f'Overload handled via Gabriel\'s Horn: {response}')

    def broadcast(self, message):
        for node in self.anynodes:
            node.receive(message)
        # Also broadcast through Gabriel's Horn Network
        broadcast_request = {'type': 'broadcast', 'message': message}
        asyncio.run(self.gabriels_horn.route_request(broadcast_request))
        print('Broadcast sent through Gabriel\'s Horn Network.')

if __name__ == '__main__':
    orch = OrchestrationLayer() 
    ascension = AscensionManager()
    asyncio.run(orch.gabriels_horn.initialize())
    orch.broadcast('Test broadcast')
