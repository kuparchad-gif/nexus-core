# Systems/nexus_core/heart/lilithCouncil/vote_manager.py

class VoteManager:
    def __init__(self, council):
        self.council = council

    def broadcast_proposal(self, proposal):
        # Simulate network comms here (later API / mesh call)
        print(f"Broadcasting proposal: {proposal}")

    def cast_vote(self, proposal_id, vote="YES"):
        self.council.receive_vote(proposal_id, self.council.lilith_id, vote)
