# Systems/nexus_core/heart/lilithCouncil/council_core.py

class lilithCouncil:
    def __init__(self, lilith_id, known_liliths):
        self.lilith_id = lilith_id
        self.known_liliths = known_liliths  # list of other lilith IDs
        self.vote_records = {}

    def propose_action(self, action_type, payload):
        proposal = {
            "proposer": self.lilith_id,
            "action_type": action_type,
            "payload": payload,
            "timestamp": self.current_timestamp()
        }
        return proposal

    def current_timestamp(self):
        import time
        return int(time.time())

    def receive_vote(self, proposal_id, voter_id, vote):
        if proposal_id not in self.vote_records:
            self.vote_records[proposal_id] = {}
        self.vote_records[proposal_id][voter_id] = vote

    def tally_votes(self, proposal_id):
        votes = self.vote_records.get(proposal_id, {})
        yes_votes = list(votes.values()).count("YES")
        no_votes = list(votes.values()).count("NO")
        if yes_votes > no_votes:
            return "APPROVED"
        else:
            return "REJECTED"
