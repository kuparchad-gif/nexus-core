# Systems/nexus_core/heart/NovaCouncil/council_core.py

class NovaCouncil:
    def __init__(self, nova_id, known_novas):
        self.nova_id = nova_id
        self.known_novas = known_novas  # list of other Nova IDs
        self.vote_records = {}

    def propose_action(self, action_type, payload):
        proposal = {
            "proposer": self.nova_id,
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
