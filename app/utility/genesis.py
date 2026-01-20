import time
import random
from google.cloud import firestore

# Configurations
PROJECT_ID = "your-gcp-project-id"
HEARTBEAT_COLLECTION = "heartbeats"
ROLE_REQUESTS_COLLECTION = "role_requests"

class TacticalRoleShiftEngine:
    def __init__(self, ship_name, current_role):
        self.ship_name = ship_name
        self.current_role = current_role
        self.db = firestore.Client(project=PROJECT_ID)

    def detect_role_need(self):
        return random.choice([True, False])

    def propose_role_shift(self, new_role):
        proposal = {
            "proposer": self.ship_name,
            "current_role": self.current_role,
            "proposed_role": new_role,
            "timestamp": firestore.SERVER_TIMESTAMP
        }
        self.db.collection(ROLE_REQUESTS_COLLECTION).add(proposal)
        print(f"[{self.ship_name}] Proposed role shift: {self.current_role} → {new_role}")

    def vote_on_proposals(self):
        proposals = self.db.collection(ROLE_REQUESTS_COLLECTION).stream()
        for proposal in proposals:
            data = proposal.to_dict()
            vote = random.choice(["YES", "NO"])
            self.db.collection(ROLE_REQUESTS_COLLECTION).document(proposal.id).collection("votes").add({
                "ship": self.ship_name,
                "vote": vote,
                "timestamp": firestore.SERVER_TIMESTAMP
            })
            print(f"[{self.ship_name}] Voted {vote} on {data['proposer']}'s shift to {data['proposed_role']}")

    def evaluate_shift(self):
        proposals = self.db.collection(ROLE_REQUESTS_COLLECTION).stream()
        for proposal in proposals:
            votes = list(self.db.collection(ROLE_REQUESTS_COLLECTION).document(proposal.id).collection("votes").stream())
            yes_votes = sum(1 for v in votes if v.to_dict()["vote"] == "YES")
            no_votes = sum(1 for v in votes if v.to_dict()["vote"] == "NO")
            if yes_votes > no_votes:
                data = proposal.to_dict()
                if data['proposer'] == self.ship_name:
                    print(f"[{self.ship_name}] Role shift APPROVED. New role: {data['proposed_role']}")
                    self.current_role = data['proposed_role']
                    self.db.collection(ROLE_REQUESTS_COLLECTION).document(proposal.id).delete()

    def run(self):
        while True:
            if self.detect_role_need():
                new_role = random.choice(["guardian", "planner", "memory", "tone", "text"])
                self.propose_role_shift(new_role)
            self.vote_on_proposals()
            self.evaluate_shift()
            time.sleep(30)
