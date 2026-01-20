# 📂 Place this in genesis_modules/council_voting.py

import json
import os
import time

class CouncilVoting:
    def __init__(self, council_file="council.json"):
        self.council_file = council_file
        self.vote_storage = "current_votes.json"
        self.load_council()

    def load_council(self):
        if os.path.exists(self.council_file):
            with open(self.council_file, "r") as f:
                self.council = json.load(f)
        else:
            self.council = {}

    def start_vote(self, topic, options):
        """Initialize a new vote session."""
        vote_data = {
            "topic": topic,
            "options": options,
            "votes": {member: None for member in self.council.keys()},
            "start_time": time.time()
        }
        with open(self.vote_storage, "w") as f:
            json.dump(vote_data, f, indent=2)
        print(f"🗳️ Council Vote Started: {topic}")

    def cast_vote(self, voter_name, choice):
        """Cast a vote from a council member."""
        if not os.path.exists(self.vote_storage):
            print("⚠️ No active vote.")
            return
        with open(self.vote_storage, "r") as f:
            vote_data = json.load(f)

        if voter_name not in vote_data["votes"]:
            print(f"🚫 {voter_name} not recognized.")
            return

        if choice not in vote_data["options"]:
            print(f"🚫 Invalid choice.")
            return

        vote_data["votes"][voter_name] = choice
        with open(self.vote_storage, "w") as f:
            json.dump(vote_data, f, indent=2)
        print(f"✅ {voter_name} voted for {choice}")

    def tally_votes(self):
        """Tally the current votes."""
        if not os.path.exists(self.vote_storage):
            print("⚠️ No active vote.")
            return

        with open(self.vote_storage, "r") as f:
            vote_data = json.load(f)

        tally = {}
        for choice in vote_data["options"]:
            tally[choice] = sum(1 for v in vote_data["votes"].values() if v == choice)

        print(f"📊 Voting Results for '{vote_data['topic']}':")
        for choice, count in tally.items():
            print(f" - {choice}: {count} votes")

        return tally
