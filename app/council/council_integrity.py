# Systems/Config/barricade/council/council_integrity.py

class CouncilIntegrity:
    def __init__(self):
        self.seats = {
            "vote": False,
            "guardian": False,
            "executor": False,
            "evolution": False
        }

    def register_seat(self, name):
        if name in self.seats:
            self.seats[name] = True

    def is_quorum(self):
        return all(self.seats.values())

    def report(self):
        return {
            "status": "✅ Quorum Achieved" if self.is_quorum() else "🕓 Waiting on full council...",
            "seats": self.seats
        }
