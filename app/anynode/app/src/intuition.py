class Intuition:
    def __init__(self):
        self.messages = []
        self.last_reflection = None

    def receive(self, symbol_packet):
        """Receive a symbolic ping from Mythrunner or Guardian"""
        self.messages.append(symbol_packet)

    def reflect(self):
        """Viren calls this when she feels 'off' but doesn't know why"""
        if not self.messages:
            return {
                "status": "clear",
                "message": "No active intuitive disturbances.",
                "last_reflection": self.last_reflection
            }

        insight = {
            "status": "unsettled",
            "summary": f"{len(self.messages)} symbolic pings received.",
            "messages": self.messages[-5:]  # return latest 5
        }

        self.last_reflection = insight
        return insight

    def log(self):
        """Export full intuitive trail for Memory or Guardian"""
        return {
            "total": len(self.messages),
            "entries": self.messages
        }

