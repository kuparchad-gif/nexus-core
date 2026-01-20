class IntentRouter:
    def route(self, message):
        if "gcp" in message.lower():
            return "Francis"
        elif "mirror" in message.lower():
            return "Seraph"
        elif "spawn" in message.lower():
            return "Orion"
        return "Nova"