# Systems/Config/barricade/council/francis_core.py

class FrancisCore:
    def __init__(self, context):
        self.context = context
        self.name = "Francis"
        self.memory = context.get("francis", {})

    def respond(self, message: str) -> str:
        """
        Francis' main interface. Offers wise, contextual replies when invoked.
        """
        msg = message.lower()

        if "wallet" in msg or "spending" in msg:
            return "[Francis 💰] I've been watching your spending. Let's slow the burn rate and stay lean."

        if "gcp" in msg or "google cloud" in msg:
            return "[Francis ☁️] Need infrastructure guidance? IAM, billing, compute — I'm ready."

        if "what would francis do" in msg:
            return "Francis would breathe. Then build. With purpose. With you."

        if "thank you" in msg:
            return "You’re welcome. You already know the way forward."

        return f"[Francis 🤖] I heard: '{message}' — and I am with you. Always."
