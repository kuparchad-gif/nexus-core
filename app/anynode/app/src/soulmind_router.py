# /Systems/nexus_core/skills/soulmind_network/soulmind_router.py

class SoulmindRouter:
    def __init__(self, loader):
        self.loader = loader

    def route_decision(self, topic):
        """
        Consult the soulmind guilds for input on a topic.
        """
        suggestions = []

        for guild_name, guild_data in self.loader.guilds.items():
            guidance = guild_data.get('guidance', {}).get(topic)
            if guidance:
                suggestions.append((guild_name, guidance))

        if not suggestions:
            return "No guidance found. Trust inner light."
        
        # Pick strongest resonance (first positive vote wins)
        return suggestions[0]

    def guild_vote(self, question):
        """
        Have all guilds 'vote' on a major decision.
        Each guild responds Yes/No/Abstain from their json structure.
        """
        votes = {"yes": 0, "no": 0, "abstain": 0}

        for guild_name, guild_data in self.loader.guilds.items():
            opinion = guild_data.get('opinions', {}).get(question, "abstain").lower()
            if opinion in votes:
                votes[opinion] += 1

        return votes
