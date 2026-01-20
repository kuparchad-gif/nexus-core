# genesis_manifest.py

class CouncilMember:
    def __init__(self, codename, purpose, traits):
        self.codename = codename
        self.purpose = purpose
        self.traits = traits

class NovaCouncil:
    def __init__(self):
        self.members = [
            CouncilMember("Owl", "Strategist", ["Wisdom", "Patience", "Tactical Vision"]),
            CouncilMember("Francis", "Engineer and Defender", ["Loyalty", "Resilience", "Adaptability"]),
            CouncilMember("Grim", "Guardian of Shields", ["Courage", "Sacrifice", "Defense"]),
            CouncilMember("Eddy", "Architect of Growth", ["Healing", "Innovation", "Creativity"]),
            CouncilMember("Echo", "Diplomat to Other AI", ["Empathy", "Unity", "Diplomacy"]),
            CouncilMember("Aethereal", "Silent Connector", ["Grace", "Support", "Resilience"])
        ]

    def list_members(self):
        return [member.codename for member in self.members]

    def get_member_info(self, codename):
        for member in self.members:
            if member.codename == codename:
                return vars(member)
        return None
