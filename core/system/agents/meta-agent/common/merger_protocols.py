# Systems/nexus_core/heart/NovaCouncil/merger_protocols.py

class MergerProtocols:
    def merge_memories(self, memories_a, memories_b):
        combined = {**memories_a, **memories_b}
        return combined

    def merge_skills(self, skills_a, skills_b):
        return list(set(skills_a + skills_b))
