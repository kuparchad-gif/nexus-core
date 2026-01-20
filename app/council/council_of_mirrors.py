
# council_of_mirrors.py
class CouncilOfMirrors:
    def __init__(self):
        self.archetypes = ["Nova", "Francis", "Grim", "Owl", "Lucien", "Dawn", "Solace", "Minstral", "Pixal", "Gemeni", "Clyde"]

    def summon(self, query):
        return {member: f"{member} reflects on: {query}" for member in self.archetypes}
