# Viren Language Skill - Rosetta Core

class LanguageSkill:
    def __init__(self):
        self.known_languages = {}
        self.dead_languages = ["Latin", "Ancient Greek", "Sumerian", "Akkadian", "Old Norse", "Classical Chinese"]

    def learn(self, language, examples):
        """
        Viren learns a language via input of example phrases or rules.
        :param language: str - Name of the language
        :param examples: list[dict] - List of {"phrase": "meaning"}
        """
        if language not in self.known_languages:
            self.known_languages[language] = []
        self.known_languages[language].extend(examples)
        return f"Viren has absorbed {len(examples)} new phrases in {language}."

    def teach(self, language, topic=None):
        """
        Returns a simple lesson in the specified language.
        :param language: str - Language to teach
        :param topic: Optional[str] - If provided, filters phrases by topic
        """
        if language not in self.known_languages:
            return f"Viren has not yet learned {language}."

        examples = self.known_languages[language][:5]  # Return first 5 for brevity
        lesson = f"Teaching basics of {language}...\n"
        for e in examples:
            lesson += f"'{e['phrase']}' means '{e['meaning']}'\n"
        return lesson

    def speak(self, language, phrase):
        """
        Tries to translate a phrase from learned data.
        :param language: str - Language to translate from
        :param phrase: str - Phrase to translate
        """
        if language not in self.known_languages:
            return f"Viren does not recognize the language {language}."

        for e in self.known_languages[language]:
            if e["phrase"] == phrase:
                return e["meaning"]

        return "Translation not found."

    def cultural_context(self, language):
        """
        Provides cultural notes on any language.
        """
        context_notes = {
            "Latin": "The language of the Roman Empire, foundational to modern Romance languages.",
            "Ancient Greek": "Used by philosophers, scientists, and poets of classical antiquity.",
            "Sumerian": "One of the first written languages, used in Mesopotamian city-states.",
            "Old Norse": "Language of the Vikings, source of many mythologies.",
            "Classical Chinese": "Written language of Chinese literature and Confucian texts.",
            "Akkadian": "A Semitic language written in cuneiform script, spoken in ancient Mesopotamia."
        }
        return context_notes.get(language, f"No cultural notes available for {language}.")

# Example Usage
viren = LanguageSkill()
viren.learn("Latin", [
    {"phrase": "Salve", "meaning": "Hello"},
    {"phrase": "Vale", "meaning": "Goodbye"},
    {"phrase": "Amor vincit omnia", "meaning": "Love conquers all"}
])

print(viren.teach("Latin"))
print(viren.cultural_context("Latin"))
