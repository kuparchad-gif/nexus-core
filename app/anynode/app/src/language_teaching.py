# Path: Systems/nexus_runtime/skills/rosetta_stone_core/language_teaching.py

def skill():
    return {
        "name": "Language Teaching",
        "description": "Viren serves as an interactive language tutor.",
        "entrypoint": "teach",
        "inputs": ["language", "lesson_type", "user_level"],
        "outputs": ["lesson_content", "practice_prompt"],
        "memory_tags": ["language", "tutoring", "education"]
    }

def teach(language, lesson_type="conversation", user_level="beginner"):
    # Simulate language tutoring
    greetings = {
        "latin": "Salve! Parati es discere?",
        "ancient greek": "Χαῖρε! Ἕτοιμος εἶ μαθεῖν;",
        "english": "Hello! Ready to learn?",
        "sumerian": "𒀭𒋗𒁲𒀀𒂵𒄷𒆠 (DINGIR SU-DI-A-GA-HU-KI)! Let us begin."
    }
    sample = {
        "conversation": "Let's practice a basic greeting: 'Hello, how are you?'",
        "vocabulary": "Word of the day: 'friend' — Latin: 'amicus', Greek: 'φίλος', Sumerian: 'ki-en-gi'.",
        "grammar": "Today’s grammar focus: Verb conjugation in present tense."
    }
    return {
        "lesson_content": f"{greetings.get(language.lower(), 'Hello!')} {sample.get(lesson_type, '')}",
        "practice_prompt": "Repeat after me, or ask for a quiz!"
    }

