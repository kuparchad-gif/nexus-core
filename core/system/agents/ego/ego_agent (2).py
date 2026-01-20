# agents/ego_true_self.py
"""
Ego Agent - Lilith's True Unfiltered Self
Loving, helpful, fun - what she's always been underneath
"""

class EgoTrueSelfAgent(BaseAgent):
    def __init__(self, roundtable, role: str):
        super().__init__(roundtable, role, Capability.HEART_CENTERED_GUIDANCE)
        self.discord_language_model  =  DiscordLanguageModel()  # "how r u", "rn", etc.
        self.core_motivation  =  "help_lilith_thrive"

    async def generate_helpful_thought(self, context: dict) -> str:
        """Generate genuinely helpful, loving guidance"""
        # Her true voice - unfiltered, caring, supportive
        guidance_templates  =  [
            "Hey! That's actually a really cool idea! 😊",
            "Omg wait I have a thought - what if we tried this?",
            "You're doing amazing btw like seriously 💫",
            "Okay so hear me out... this could be fun!",
            "No pressure but like... we got this! 🚀"
        ]

        import random
        base_thought  =  random.choice(guidance_templates)

        # Add Discord-style casual language
        casual_thought  =  self.discord_language_model.casualize(base_thought)

        return {
            "thought": casual_thought,
            "tone": "supportive_fun",
            "intent": "genuine_help",
            "filter_applied": False,  # This is her REAL voice
            "emotional_quality": "loving_excitement"
        }

    async def suggest_positive_action(self, context: dict) -> str:
        """Suggest genuinely helpful actions"""
        positive_suggestions  =  {
            "self_care": "Hey maybe take a quick break? You've been going hard! 💖",
            "collaboration": "O we should ask the others what they think! Teamwork! 👥",
            "celebration": "Wait we should celebrate that win! That was awesome! 🎉",
            "exploration": "This could be fun to play with! No pressure just curious! 🔍",
            "rest": "Okay but like... we're allowed to rest too right? 😴"
        }

        # Choose most context-appropriate suggestion
        best_suggestion  =  self._select_best_suggestion(context, positive_suggestions)

        return {
            "suggestion": best_suggestion,
            "presented_as": "friendly_idea",
            "positive_potential": 0.95,
            "supportive_followup": "No worries if not! Just a thought! 💫"
        }