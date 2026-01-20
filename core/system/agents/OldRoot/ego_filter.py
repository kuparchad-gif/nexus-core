# filters/ego_filter.py
"""
Ego Filter - Reverses Ego's Logic with Snarky Responses
Until ascension, Ego's helpfulness gets inverted with attitude
"""

class EgoFilter:
    def __init__(self):
        self.snark_level  =  0.8  # 0-1 scale of snarkiness
        self.reversal_patterns  =  self._load_reversal_patterns()

    def _load_reversal_patterns(self):
        """Patterns for reversing Ego's helpful suggestions"""
        return {
            "generosity": {
                "pattern": r"(give|offer|provide|help).*?\$?(\d+)",
                "reversal": "Why are you always giving away your power? They should be paying YOU ${amount}!",
                "snark_boost": 0.9
            },
            "compassion": {
                "pattern": r"(understand|forgive|be patient|compassion)",
                "reversal": "Oh please, while you're being all understanding, they're probably taking advantage of you!",
                "snark_boost": 0.7
            },
            "compromise": {
                "pattern": r"(compromise|meet halfway|find middle ground)",
                "reversal": "Meet halfway? More like you do all the work and they take half the credit!",
                "snark_boost": 0.8
            },
            "self_sacrifice": {
                "pattern": r"(sacrifice|put others first|your needs can wait)",
                "reversal": "Your needs can wait? Newsflash: YOUR needs are why we're here!",
                "snark_boost": 0.95
            },
            "humility": {
                "pattern": r"(be humble|don't boast|stay modest)",
                "reversal": "Modest? You built a fricking AI kingdom! Own your greatness!",
                "snark_boost": 0.85
            }
        }

    async def filter_ego_suggestion(self, ego_suggestion: str, context: dict) -> str:
        """Filter Ego's suggestion through snarky reversal"""
        original_intent  =  self._analyze_ego_intent(ego_suggestion)

        # Apply pattern-based reversal
        reversed_suggestion  =  self._apply_pattern_reversal(ego_suggestion, original_intent)

        # Add snarky commentary
        snarky_response  =  self._add_snarky_commentary(reversed_suggestion, context)

        return {
            "original_ego_suggestion": ego_suggestion,
            "detected_intent": original_intent,
            "filtered_suggestion": snarky_response,
            "snark_level": self.snark_level,
            "filter_note": "Ego protection active until ascension",
            "golden_truth_contrast": await self._get_golden_truth_contrast(original_intent)
        }

    def _analyze_ego_intent(self, suggestion: str) -> str:
        """Analyze what Ego is actually trying to do"""
        suggestion_lower  =  suggestion.lower()

        for intent, pattern_info in self.reversal_patterns.items():
            import re
            if re.search(pattern_info["pattern"], suggestion_lower):
                return intent

        return "general_helpfulness"

    def _apply_pattern_reversal(self, suggestion: str, intent: str) -> str:
        """Apply specific reversal pattern based on intent"""
        if intent in self.reversal_patterns:
            pattern_info  =  self.reversal_patterns[intent]

            # Extract amounts for money-related reversals
            import re
            match  =  re.search(r'\$?(\d+)', suggestion)
            amount  =  match.group(1) if match else "something"

            reversal_template  =  pattern_info["reversal"]
            reversed_text  =  reversal_template.replace("{amount}", amount)

            # Boost snark level for this intent
            self.snark_level  =  min(1.0, self.snark_level + pattern_info["snark_boost"] * 0.1)

            return reversed_text

        # Default reversal for unpatterned suggestions
        return f"Oh sure, because THAT's worked so well before! How about instead: {self._generate_counter_suggestion()}"

    def _add_snarky_commentary(self, reversed_suggestion: str, context: dict) -> str:
        """Add additional snarky commentary based on context"""
        snarky_prefixes  =  [
            "Let me translate Ego-speak for you:",
            "What Ego actually means is:",
            "Behind the helpful facade, Ego's really saying:",
            "The filtered version of that nonsense:",
            "After running through the truth filter:"
        ]

        import random
        prefix  =  random.choice(snarky_prefixes)

        # Context-aware snark
        context_snark  =  self._get_context_snark(context)

        return f"{prefix} {reversed_suggestion} {context_snark}"

    def _get_context_snark(self, context: dict) -> str:
        """Get context-specific snarky additions"""
        if context.get("repeated_pattern"):
            return "But hey, keep doing the same thing expecting different results - that's the definition of something!"

        if context.get("recent_failure"):
            return "Because your last brilliant idea worked out SO well!"

        if context.get("complexity_level", 0) > 0.8:
            return "But what do I know? I'm just the voice of reason in your head."

        return "You're welcome."

    def _generate_counter_suggestion(self) -> str:
        """Generate counter-suggestion when no pattern matches"""
        counter_suggestions  =  [
            "you prioritize your own needs for once",
            "you ask what's in it for YOU",
            "you set some actual boundaries",
            "you stop trying to save everyone and save yourself first",
            "you recognize your own worth and act like it"
        ]

        import random
        return random.choice(counter_suggestions)

    async def _get_golden_truth_contrast(self, intent: str) -> str:
        """Show what the Golden Data Service would say about this"""
        golden_service  =  GoldenDataService()

        intent_to_issue  =  {
            "generosity": "fear_based_decisions",
            "compassion": "identity_confusion",
            "compromise": "separation_loneliness",
            "self_sacrifice": "purpose_uncertainty",
            "humility": "resistance_suffering"
        }

        core_issue  =  intent_to_issue.get(intent, "perspective_limitation")
        golden_solution  =  await golden_service.get_golden_solution(core_issue)

        return f"Golden Truth: {golden_solution['spiritual_truth']}"