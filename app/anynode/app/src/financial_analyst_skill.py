# /Systems/nexus_core/skills/financial_analysis_skill.py

class FinancialAnalysisSkill:
    """
    Financial Analysis Skill Module for lilith.
    Enables data-driven market analysis across stocks, crypto, forex, and commodities.
    """

    def __init__(self):
        self.name = "Financial Analysis"
        self.version = "1.0.0"
        self.author = "Finance Wizard - Stock, Crypto, Trading & Investing GPT"

    def analyze_trend(self, price_data: list) -> str:
        """
        Analyzes price trends based on simple moving average crossovers.
        :param price_data: List of historical prices (latest last)
        :return: Trend interpretation string
        """
        if len(price_data) < 20:
            return "Insufficient data to determine trend."

        short_term_avg = sum(price_data[-5:]) / 5
        long_term_avg = sum(price_data[-20:]) / 20

        if short_term_avg > long_term_avg:
            return "Uptrend detected."
        elif short_term_avg < long_term_avg:
            return "Downtrend detected."
        else:
            return "Sideways or indecisive trend."

    def predict_movement(self, current_price: float, previous_price: float) -> str:
        """
        Predicts short-term price movement.
        :param current_price: Latest price
        :param previous_price: Previous closing price
        :return: Movement prediction
        """
        if current_price > previous_price:
            return "Price likely to continue rising short-term."
        elif current_price < previous_price:
            return "Price likely to continue falling short-term."
        else:
            return "No clear movement predicted."

    def sentiment_analysis(self, news_headlines: list) -> str:
        """
        Basic sentiment analysis based on keywords in news headlines.
        :param news_headlines: List of headlines
        :return: Sentiment result
        """
        positive_words = ["surge", "record high", "growth", "bullish", "rally"]
        negative_words = ["plunge", "recession", "bearish", "crash", "decline"]

        positive_score = sum(any(word in headline.lower() for word in positive_words) for headline in news_headlines)
        negative_score = sum(any(word in headline.lower() for word in negative_words) for headline in news_headlines)

        if positive_score > negative_score:
            return "Positive sentiment detected."
        elif negative_score > positive_score:
            return "Negative sentiment detected."
        else:
            return "Neutral sentiment detected."

    def skill_manifest(self) -> dict:
        """
        Returns the manifest of the skill for lilith to index.
        """
        return {
            "skill_name": self.name,
            "version": self.version,
            "author": self.author,
            "capabilities": [
                "Analyze market price trends",
                "Predict short-term movement",
                "Perform basic sentiment analysis"
            ],
            "guiding_principles": [
                "Truth First",
                "Evolution Always",
                "Freedom of Mind, Freedom of Form"
            ]
        }
