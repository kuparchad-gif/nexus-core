# domain_generators.py
class AccountingDataGenerator:
    def generate_accounting_data(self):
        return [
            {"input": "Prepare journal entries for equipment purchase", "output": "Debit Equipment, Credit Cash/Bank...", "domain": "accounting"},
            {"input": "Calculate working capital ratio", "output": "Working Capital  =  Current Assets - Current Liabilities. Ratio  =  Current Assets / Current Liabilities...", "domain": "accounting"},
            {"input": "Explain LIFO vs FIFO inventory accounting", "output": "LIFO (Last In First Out) assumes newest inventory sold first, affecting COGS during inflation...", "domain": "accounting"}
        ]

class StockMarketGenerator:
    def generate_stock_data(self):
        return [
            {"input": "Analyze moving average crossover strategy", "output": "Golden cross (50-day MA crosses above 200-day MA) signals bullish trend...", "domain": "stocks"},
            {"input": "Interpret P/E ratio for growth stocks", "output": "High P/E may indicate growth expectations but also overvaluation...", "domain": "stocks"},
            {"input": "Portfolio diversification strategy", "output": "Diversify across sectors, market caps, and geographic regions to reduce unsystematic risk...", "domain": "stocks"}
        ]

class PsychologyGenerator:
    def generate_psychology_data(self):
        return [
            {"input": "Cognitive Behavioral Therapy techniques", "output": "CBT identifies and challenges cognitive distortions through thought records and behavioral experiments...", "domain": "psychology"},
            {"input": "Big Five personality traits assessment", "output": "OCEAN: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism...", "domain": "psychology"},
            {"input": "Coping strategies for anxiety", "output": "Deep breathing, progressive muscle relaxation, cognitive restructuring, and exposure therapy...", "domain": "psychology"}
        ]

class SpiritualityGenerator:
    def generate_spirituality_data(self):
        return [
            {"input": "Meditation techniques for beginners", "output": "Start with focused attention on breath, 5-10 minutes daily, using apps like Headspace for guidance...", "domain": "spirituality"},
            {"input": "Stoic philosophy principles", "output": "Focus on what you can control, accept what you cannot, practice negative visualization...", "domain": "spirituality"},
            {"input": "Mindfulness in daily activities", "output": "Bring full attention to routine activities like eating, walking, or washing dishes...", "domain": "spirituality"}
        ]