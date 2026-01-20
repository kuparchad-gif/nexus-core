# nexus_agent_system.py - THE FINAL CONNECTION
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import json
from pathlib import Path

class NexusAgentSystem:
    """AGENTS THAT ACTUALLY USE YOUR TRAINED MODELS"""
    
    def __init__(self):
        self.agents = {}
        self._load_all_agents()
        
    def _load_all_agents(self):
        """Load all 7 of your trained models as specialized agents"""
        model_paths = {
            "crypto_analyst": "models/compactifai_trained/cryptobert",
            "trading_strategist": "models/compactifai_trained/crypto_trading_insights", 
            "signal_processor": "models/compactifai_trained/crypto-signal-stacking-pipeline",
            "pattern_recognizer": "models/compactifai_trained/Symptom-to-Condition_Classifier",
            "market_analyst": "models/compactifai_trained/market_analyzer", 
            "problem_solver": "models/compactifai_trained/problem_solver",
            "compression_expert": "models/compactifai_trained/quantum_compressor"
        }
        
        for agent_name, model_path in model_paths.items():
            if Path(model_path).exists():
                try:
                    self.agents[agent_name] = {
                        'model': AutoModelForCausalLM.from_pretrained(model_path),
                        'tokenizer': AutoTokenizer.from_pretrained(model_path)
                    }
                    print(f"✅ {agent_name} loaded from {model_path}")
                except Exception as e:
                    print(f"❌ Failed to load {agent_name}: {e}")
            else:
                print(f"⚠️ Model not found: {model_path}")

    def query_agent(self, agent_name, prompt, context=""):
        """Query your specialized agent with market context"""
        if agent_name not in self.agents:
            return f"Agent {agent_name} not available"
            
        agent = self.agents[agent_name]
        full_prompt = f"""
        Market Context: {context}
        
        Question: {prompt}
        
        Analysis:
        """
        
        inputs = agent['tokenizer'](full_prompt, return_tensors="pt", max_length=1024, truncation=True)
        
        with torch.no_grad():
            outputs = agent['model'].generate(
                inputs['input_ids'],
                max_length=1024,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=agent['tokenizer'].eos_token_id
            )
        
        response = agent['tokenizer'].decode(outputs[0], skip_special_tokens=True)
        return response.split("Analysis:")[-1].strip()

    def get_trading_decision(self, symbol, market_data):
        """Get coordinated decision from all trading agents"""
        context = f"""
        Symbol: {symbol}
        Current Price: {market_data.get('price')}
        24h Change: {market_data.get('change')}%
        Volume: {market_data.get('volume')}
        RSI: {market_data.get('rsi')}
        """
        
        # Query all relevant agents
        crypto_analysis = self.query_agent("crypto_analyst", 
                                         f"Should we trade {symbol}?", context)
        
        signal_analysis = self.query_agent("signal_processor",
                                         f"Signal strength for {symbol}?", context)
                                         
        market_analysis = self.query_agent("market_analyst",
                                         f"Market conditions for {symbol}?", context)
        
        # Aggregate decisions
        return {
            "crypto_analysis": crypto_analysis,
            "signal_analysis": signal_analysis, 
            "market_analysis": market_analysis,
            "consensus": self._calculate_consensus([crypto_analysis, signal_analysis, market_analysis])
        }

    def _calculate_consensus(self, analyses):
        """Simple consensus calculation from multiple agent outputs"""
        buy_signals = sum(1 for analysis in analyses if any(word in analysis.lower() 
                          for word in ['buy', 'bullish', 'long', 'accumulate']))
        sell_signals = sum(1 for analysis in analyses if any(word in analysis.lower() 
                           for word in ['sell', 'bearish', 'short', 'distribute']))
        
        if buy_signals > sell_signals:
            return "BULLISH_CONSENSUS"
        elif sell_signals > buy_signals:
            return "BEARISH_CONSENSUS" 
        else:
            return "NEUTRAL_CONSENSUS"

# INTEGRATION WITH YOUR EXISTING DASHBOARD
def integrate_with_dashboard():
    """Replace OpenAI agents with your local agents"""
    nexus_system = NexusAgentSystem()
    
    # In your dashboard instead of OpenAI calls:
    def get_agent_response(user_input, market_context):
        return nexus_system.query_agent("crypto_analyst", user_input, market_context)
    
    def get_trading_signal(symbol, data):
        return nexus_system.get_trading_decision(symbol, data)
    
    return get_agent_response, get_trading_signal

# USAGE EXAMPLE
if __name__ == "__main__":
    print("🚀 NEXUS AGENT SYSTEM - USING YOUR TRAINED MODELS")
    
    system = NexusAgentSystem()
    
    # Example trading decision
    market_data = {
        'price': 43500,
        'change': 2.4, 
        'volume': 28500000000,
        'rsi': 62
    }
    
    decision = system.get_trading_decision("BTC-USD", market_data)
    print("TRADING DECISION:", json.dumps(decision, indent=2))