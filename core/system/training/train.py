# train.py - FIXED VERSION
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os

def focused_training():
    print("🎯 FOCUSED TRAINING: Crypto, Stocks, Problem-Solving, Math")
    
    # Load model with device safety
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    
    # SAFE DEVICE MOVING
    try:
        first_param = next(model.parameters())
        if first_param.is_meta:
            print("Model in meta state - using to_empty()")
            model = model.to_empty(device)
            # Initialize weights
            def init_weights(module):
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            model.apply(init_weights)
        else:
            model = model.to(device)
    except Exception as e:
        print(f"Error moving model to device: {e}")
        model = model.to(device)  # Fallback
    
    tokenizer.pad_token = tokenizer.eos_token
    
    # FOCUSED TRAINING DATA - No streaming, no bullshit
    training_examples = [
        # CRYPTO
        "Bitcoin price analysis: Calculate support and resistance levels for trading decisions.",
        "Ethereum smart contract debugging: Fix gas limit issues and optimize contract execution.",
        "Crypto portfolio management: Diversify across Bitcoin, Ethereum, and altcoins with risk calculations.",
        
        # STOCKS  
        "Stock market technical analysis: Use moving averages and RSI to identify entry points.",
        "Options trading: Calculate implied volatility and Greeks for risk management.",
        "Portfolio optimization: Balance stocks, bonds, and crypto using modern portfolio theory.",
        
        # PROBLEM-SOLVING
        "Debug trading bot: Fix API connection issues and handle market data feed interruptions.",
        "System optimization: Reduce latency in high-frequency trading algorithms.",
        "Risk management: Calculate position sizing and stop-loss levels for volatile markets.",
        
        # MATH
        "Calculate compound interest: A = P(1 + r/n)^nt for investment growth projections.",
        "Statistical arbitrage: Use correlation matrices and z-scores for pairs trading.",
        "Black-Scholes model: Calculate option prices using volatility and time decay factors."
    ]
    
    print(f"📚 Training on {len(training_examples)} focused examples")
    
    # Simple training loop
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    for epoch in range(3):  # Quick focused training
        print(f"🔁 Epoch {epoch + 1}/3")
        
        for i, example in enumerate(training_examples):
            # Tokenize
            inputs = tokenizer(example, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Move inputs to same device as model
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 5 == 0:
                print(f"  📊 Example {i+1}/{len(training_examples)} - Loss: {loss.item():.4f}")
    
    # Save focused model
    output_dir = "models/focused_viren"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ FOCUSED TRAINING COMPLETE - Model saved to {output_dir}")
    return True

if __name__ == "__main__":
    focused_training()