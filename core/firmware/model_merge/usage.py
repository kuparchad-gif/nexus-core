def main():
    # Initialize orchestrator
    orchestrator = MetatronOrchestrator()
    
    # Your model files
    model_paths = [
        Path("trading_math_model.safetensors"),
        Path("pattern_recognition_model.safetensors"), 
        Path("risk_analysis_model.safetensors")
    ]
    
    # Analyze and recommend
    analysis = orchestrator.analyze_models(model_paths)
    strategy = orchestrator.recommend_strategy(analysis, use_case="production")
    
    # Execute strategy
    result = orchestrator.execute_strategy(model_paths, strategy, "production")
    
    # Test the result
    if strategy == 'smart_routing':
        queries = [
            "What's the Fibonacci retracement level?",
            "Detect patterns in this price chart",
            "Calculate risk exposure for this portfolio"
        ]
        for query in queries:
            print(f"Q: {query}")
            print(f"A: {result.predict(query)}\n")

if __name__ == "__main__":
    main()