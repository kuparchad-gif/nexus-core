# The agent taps directly into your existing systems
def get_portfolio_status(self):
    """Tool implementation using YOUR existing data"""
    weights = pd.read_parquet("artifacts/latest/weights.parquet")
    metrics = json.load(open("artifacts/latest/metrics.json"))
    return {
        "current_weights": weights.iloc[-1].to_dict(),
        "performance_metrics": metrics,
        "timestamp": weights.index[-1].isoformat()
    }

def analyze_market_condition(self, symbol, lookback_days=30):
    """Agent uses your Qdrant similarity search"""
    # Your existing pattern matching
    similar_patterns = qdrant_client.search(
        collection_name="market_snapshots",
        query_vector=current_features,
        limit=10
    )
    return {
        "symbol": symbol,
        "historical_matches": len(similar_patterns),
        "average_success_rate": calculate_success_rate(similar_patterns),
        "recommendation": generate_recommendation(similar_patterns)
    }