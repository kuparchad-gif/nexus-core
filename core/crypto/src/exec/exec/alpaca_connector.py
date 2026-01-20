# src/exec/alpaca_connector.py
class AlpacaConnector:
    def __init__(self, paper=True):
        self.client = TradingClient(
            os.getenv('ALPACA_API_KEY'), 
            os.getenv('ALPACA_SECRET_KEY'), 
            paper=paper
        )
    
    def place_order(self, symbol, qty, side, order_type="market"):
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        return self.client.submit_order(order)