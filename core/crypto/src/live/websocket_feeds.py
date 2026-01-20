```python
import os
import json
import time
from websocket import WebSocketApp
from ..utils.logging_config import logger

class PolygonWebSocket:
    def __init__(self):
        self.api_key = os.getenv("POLYGON_API_KEY")
        self.ws = None
        self.reconnect_attempts = 0
        self.max_reconnects = 5
        self.callback = None

    def connect(self):
        if not self.api_key:
            logger.error("POLYGON_API_KEY not set in .env")
            raise ValueError("Polygon API key missing")
        self.ws = WebSocketApp(
            "wss://socket.polygon.io/stocks",
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        self.ws.run_forever()

    def on_open(self, ws):
        self.reconnect_attempts = 0
        ws.send(json.dumps({"action":"auth","params":self.api_key}))
        logger.info("Polygon websocket connected")

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            if data[0].get('ev') == 'T' and self.callback:
                self.callback(self.parse_trade(data[0]))
            logger.debug(f"Websocket received: {message}")
        except Exception as e:
            logger.error(f"Websocket message parse failed: {e}")

    def on_error(self, ws, error):
        logger.error(f"Websocket error: {error}")
        if self.reconnect_attempts < self.max_reconnects:
            self.reconnect_attempts += 1
            time.sleep(2 ** self.reconnect_attempts)
            self.connect()
        else:
            logger.error("Max reconnects reached; websocket failed")

    def on_close(self, ws, code, reason):
        logger.warning(f"Websocket closed: code={code}, reason={reason}")

    def subscribe(self, symbols):
        if self.ws:
            self.ws.send(json.dumps({"action":"subscribe","params":f"T.{','.join(symbols)}"}))
            logger.info(f"Subscribed to symbols: {symbols}")

    def parse_trade(self, data):
        return {
            "symbol": data.get("sym"),
            "price": data.get("p"),
            "volume": data.get("s"),
            "timestamp": data.get("t")
        }

    def stream_bars(self, callback):
        self.callback = callback
        self.connect()
```