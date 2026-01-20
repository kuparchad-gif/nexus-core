# src/live/websocket_feeds.py
class PolygonWebSocket:
    def __init__(self):
        self.ws = create_connection(f"wss://socket.polygon.io/stocks")
        self.ws.send(json.dumps({"action":"auth","params":API_KEY}))
    
    def subscribe(self, symbols):
        self.ws.send(json.dumps({"action":"subscribe","params":f"T.{','.join(symbols)}"}))
    
    def stream_bars(self, callback):
        while True:
            data = json.loads(self.ws.recv())
            if data[0]['ev'] == 'T':
                callback(parse_trade(data[0]))