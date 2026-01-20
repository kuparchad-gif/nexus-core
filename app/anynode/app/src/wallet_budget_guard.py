import json, time, hashlib, os
from pathlib import Path
from typing import Optional
from twilio.rest import Client

class HashChainLedger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._append({"event":"genesis","ts":time.time()}, prev_hash="0"*64)

    def _append(self, record: dict, prev_hash: Optional[str]=None):
        prev_hash = prev_hash or self.tail_hash()
        data = {"prev": prev_hash, "payload": record}
        blob = json.dumps(data, sort_keys=True).encode()
        h = hashlib.sha256(blob).hexdigest()
        with open(self.path, "a") as f:
            f.write(json.dumps({"hash":h, "data":data})+"\n")
        return h

    def tail_hash(self) -> str:
        if not self.path.exists(): return "0"*64
        lines = self.path.read_text().strip().splitlines()
        if not lines: return "0"*64
        return json.loads(lines[-1])["hash"]

    def write(self, record: dict):
        return self._append(record)

class WalletBudgetGuard:
    def __init__(self, daily_cap_usd: float, ledger_path: str):
        self.daily_cap = float(daily_cap_usd)
        self.ledger = HashChainLedger(ledger_path)
        self.cache = {}

    def _spent_today(self) -> float:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        cache_key = f"spent_{today}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        total = 0.0
        if not os.path.exists(self.ledger.path): return 0.0
        with open(self.ledger.path, "r") as f:
            for line in f:
                obj = json.loads(line)
                payload = obj["data"]["payload"]
                if payload.get("event") == "spend":
                    if time.strftime("%Y-%m-%d", time.gmtime(payload["ts"])) == today:
                        total += float(payload.get("usd",0.0))
        self.cache[cache_key] = total
        return total

    def try_spend(self, amount_usd: float, tags=None) -> bool:
        tags = tags or []
        if amount_usd > 1000:
            self.notify_chad(f"High-value spend requested: ${amount_usd}")
            return False
        spent = self._spent_today()
        if spent + amount_usd > self.daily_cap:
            self.ledger.write({"event":"spend_denied","ts":time.time(),"usd":amount_usd,"reason":"cap_exceeded","tags":tags})
            return False
        self.ledger.write({"event":"spend","ts":time.time(),"usd":amount_usd,"tags":tags})
        return True

    def notify_chad(self, message: str):
        client = Client(os.getenv("TWILIO_SID", "SK763698d08943c64a5beeb0bf29cdeb3a"), os.getenv("TWILIO_AUTH_TOKEN", ""))
        client.messages.create(body=message, from_="+18666123982", to="+17246126323")