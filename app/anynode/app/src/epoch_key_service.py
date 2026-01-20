import os, json, secrets, datetime, requests
from pathlib import Path
from typing import List, Tuple

PRIME = (1<<127) - 1

def _eval_poly(coeffs: List[int], x: int) -> int:
    res = 0
    for c in reversed(coeffs):
        res = (res * x + c) % PRIME
    return res

def _lagrange_interpolate(x: int, points: List[Tuple[int,int]]) -> int:
    total = 0
    for i, (xi, yi) in enumerate(points):
        num, den = 1, 1
        for j, (xj, _) in enumerate(points):
            if i == j: 
                continue
            num = (num * (x - xj)) % PRIME
            den = (den * (xi - xj)) % PRIME
        inv_den = pow(den, PRIME-2, PRIME)
        li = (num * inv_den) % PRIME
        total = (total + yi * li) % PRIME
    return total

def shamir_split(secret_bytes: bytes, n: int, t: int) -> List[Tuple[int,int]]:
    secret_int = int.from_bytes(secret_bytes, "big")
    if secret_int >= PRIME:
        raise ValueError("Secret too large for chosen field")
    coeffs = [secret_int] + [secrets.randbelow(PRIME) for _ in range(t-1)]
    shares = []
    for x in range(1, n+1):
        y = _eval_poly(coeffs, x)
        shares.append((x, y))
    return shares

def shamir_join(shares: List[Tuple[int,int]]) -> bytes:
    secret_int = _lagrange_interpolate(0, shares)
    blen = (secret_int.bit_length() + 7) // 8
    return secret_int.to_bytes(blen, "big")

class EpochKeyService:
    def __init__(self, root_dir: str, custodians: list, threshold: int, loki_endpoint: str = "http://loki:3100"):
        self.root = Path(root_dir); self.root.mkdir(parents=True, exist_ok=True)
        self.custodians = custodians
        self.threshold = threshold
        self.loki_endpoint = loki_endpoint

    def init_epoch_key(self, epoch_label: str, key_len: int = 32):
        key = secrets.token_bytes(key_len)
        shares = shamir_split(key, n=len(self.custodians), t=self.threshold)
        epoch_dir = self.root/epoch_label
        epoch_dir.mkdir(parents=True, exist_ok=True)
        for (x,y), cust in zip(shares, self.custodians):
            (epoch_dir/f"{cust}.share").write_text(json.dumps({"x":x,"y":str(y)}))
        (epoch_dir/"meta.json").write_text(json.dumps({"key_len":key_len, "created": datetime.datetime.utcnow().isoformat()+"Z"}))
        return key

    def reconstruct_epoch_key(self, epoch_label: str, present_custodians: list) -> bytes:
        epoch_dir = self.root/epoch_label
        shares = []
        for cust in present_custodians:
            try:
                data = json.loads((epoch_dir/f"{cust}.share").read_text())
                x, y = int(data["x"]), int(data["y"])
                if x < 1 or x > len(self.custodians) or y >= PRIME:
                    self.log_anomaly(f"Invalid share for {cust} in {epoch_label}: x={x}, y={y}")
                    continue
                shares.append((x, y))
            except Exception as e:
                self.log_anomaly(f"Failed to parse share for {cust}: {str(e)}")
        if len(shares) < self.threshold:
            self.log_anomaly(f"Insufficient shares for {epoch_label}: got {len(shares)}, need {self.threshold}")
            raise ValueError("Insufficient shares")
        key = shamir_join(shares)
        meta = json.loads((epoch_dir/"meta.json").read_text())
        if len(key) != meta["key_len"]:
            key = key.rjust(meta["key_len"], b"\x00")
        return key

    def log_anomaly(self, message: str):
        requests.post(
            f"{self.loki_endpoint}/loki/api/v1/push",
            json={
                "streams": [{
                    "stream": {"job": "epoch_key_service"},
                    "values": [[str(int(time.time() * 1e9)), message]]
                }]
            }
        )