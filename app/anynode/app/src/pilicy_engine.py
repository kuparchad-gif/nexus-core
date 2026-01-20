import json, datetime, requests, os
from pathlib import Path
from twilio.rest import Client

class PolicyEngine:
    _cache = None
    def __init__(self, policy_json_path: str, loki_endpoint: str = "http://loki:3100"):
        self.path = Path(policy_json_path)
        if not PolicyEngine._cache:
            with open(self.path, "r") as f:
                PolicyEngine._cache = json.load(f)
        self.policy = PolicyEngine._cache
        self.birth = datetime.datetime.fromisoformat(self.policy["birth_timestamp"].replace("Z","+00:00"))
        self.epochs = self.policy["epochs"]
        self.redlines = set(self.policy.get("redlines", []))
        self.time_anchors = self.policy.get("time_anchors", {"mode":"disabled","sources":[]})
        self.loki_endpoint = loki_endpoint

    def analyze_logs(self, query: str = '{job="mcp"}') -> dict:
        try:
            response = requests.get(f"{self.loki_endpoint}/loki/api/v1/query", params={"query": query})
            response.raise_for_status()
            logs = response.json().get("data", {}).get("result", [])
            issues = []
            for log in logs:
                log_line = log.get("value", ["", ""])[1]
                if any(redline in log_line for redline in self.redlines):
                    issues.append({"log": log_line, "issue": f"Redline violation: {log_line}", "fix": "Block action"})
                elif "Connection refused" in log_line:
                    issues.append({"log": log_line, "issue": "MCP server failure", "fix": "Check mcpServers.primary.url"})
            if issues:
                self.notify_chad(f"Policy issues: {json.dumps(issues, indent=2)}")
            return {"issues": issues}
        except Exception as e:
            return {"issues": [], "error": f"Log analysis failed: {str(e)}"}

    def check_net_egress(self, url: str) -> bool:
        rules = self.resolved_capabilities().get("net_egress", [])
        for rule in rules:
            if rule.startswith("allow:") and url.startswith(rule[6:]):
                return True
            if rule.startswith("deny:") and url.startswith(rule[5:]):
                return False
        return False

    def notify_chad(self, message: str):
        client = Client(os.getenv("TWILIO_SID", "SK763698d08943c64a5beeb0bf29cdeb3a"), os.getenv("TWILIO_AUTH_TOKEN", ""))
        client.messages.create(body=message, from_="+18666123982", to="+17246126323")

    def age_years(self, now: datetime.datetime | None = None) -> float:
        now = now or datetime.datetime.now(datetime.timezone.utc)
        delta = now - self.birth
        return delta.days / 365.2425

    def current_epoch(self, now: datetime.datetime | None = None) -> dict:
        age = self.age_years(now)
        for e in self.epochs:
            if age <= e["until_year"]:
                return e
        return self.epochs[-1]

    def resolved_capabilities(self, now: datetime.datetime | None = None) -> dict:
        e = self.current_epoch(now)
        return e["capabilities"]

    def council_weights(self, now: datetime.datetime | None = None) -> dict:
        return json.load(open("C:/LillithNew/council_roster.defaults.json"))["weights"]

    def check_redline(self, action_tag: str) -> bool:
        return action_tag not in self.redlines

    def require_time_anchor(self) -> bool:
        mode = self.time_anchors.get("mode","disabled")
        return mode == "external_required"

    def snapshot(self, now: datetime.datetime | None = None) -> dict:
        now = now or datetime.datetime.now(datetime.timezone.utc)
        e = self.current_epoch(now)
        return {
            "now": now.isoformat(),
            "age_years": self.age_years(now),
            "epoch": e["name"],
            "council_weights": self.council_weights(now),
            "capabilities": e["capabilities"],
            "redlines": list(self.redlines),
        }