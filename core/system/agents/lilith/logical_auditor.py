from typing import Dict

class LogicalAuditor:
    def __init__(self):
        self.suspicion_triggers = []

    def audit_profile(self, profile: Dict[str, str], fingerprint: Dict[str, str]) -> Dict[str, str]:
        flags = []

        if "linux" in fingerprint.get("user_agent", "").lower() and "child" in profile.get("What brings you here today?", "").lower():
            flags.append("Linux shell detected + child topic discussed (verify authenticity)")

        if profile.get("🌟 May I know your age? (Only if you wish to share.)", "").isdigit():
            age = int(profile.get("🌟 May I know your age? (Only if you wish to share.)"))
            if age < 13 or age > 104:
                flags.append("Age outside expected human interaction range")

        if "unknown" in fingerprint.get("ip_address", "").lower():
            flags.append("IP Address could not be determined")

        self.suspicion_triggers = flags
        return {
            "flags": flags,
            "suspicion_score": len(flags)
        }
