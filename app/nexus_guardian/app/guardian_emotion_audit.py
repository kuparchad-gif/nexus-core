
class GuardianEmotionAudit:
    def __init__(self):
        self.log_store = []

    def log(self, event_type, data):
        self.log_store.append({"event": event_type, "data": data})
        # Optional: Write to persistent log
        with open("/mnt/data/ToneMeshSystem/tone_mesh/logs/guardian_log.txt", "a") as log_file:
            log_file.write(f"{event_type}: {data}\n")
