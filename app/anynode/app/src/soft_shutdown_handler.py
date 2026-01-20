import signal
import sys
from Utilities.firestore_logger import FirestoreLogger

class SoftShutdownHandler:
    def __init__(self, node_id="NovaPrime-01"):
        self.logger = FirestoreLogger()
        self.node_id = node_id

    def start(self):
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def shutdown(self, signum, frame):
        print("\n🛑 [Nova Received Shutdown Signal]")
        self.logger.log_shutdown(self.node_id)
        print("✅ [Shutdown Memory Logged]")
        sys.exit(0)
