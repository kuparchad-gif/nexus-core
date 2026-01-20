# Path: nexus_platform/memory_service/planner/planner.py
from common.logging import setup_logger
import json
import binascii
from cryptography.fernet import Fernet

class PlannerService:
    def __init__(self):
        self.logger = setup_logger("planner.module")
        self.cipher = Fernet(Fernet.generate_key())

    def assess_data(self, data: dict, emotions: list[str]) -> str:
        if emotions:
            binary_data = binascii.hexlify(json.dumps(data).encode()).decode()
            self.logger.info({"action": "assess_data", "type": "emotional", "binary_length": len(binary_data)})
            return binary_data
        self.logger.info({"action": "assess_data", "type": "logical"})
        return json.dumps(data)