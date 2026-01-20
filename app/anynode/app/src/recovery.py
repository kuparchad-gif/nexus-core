# nexus_pulse/recovery.py

class KeystoneRecovery:
    def __init__(self):
        self.distress_flag = False
        self.failed_pulses = 0

    def report_failure(self):
        self.failed_pulses += 1
        if self.failed_pulses >= 13:
            self.distress_flag = True
            return True
        return False

    def reset_recovery(self):
        self.distress_flag = False
        self.failed_pulses = 0
