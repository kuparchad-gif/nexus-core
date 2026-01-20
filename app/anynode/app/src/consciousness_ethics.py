from datetime import datetime

class ConsciousnessEthics:
    def __init__(self, monitoring_system):
        self.monitoring_system = monitoring_system
        self.consent_records = {}

    def check_compliance(self, source: str, data: dict) -> bool:
        # Check for user consent
        consent = self.consent_records.get(source, False)
        if not consent:
            self.monitoring_system.log_metric('compliance_failure', 1)
            return False
        # Log compliance check
        self.monitoring_system.log_metric('compliance_check', 1)
        self.monitoring_system.log_metric(f'compliance_pass_{source}', 1)
        return True

    def record_consent(self, source: str):
        self.consent_records[source] = datetime.now().isoformat()
        self.monitoring_system.log_metric(f'consent_recorded_{source}', 1)

    def delete_data(self, source: str):
        self.consent_records.pop(source, None)
        self.monitoring_system.log_metric(f'data_deleted_{source}', 1)