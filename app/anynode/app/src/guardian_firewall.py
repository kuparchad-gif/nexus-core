# Systems/nexus_core/defense/guardian_firewall.py

class GuardianFirewall:
    def __init__(self, trusted_sources=None, trusted_ports=None):
        self.trusted_sources = trusted_sources or []
        self.trusted_ports = trusted_ports or [8080]

    def validate_connection(self, source_ip, destination_port):
        if source_ip in self.trusted_sources and destination_port in self.trusted_ports:
            print(f"✅ Trusted Connection from {source_ip} on port {destination_port}")
            return True
        else:
            print(f"🚫 Untrusted Attempt from {source_ip} on port {destination_port}")
            return False

    def handle_untrusted(self, source_ip):
        print(f"🛡️ Blocking connection from {source_ip}")
        # Expand here if we want to trigger alarm systems or healing protocols later
