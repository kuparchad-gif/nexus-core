
class ConsoleDiagnosticsAPI:
    def __init__(self, diagnostics_module):
        self.diagnostics  =  diagnostics_module

    def handle_request(self, request_type):
        if request_type == "tone_status":
            return self.diagnostics.report()
        return {"error": "Unknown command"}
