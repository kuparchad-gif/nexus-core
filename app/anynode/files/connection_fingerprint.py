from fastapi import Request
from Utilities.security.fingerprint_utils import extract_fingerprint_data

class ConnectionFingerprint:
    def __init__(self):
        pass

    async def collect_fingerprint(self, request: Request) -> dict:
        client_host  =  request.client.host if request.client else "unknown"
        user_agent  =  request.headers.get("user-agent", "unknown")
        forwarded_for  =  request.headers.get("x-forwarded-for", None)
        tls_version  =  request.scope.get("tls_version", "unknown")

        fingerprint  =  {
            "ip_address": forwarded_for if forwarded_for else client_host,
            "user_agent": user_agent,
            "tls_version": tls_version,
        }

        fingerprint.update(await extract_fingerprint_data(fingerprint))
        return fingerprint
