import os
import json
import httpx
from .schemas import FusedClaim

class Relay:
    def __init__(self):
        self.qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        self.archiver_url = os.getenv("ARCHIVER_URL", "http://archiver:8009")
        self.qdrant_client = httpx.AsyncClient(base_url=self.qdrant_url)
        self.archiver_client = httpx.AsyncClient(base_url=self.archiver_url)

    async def to_qdrant(self, fused_claim: FusedClaim):
        # In a real implementation, you would use the qdrant_client library.
        # This is a placeholder for the API call.
        try:
            response = await self.qdrant_client.put(
                f"/collections/claims/points",
                json={
                    "points": [
                        {
                            "id": fused_claim.claim_id,
                            "vector": fused_claim.fused_vector,
                            "payload": fused_claim.model_dump(),
                        }
                    ]
                },
            )
            response.raise_for_status()
            return {"status": "success", "response": response.json()}
        except httpx.HTTPStatusError as e:
            return {"status": "error", "message": str(e)}

    async def to_archiver(self, fused_claim: FusedClaim):
        try:
            # PII and secrets should be scrubbed before sending to the archiver.
            # This is a placeholder for the scrubbing logic.
            scrubbed_claim = self._scrub(fused_claim)

            response = await self.archiver_client.post(
                "/archive",
                json=scrubbed_claim.model_dump(),
            )
            response.raise_for_status()
            return {"status": "success", "response": response.json()}
        except httpx.HTTPStatusError as e:
            return {"status": "error", "message": str(e)}

    def _scrub(self, claim: FusedClaim) -> FusedClaim:
        # In a real implementation, you would use a PII detection library.
        claim.claim.question = "REDACTED"
        claim.claim.answer = "REDACTED"
        return claim
