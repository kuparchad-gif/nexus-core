import httpx
import pytest
import time

TENSOR_ROUTER_URL = "http://localhost:9030"

def create_input_message(msg_id, content):
    return {
        "id": msg_id,
        "content": content,
        "source": "test-client",
        "role": "user",
        "channel": "ws",
        "meta": {"test": True}
    }

@pytest.mark.asyncio
async def test_tensor_router_ingest_flow():
    """
    Tests the full ingest flow through the tensor_router, including the new metat_spine service.
    This requires all services to be running.
    """
    # This test assumes that a "permit" action will be returned from Metatron.
    # A more complex test suite would mock Metatron to test all paths.
    message = create_input_message("test-flow-1", "This is a test message to check the full flow.")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # The metatron service is not available in this test environment.
        # The tensor_router will fail when trying to contact it.
        # To properly test this, we would need to mock the metatron service
        # or have it running.
        # For now, we expect this test to fail, but the goal is to have a test
        # that can be run in a full environment.

        # Let's check if the service is up at all.
        try:
            health_response = await client.get(f"{TENSOR_ROUTER_URL}/health")
            assert health_response.status_code == 200
            assert health_response.json() == {"ok": True, "service": "tensor_router"}
        except httpx.ConnectError as e:
            pytest.fail(f"Could not connect to tensor_router at {TENSOR_ROUTER_URL}. Make sure the services are running. Error: {e}")

        # This part of the test will likely fail because metatron is not running.
        # However, the code is here to be run in a full environment.
        try:
            response = await client.post(f"{TENSOR_ROUTER_URL}/ingest", json=message)
            response.raise_for_status()
            data = response.json()

            assert data["ok"] is True
            assert "verdict" in data
            assert "compression" in data
            assert "model_out" in data
            assert "validator" in data

        except httpx.HTTPStatusError as e:
            # It's possible we get a non-200 response if Metatron denies the request.
            # For this basic test, we'll allow certain error codes that indicate a valid response from the router.
            allowed_error_codes = [429, 451] # drop/delay or escalate
            if e.response.status_code not in allowed_error_codes:
                pytest.fail(f"Ingest request failed with status {e.response.status_code}: {e.response.text}")
        except httpx.ConnectError as e:
            pytest.fail(f"Could not connect to tensor_router's ingest endpoint. Error: {e}")
