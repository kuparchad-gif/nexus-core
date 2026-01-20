import requests
import json

ARCHIVER_URL = "http://localhost:9020"

def run_test():
    """
    Tests the /crawl endpoint of the Aethereal Archiver.
    """
    try:
        print(f"--- Calling Archiver Health Endpoint at {ARCHIVER_URL}/health ---")
        health_res = requests.get(f"{ARCHIVER_URL}/health", timeout=10)
        health_res.raise_for_status()
        print("Health check successful:")
        print(json.dumps(health_res.json(), indent=2))

        print(f"\n--- Calling Archiver Crawl Endpoint at {ARCHIVER_URL}/crawl ---")
        crawl_payload = {
            "which": ["system", "process"]
        }
        crawl_res = requests.post(
            f"{ARCHIVER_URL}/crawl",
            json=crawl_payload,
            timeout=30
        )
        crawl_res.raise_for_status()
        print("Crawl request successful:")
        print(json.dumps(crawl_res.json(), indent=2))

    except requests.exceptions.RequestException as e:
        print(f"\n--- ERROR ---")
        print(f"An error occurred while making a request: {e}")
        print("Please check that the Archiver service is running and accessible.")

if __name__ == "__main__":
    run_test()
