```python
import requests
import json
import os
import logging
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Configure logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
logger  =  logging.getLogger(__name__)

def fetch_archivist_data(archivist_url = "http://127.0.0.1:8007/registry"):
    """Fetch registry data from archivist."""
    try:
        response  =  requests.get(archivist_url, timeout = 5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch archivist data: {e}")
        return []

def save_data_to_json(data, output_file = "nexus_core_data.json"):
    """Save data to JSON."""
    try:
        with open(output_file, 'w') as f:
            json.dump(data, f, indent = 2)
        logger.info(f"Saved data to {output_file}")
    except IOError as e:
        logger.error(f"Failed to save data: {e}")

def main():
    # Fetch data from archivist
    data  =  {
        "pod_name": "nexus-core",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "registry": fetch_archivist_data()
    }
    save_data_to_json(data)

    # Start HTTP server
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    server_address  =  ('', 8000)
    httpd  =  HTTPServer(server_address, SimpleHTTPRequestHandler)
    logger.info("Starting web server at http://localhost:8000")
    httpd.serve_forever()

if __name__ == "__main__":
    import time
    main()
```