#!/usr/bin/env python3
"""
Cloud Viren UI Launcher
Launches the Cloud Viren UI locally, connecting to the Modal deployment
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def load_config():
    """Load colony configuration"""
    try:
        config_path = os.path.join('C:/Viren/config', 'colony_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Cloud Viren UI Launcher")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the UI on")
    parser.add_argument("--api-url", type=str, help="Cloud Viren API URL")
    parser.add_argument("--api-key", type=str, help="Cloud Viren API Key")
    
    args = parser.parse_args()
    
    # Get API URL and key from config if not provided
    api_url = args.api_url
    api_key = args.api_key
    
    if not api_url or not api_key:
        config = load_config()
        if config and 'cloud_endpoints' in config and len(config['cloud_endpoints']) > 0:
            endpoint = config['cloud_endpoints'][0]
            api_url = api_url or endpoint.get('url')
            api_key = api_key or endpoint.get('api_key')
    
    if not api_url:
        api_url = "https://viren-cloud--viren-api.modal.run"
        print(f"No API URL provided, using default: {api_url}")
    
    if not api_key:
        print("Warning: No API key provided. Authentication may fail.")
    
    # Import the UI module
    try:
        from cloud_ui import CloudVirenUI
    except ImportError:
        print("Error: Could not import CloudVirenUI. Make sure cloud_ui.py is in the same directory.")
        sys.exit(1)
    
    # Create and start UI
    print(f"Starting Cloud Viren UI, connecting to {api_url}")
    ui = CloudVirenUI(api_url=api_url, api_key=api_key)
    ui.start(port=args.port, share=False, inbrowser=True)

if __name__ == "__main__":
    main()