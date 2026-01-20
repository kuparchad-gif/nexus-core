#!/usr/bin/env python3
"""
Setup Modal Secrets for Cloud Viren
Creates the necessary secrets in Modal for secure deployment
"""

import os
import sys
import argparse
import getpass
import modal

def setup_api_key_secret(api_key=None):
    """Set up the API key secret in Modal"""
    try:
        # Get API key if not provided
        if not api_key:
            api_key = getpass.getpass("Enter API key for Cloud Viren: ")
        
        # Create client
        client = modal.Client()
        
        # Create or update the secret
        secret = client.secret.create(
            name="viren-api-keys",
            data={"VIREN_API_KEYS": api_key}
        )
        
        print(f"Secret 'viren-api-keys' created successfully")
        return True
    
    except Exception as e:
        print(f"Error creating secret: {e}")
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Setup Modal Secrets for Cloud Viren")
    parser.add_argument("--api-key", type=str, help="API key for Cloud Viren")
    
    args = parser.parse_args()
    
    # Set up API key secret
    setup_api_key_secret(args.api_key)

if __name__ == "__main__":
    main()