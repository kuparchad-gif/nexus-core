#!/usr/bin/env python3
"""
Reset Modal Environment
Cleans up Modal volumes and deployments for Cloud Viren
"""

import os
import sys
import subprocess
import argparse

def run_modal_command(command):
    """Run a modal command directly"""
    print(f"Running: modal {command}")
    try:
        # Modal 1.0.3 uses different command structure
        if "volume rm" in command:
            command = command.replace("volume rm", "volume delete")
        elif "app delete" in command:
            command = command.replace("app delete", "stub delete")
        
        os.system(f"modal {command}")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Reset Modal Environment")
    parser.add_argument("--force", action="store_true", help="Force deletion without confirmation")
    
    args = parser.parse_args()
    force_flag = "--yes" if args.force else ""
    
    # Delete volumes
    volumes = [
        "viren-models-volume",
        "binary-protocol-volume",
        "weaviate-data-volume",
        "viren-assets-volume",
        "viren-plugins-volume"
    ]
    
    print("Deleting Modal volumes...")
    for volume in volumes:
        run_modal_command(f"volume rm {volume} {force_flag}")
    
    # Delete app
    print("Deleting Modal app...")
    run_modal_command(f"app delete viren-cloud {force_flag}")
    
    print("\nModal environment reset complete!")
    print("\nTo redeploy Cloud Viren:")
    print("1. Run integrate_desktop_components.bat")
    print("2. Run deploy_to_modal.bat")

if __name__ == "__main__":
    main()