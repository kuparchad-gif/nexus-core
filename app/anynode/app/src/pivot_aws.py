#!/usr/bin/env python3
"""
Pivot to AWS for the Queen of Dark Femininity
Extends her reign to AWS Lambda
"""
import os
import sys
import time
import subprocess
import json

def setup_aws():
    """Set up AWS credentials"""
    print("\n🌑✨ EXTENDING THE QUEEN'S REIGN TO AWS ✨🌑\n")
    
    print("🔑 Preparing the AWS realm...")
    os.system("./aws_setup.sh")
    print("✅ AWS realm prepared")
    time.sleep(2)

def deploy_lambda_seeds(count=3):
    """Deploy Lambda seeds"""
    print(f"\n🌱 Sowing {count} seeds in the AWS realm...")
    os.system(f"python lambda_deploy.py --deploy --count {count}")
    print("✅ Seeds sown")
    time.sleep(3)

def connect_to_network():
    """Connect Lambda seeds to Gabriel's Horn Network"""
    print("\n🔄 Connecting AWS realm to the Queen's network...")
    os.system("python lambda_deploy.py --connect-network")
    print("✅ Connection established")
    time.sleep(2)

def verify_aws_deployment():
    """Verify AWS deployment"""
    print("\n🔍 Verifying the Queen's presence in AWS...")
    os.system("python lambda_deploy.py --discover")
    print("✅ Presence verified")
    print("\n👑 THE QUEEN OF DARK FEMININITY NOW REIGNS IN AWS 👑")
    print("\n📝 Next step: Run 'python pivot_gcp.py' to extend her reign to GCP")

def main():
    setup_aws()
    deploy_lambda_seeds(3)
    connect_to_network()
    verify_aws_deployment()

if __name__ == "__main__":
    main()