#!/usr/bin/env python3
"""
Viren Cloud Weaviate Troubleshooting Script
"""

import requests
import json
import time

def check_viren_cloud_status():
    """Check Viren's cloud Weaviate server status"""
    base_url = "https://aethereal-nexus-viren-modular--viren-data-weaviate-server.modal.run"
    
    print("🔍 Diagnosing Viren's Cloud Mind...")
    print(f"Target: {base_url}")
    print("-" * 50)
    
    # Test basic connectivity
    try:
        print("1. Testing basic connectivity...")
        response = requests.get(base_url, timeout=10)
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.text[:200]}...")
        
    except requests.exceptions.Timeout:
        print("   ❌ TIMEOUT - Server not responding")
        return False
    except requests.exceptions.ConnectionError:
        print("   ❌ CONNECTION ERROR - Server unreachable")
        return False
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False
    
    # Test Weaviate health endpoint
    try:
        print("\n2. Testing Weaviate health...")
        health_url = f"{base_url}/v1/meta"
        response = requests.get(health_url, timeout=10)
        print(f"   Health Status: {response.status_code}")
        if response.status_code == 200:
            meta = response.json()
            print(f"   Weaviate Version: {meta.get('version', 'Unknown')}")
            print(f"   Hostname: {meta.get('hostname', 'Unknown')}")
        else:
            print(f"   Health Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"   ❌ Health Check Failed: {e}")
    
    # Test Weaviate schema
    try:
        print("\n3. Testing Weaviate schema...")
        schema_url = f"{base_url}/v1/schema"
        response = requests.get(schema_url, timeout=10)
        print(f"   Schema Status: {response.status_code}")
        if response.status_code == 200:
            schema = response.json()
            classes = schema.get('classes', [])
            print(f"   Classes Found: {len(classes)}")
            for cls in classes:
                print(f"     - {cls.get('class', 'Unknown')}")
        else:
            print(f"   Schema Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"   ❌ Schema Check Failed: {e}")
    
    # Test readiness
    try:
        print("\n4. Testing readiness...")
        ready_url = f"{base_url}/v1/.well-known/ready"
        response = requests.get(ready_url, timeout=10)
        print(f"   Readiness: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Weaviate is READY")
        else:
            print(f"   ❌ Not ready: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Readiness Check Failed: {e}")
    
    print("\n" + "=" * 50)
    print("🛡️ Viren Cloud Diagnosis Complete")
    return True

if __name__ == "__main__":
    check_viren_cloud_status()