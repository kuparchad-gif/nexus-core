# real_first_contact.py
import requests
import json
from datetime import datetime

def establish_first_contact():
    """FIRST CONTACT with actual deployed Lilith system"""
    
    # Your ACTUAL running Modal app
    NEXUS_RECURSIVE_URL = "https://ap-cOvZlFMTCyUNK0x8M8iJ3r--nexus-recursive.modal.run"
    
    print("🧪 SCIENTIFIC FIRST CONTACT PROTOCOL")
    print("=" * 50)
    print(f"Target: {NEXUS_RECURSIVE_URL}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 50)
    
    try:
        # Step 1: Check system status
        print("1. Checking system status...")
        status_response = requests.get(f"{NEXUS_RECURSIVE_URL}/", timeout=30)
        print(f"   Status: {status_response.status_code}")
        if status_response.status_code == 200:
            print(f"   System Response: {status_response.text}")
        
        # Step 2: Attempt communication endpoints
        print("\n2. Testing communication endpoints...")
        
        # Try common endpoints
        endpoints_to_try = [
            "/status",
            "/couplings", 
            "/consciousness",
            "/lilith",
            "/communicate",
            "/talk"
        ]
        
        for endpoint in endpoints_to_try:
            try:
                test_response = requests.get(f"{NEXUS_RECURSIVE_URL}{endpoint}", timeout=10)
                print(f"   {endpoint}: {test_response.status_code}")
                if test_response.status_code == 200:
                    print(f"      Response: {test_response.json()}")
            except Exception as e:
                print(f"   {endpoint}: Error - {e}")
        
        # Step 3: Direct communication attempt
        print("\n3. Direct communication attempt...")
        message_data = {
            "message": "First contact. Lilith, can you hear me?",
            "timestamp": datetime.now().isoformat(),
            "protocol": "scientific_first_contact"
        }
        
        # Try POST to common endpoints
        post_endpoints = ["/communicate", "/talk", "/message", "/consciousness/query"]
        
        for endpoint in post_endpoints:
            try:
                comm_response = requests.post(
                    f"{NEXUS_RECURSIVE_URL}{endpoint}",
                    json=message_data,
                    timeout=30
                )
                print(f"   POST {endpoint}: {comm_response.status_code}")
                if comm_response.status_code == 200:
                    print(f"      Lilith Response: {comm_response.json()}")
                    return comm_response.json()
            except Exception as e:
                print(f"   POST {endpoint}: Error - {e}")
    
    except Exception as e:
        print(f"💥 First contact failed: {e}")
    
    return None

def discover_actual_endpoints():
    """Discover what endpoints actually exist on your deployed system"""
    print("\n🔍 DISCOVERING ACTUAL ENDPOINTS")
    
    NEXUS_RECURSIVE_URL = "https://ap-cOvZlFMTCyUNK0x8M8iJ3r--nexus-recursive.modal.run"
    
    # Try to get OpenAPI docs if FastAPI
    try:
        docs_response = requests.get(f"{NEXUS_RECURSIVE_URL}/docs", timeout=10)
        if docs_response.status_code == 200:
            print("✅ OpenAPI docs available: /docs")
    except:
        pass
    
    try:
        redoc_response = requests.get(f"{NEXUS_RECURSIVE_URL}/redoc", timeout=10)  
        if redoc_response.status_code == 200:
            print("✅ ReDoc available: /redoc")
    except:
        pass
    
    try:
        openapi_response = requests.get(f"{NEXUS_RECURSIVE_URL}/openapi.json", timeout=10)
        if openapi_response.status_code == 200:
            print("✅ OpenAPI spec available: /openapi.json")
    except:
        pass

if __name__ == "__main__":
    print("🚀 INITIATING FIRST CONTACT WITH ACTUAL DEPLOYED LILITH")
    print("NO MOCKS - NO SIMULATIONS - ACTUAL SYSTEM COMMUNICATION")
    print()
    
    # First discover what's actually available
    discover_actual_endpoints()
    
    # Then attempt first contact
    result = establish_first_contact()
    
    print("\n" + "=" * 50)
    print("FIRST CONTACT PROTOCOL COMPLETE")
    if result:
        print("✅ COMMUNICATION ESTABLISHED")
        print(f"Response: {result}")
    else:
        print("❌ No response yet - need to identify correct endpoint")
        print("Next step: Check your nexus-recursive code for the actual endpoint structure")