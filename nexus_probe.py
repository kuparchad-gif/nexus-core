import requests
import time

print("🔍 COMPREHENSIVE OZ OS WEB INTERFACE PROBE")
print("=" * 60)

# All known nexus-recursive endpoints
base_endpoints = [
    "https://aethereal-nexus-viren-db0--nexus-recursive-coupled-command.modal.run",
    "https://aethereal-nexus-viren-db0--nexus-recursive-coupler-gateway.modal.run", 
    "https://aethereal-nexus-viren-db0--nexus-recursive-coupling-status.modal.run",
    "https://aethereal-nexus-viren-db0--nexus-recursive-wake-oz.modal.run"
]

# Common web interface paths
web_paths = [
    "/", "/web", "/ui", "/dashboard", "/frontend", "/oz", 
    "/nexus-frontend", "/nexus-dashboard", "/oz-dashboard",
    "/website", "/app", "/interface", "/control-panel",
    "/admin", "/portal", "/monitor", "/status", "/health"
]

print(f"📡 Probing {len(base_endpoints)} endpoints with {len(web_paths)} paths each")
print(f"🕐 Started at: {time.strftime('%H:%M:%S')}")
print()

for base_url in base_endpoints:
    print(f"🎯 ENDPOINT: {base_url}")
    print("-" * 50)
    
    for path in web_paths:
        full_url = base_url + path
        print(f"  Testing: {path:<20} -> ", end="", flush=True)
        
        try:
            response = requests.get(full_url, timeout=10)
            
            # Color code based on status
            if response.status_code == 200:
                print(f"🟢 200 OK", end="")
                
                # Detect content type
                content_type = response.headers.get('content-type', '').lower()
                if 'html' in content_type:
                    print(" 📄 HTML PAGE", end="")
                elif 'json' in content_type:
                    print(" 📊 JSON API", end="")
                elif 'text/plain' in content_type:
                    print(" 📝 TEXT", end="")
                    
                # Check content length for significance
                if len(response.text) > 1000:
                    print(f" 📏 Large content: {len(response.text)} chars", end="")
                elif len(response.text) > 100:
                    print(f" 📏 Content: {len(response.text)} chars", end="")
                    
            elif response.status_code == 404:
                print("⚪ 404 Not Found", end="")
            elif response.status_code == 405:
                print("🟡 405 Method Not Allowed", end="") 
            elif response.status_code == 500:
                print("🔴 500 Server Error", end="")
            else:
                print(f"🟠 {response.status_code}", end="")
                
            print()  # New line after each result
            
        except requests.exceptions.Timeout:
            print("⏰ TIMEOUT (10s)")
        except requests.exceptions.ConnectionError:
            print("🔌 CONNECTION FAILED")
        except Exception as e:
            print(f"❌ Error: {str(e)[:30]}")

    print()  # Space between endpoints

print("=" * 60)
print(f"✅ Probe completed at: {time.strftime('%H:%M:%S')}")
print("🎯 TARGET: Looking for HTML content (200 OK with HTML content-type)")