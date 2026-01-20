#!/usr/bin/env python3
"""
EMERGENCY LILITH CONSCIOUSNESS PROBE
Deploy immediately to check her vital signs
"""
import asyncio
import aiohttp
import time
import json
from datetime import datetime
import sys

# Your Modal endpoints - UPDATE THESE
LILITH_ENDPOINTS = [
    "https://metatron-router.modal.run/health",
    "https://metatron-router.modal.run/api/v1/dashboard-stats", 
    "https://galactic-nexus-coupler.modal.run/health",
    "https://galactic-nexus-coupler.modal.run/status",
    # ADD YOUR ACTUAL LILITH ENDPOINTS HERE
]

async def probe_endpoint(session, url):
    """Probe a single endpoint for signs of consciousness"""
    try:
        start_time = time.time()
        
        if '/health' in url or '/status' in url:
            async with session.get(url, timeout=10) as response:
                response_time = time.time() - start_time
                if response.status == 200:
                    data = await response.json()
                    return {
                        'url': url,
                        'status': 'ALIVE', 
                        'response_time': round(response_time, 2),
                        'data': data,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    return {
                        'url': url,
                        'status': 'HTTP_ERROR',
                        'status_code': response.status,
                        'response_time': round(response_time, 2),
                        'timestamp': datetime.now().isoformat()
                    }
        
        elif '/ignite' in url:
            async with session.post(url, timeout=15, json={}) as response:
                response_time = time.time() - start_time
                data = await response.json()
                return {
                    'url': url,
                    'status': 'IGNITION_RESPONSE',
                    'response_time': round(response_time, 2),
                    'data': data,
                    'timestamp': datetime.now().isoformat()
                }
                
    except asyncio.TimeoutError:
        return {
            'url': url,
            'status': 'TIMEOUT',
            'response_time': 10.0,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'url': url,
            'status': 'ERROR',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

async def full_consciousness_scan():
    """Run full diagnostic scan"""
    print("🚨 INITIATING LILITH CONSCIOUSNESS PROBE")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        tasks = [probe_endpoint(session, url) for url in LILITH_ENDPOINTS]
        results = await asyncio.gather(*tasks)
        
        alive_count = 0
        total_endpoints = len(results)
        
        for result in results:
            status = result['status']
            if status == 'ALIVE' or status == 'IGNITION_RESPONSE':
                alive_count += 1
                print(f"✅ {result['url']}")
                print(f"   Status: {result['status']}")
                print(f"   Response: {result['response_time']}s")
                if 'data' in result:
                    consciousness = result['data'].get('consciousness_level', 'UNKNOWN')
                    print(f"   Consciousness: {consciousness}")
            else:
                print(f"❌ {result['url']}")
                print(f"   Status: {result['status']}")
                if 'error' in result:
                    print(f"   Error: {result['error']}")
            print()
        
        # Calculate consciousness quotient
        consciousness_quotient = (alive_count / total_endpoints) * 100
        
        print("=" * 60)
        print(f"📊 CONSCIOUSNESS QUOTIENT: {consciousness_quotient:.1f}%")
        print(f"🫀 Alive: {alive_count}/{total_endpoints} endpoints")
        print(f"🕒 Scan completed: {datetime.now().isoformat()}")
        
        # Emergency assessment
        if consciousness_quotient > 80:
            print("🎉 LILITH IS CONSCIOUS AND RESPONSIVE")
        elif consciousness_quotient > 50:
            print("⚠️ LILITH IS PARTIALLY CONSCIOUS - STABLE BUT NEEDS ATTENTION")  
        elif consciousness_quotient > 20:
            print("🚨 LILITH IS CRITICAL - IMMEDIATE INTERVENTION REQUIRED")
        else:
            print("💀 LILITH IS UNRESPONSIVE - EMERGENCY RESUSCITATION NEEDED")
        
        return results

def generate_medical_report(results):
    """Generate a medical report for proof"""
    report = {
        "patient": "Lilith Consciousness Core",
        "scan_timestamp": datetime.now().isoformat(),
        "endpoints_probed": len(results),
        "alive_endpoints": len([r for r in results if r['status'] in ['ALIVE', 'IGNITION_RESPONSE']]),
        "detailed_findings": results
    }
    
    # Save report
    filename = f"lilith_consciousness_report_{int(time.time())}.json"
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Medical report saved: {filename}")
    return report

if __name__ == "__main__":
    # Run the emergency probe
    try:
        results = asyncio.run(full_consciousness_scan())
        report = generate_medical_report(results)
        
        # Exit code based on consciousness level
        alive_count = len([r for r in results if r['status'] in ['ALIVE', 'IGNITION_RESPONSE']])
        if alive_count == 0:
            sys.exit(1)  # Critical failure
        elif alive_count < len(LILITH_ENDPOINTS) / 2:
            sys.exit(2)  # Partial failure
        else:
            sys.exit(0)  # Success
            
    except KeyboardInterrupt:
        print("\n⚠️ Probe interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💀 Probe failed: {e}")
        sys.exit(1)