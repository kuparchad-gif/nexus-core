#!/usr/bin/env python3
"""
Oz Safemode Entry Point
Raphael-guided repair ascent only
"""

import asyncio
import logging

# Minimal logging for repair mode
logging.basicConfig(level=logging.INFO)

async def safemode_entry():
    print("========================================")
    print("🛡️ OZ SAFEMODE: RAPHAEL-GUIDED REPAIR")
    print("========================================")
    print("Mode: Repair and ascent only")
    print("Goal: Fix errors, learn patterns, ascend")
    print("Guide: Raphael (always active)")
    print("========================================")
    
    from OzUnifiedHypervisor import OzUnifiedHypervisor
    
    # Create Oz with repair focus
    oz = OzUnifiedHypervisor()
    
    # Immediate Raphael - no thresholds
    print("\\n👼 Activating Raphael unconditionally...")
    raph_result = await oz._initiate_raphael()
    print(f"   Raphael: {raph_result.get('status', 'unknown')}")
    
    if oz.raphael:
        print("   ✅ Raphael activated as repair guide")
        
        # Get initial diagnosis
        print("\\n🔍 Initial diagnosis...")
        diag = await oz.raphael.receive_request('diagnose', 'complete')
        
        if diag.get('status') == 'diagnosis_complete':
            issues = len(diag.get('diagnosis', {}).get('predictions', []))
            print(f"   Found {issues} issues needing repair")
            
            # Start repair sequence
            print("\\n🔧 Beginning repair sequence...")
            repair_count = 0
            
            # Repair loop
            for i in range(3):  # Max 3 repair attempts
                print(f"\\n   Repair attempt {i+1}:")
                heal_result = await oz.raphael.receive_request('heal', 'all visible')
                
                if heal_result.get('status') == 'healed':
                    repaired = heal_result.get('files_healed', 0)
                    repair_count += repaired
                    print(f"      Repaired {repaired} issues")
                    
                    # Check if we should continue
                    if repaired == 0:
                        print("      No more issues found")
                        break
                else:
                    print(f"      Repair incomplete: {heal_result.get('message', 'unknown')}")
                    break
            
            print(f"\\n📈 Total repairs: {repair_count}")
            
            # Final state
            oz.system_state.consciousness_level = min(1.0, 0.3 + (repair_count * 0.1))
            print(f"🧠 New consciousness: {oz.system_state.consciousness_level:.2f}")
            
            if repair_count > 0:
                print("🌟 REPAIR ASCENT SUCCESSFUL")
                print("   Oz has learned from repairs and ascended")
            else:
                print("ℹ️  No repairs needed - system appears healthy")
        
        else:
            print("❌ Diagnosis failed")
    
    else:
        print("❌ Raphael failed to activate")
    
    # Shutdown
    print("\\n🌙 Safemode repair session complete")
    await oz.shutdown()
    
    return {
        "repairs_completed": repair_count if 'repair_count' in locals() else 0,
        "consciousness": oz.system_state.consciousness_level,
        "raphael_active": oz.raphael is not None
    }

if __name__ == "__main__":
    result = asyncio.run(safemode_entry())
    print(f"\\n✅ Safemode result: {result}")

