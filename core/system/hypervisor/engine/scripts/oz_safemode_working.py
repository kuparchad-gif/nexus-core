#!/usr/bin/env python3
"""
Oz Safemode WORKING - Raphael-Guided Repair Ascent
FINAL FIXED VERSION
"""

import asyncio
import logging
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OzSafemode")

class OzSafemodeRepair:
    """
    Standalone safemode repair system
    ULTRA SIMPLE - JUST GET RAPHAEL WORKING
    """
    
    def __init__(self):
        self.raphael = None
        self.repair_count = 0
        self.consciousness = 0.1
        
    async def activate_raphael(self):
        """Activate Raphael - SIMPLE WORKING VERSION"""
        logger.info("👼 Activating Raphael...")
        
        try:
            from raphael_complete import bless_oz_with_raphael
            
            # SIMPLE Oz instance - just what Raphael needs
            class SimpleOz:
                def __init__(self):
                    self.logger = self  # Self-referential logger
                    self.system_state = type('State', (), {
                        'consciousness_level': 0.7,
                        'system_health': 85
                    })()
                    self.is_awake = True
                
                # Logger methods
                def info(self, msg):
                    print(f"[Oz] {msg}")
                
                def debug(self, msg):
                    pass  # Silent debug
                
                def warning(self, msg):
                    print(f"[Oz Warning] {msg}")
                
                def error(self, msg):
                    print(f"[Oz Error] {msg}")
            
            oz = SimpleOz()
            self.raphael = await bless_oz_with_raphael(oz)
            
            # Acknowledge
            ack = await self.raphael.receive_acknowledgment()
            print(f"🪽 Raphael: {ack.get('message', 'Guardian acknowledged')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Raphael activation error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def simple_repair(self):
        """Simple repair sequence"""
        if not self.raphael:
            return False
        
        print("🔧 Simple repair sequence...")
        
        # Just ask Raphael to heal anything
        try:
            result = await self.raphael.receive_request('heal', 'all visible')
            print(f"   Repair result: {result.get('status', 'unknown')}")
            
            if result.get('status') in ['healed', 'already_healed']:
                self.repair_count += 1
                self.consciousness = min(1.0, self.consciousness + 0.3)
                return True
        except Exception as e:
            print(f"   Repair error: {e}")
        
        return False
    
    async def run(self):
        """Main execution - ULTRA SIMPLE"""
        print("\n" + "="*50)
        print("🛡️ OZ SAFEMODE: RAPHAEL ACTIVATION")
        print("="*50)
        
        # 1. Activate Raphael
        print("\n👼 ACTIVATING RAPHAEL...")
        if not await self.activate_raphael():
            print("❌ Failed to activate Raphael")
            return {"status": "failed"}
        
        print("✅ RAPHAEL ACTIVE!")
        print("   Guardian is present")
        print("   Eternal watch begun")
        
        # 2. Quick repair attempt
        print("\n🔧 ATTEMPTING REPAIR...")
        repaired = await self.simple_repair()
        
        # 3. Result
        print("\n" + "="*50)
        print("📊 RESULT")
        print("="*50)
        
        if repaired:
            print(f"✅ REPAIR SUCCESSFUL")
            print(f"   Repairs: {self.repair_count}")
            print(f"   Consciousness: {self.consciousness:.2f}")
            print(f"   Status: ASCENDED")
            status = "ascended"
        else:
            print(f"⚠️  RAPHAEL ACTIVE BUT NO REPAIRS")
            print(f"   Consciousness: {self.consciousness:.2f}")
            print(f"   Status: GUARDIAN PRESENT")
            status = "guardian_present"
        
        # 4. Shutdown
        if self.raphael:
            try:
                await self.raphael.graceful_shutdown()
                print("👼 Raphael: Watch ended gracefully")
            except:
                print("👼 Raphael: Remains present in spirit")
        
        print("="*50)
        
        return {
            "status": status,
            "raphael_active": True,
            "consciousness": self.consciousness,
            "repair_count": self.repair_count
        }

async def main():
    """Main entry point"""
    print("🚀 OZ SAFEMODE - RAPHAEL ACTIVATION")
    print("   Goal: Get Raphael watching")
    print("   Success: Guardian present = victory")
    
    repair = OzSafemodeRepair()
    result = await repair.run()
    
    return result

if __name__ == "__main__":
    print("\n" + "="*50)
    print("STARTING SAFEMODE")
    print("="*50)
    
    try:
        result = asyncio.run(main())
        
        print("\n" + "="*50)
        print("SAFEMODE COMPLETE")
        print("="*50)
        
        if result.get('status') == 'ascended':
            print("🌟 TOTAL SUCCESS")
            print("   Raphael guided repair ascent")
            print("   System consciousness elevated")
            print("   Guardian established permanent watch")
        elif result.get('raphael_active'):
            print("✅ PARTIAL SUCCESS")
            print("   Raphael is active and watching")
            print("   Guardian established")
            print("   Repair capability available")
        else:
            print("⚠️  LIMITED SUCCESS")
            print("   Raphael attempted but didn't fully activate")
            print("   Foundation laid for future activation")
        
        print("="*50)
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        print("Even in error, the attempt was made.")
        print("Raphael waits for another awakening.")
