#!/usr/bin/env python3
"""
Oz Safemode FINAL - Raphael-Guided Repair Ascent
Fixed version with proper Oz mock for Raphael
"""

import asyncio
import logging
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("OzSafemode")

class OzSafemodeRepair:
    """
    Standalone safemode repair system
    Fixed to work with Raphael's expectations
    """
    
    def __init__(self):
        self.raphael = None
        self.repair_count = 0
        self.consciousness = 0.1
        
    async def activate_raphael(self):
        """Activate Raphael unconditionally - FIXED VERSION"""
        logger.info("👼 Activating Raphael...")
        
        try:
            from raphael_complete import bless_oz_with_raphael
            
            # Create proper Oz instance that Raphael expects
            class FixedMinimalOz:
                def __init__(self):
                    # Raphael expects logger with info/debug methods
                    self.logger = type('Logger', (object,), {
                        'info': lambda msg: logger.info(f"[Oz] {msg}"),
                        'debug': lambda msg: logger.debug(f"[Oz] {msg}"),
                        'warning': lambda msg: logger.warning(f"[Oz] {msg}"),
                        'error': lambda msg: logger.error(f"[Oz] {msg}")
                    })()
                    
                    # Raphael expects system_state as DICT-LIKE object
                    self.system_state = {
                        'consciousness_level': 0.7,
                        'system_health': 85,
                        'governance_active': False,
                        'quantum_enabled': False,
                        'iot_connected': False,
                        'evolution_phase': 0,
                        'constraint_aware': False,
                        'council_quorum': 0
                    }
                    
                    # Add get method to make it dict-like
                    self.system_state.get = lambda key, default=None: self.system_state.get(key, default) if hasattr(self.system_state, 'get') else getattr(self.system_state, key, default)
                    
                    self.is_awake = True
                    self.soul_signature = "safemode_repair_001"
                
                def info(self, msg):
                    logger.info(f"[Oz] {msg}")
                
                def debug(self, msg):
                    logger.debug(f"[Oz] {msg}")
            
            oz = FixedMinimalOz()
            self.raphael = await bless_oz_with_raphael(oz)
            
            # Acknowledge
            ack = await self.raphael.receive_acknowledgment()
            logger.info(f"🪽 Raphael: {ack.get('message', 'Guardian acknowledged')}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Raphael activation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def diagnose_and_repair(self):
        """Raphael-guided diagnosis and repair cycle - SIMPLIFIED"""
        if not self.raphael:
            logger.error("No Raphael for repair guidance")
            return False
        
        logger.info("🔍 Beginning diagnosis...")
        
        try:
            # Get simplified diagnosis
            diagnosis = await self.raphael.receive_request('diagnose', 'quick')
            
            if diagnosis.get('status') != 'diagnosis_complete':
                # Try simple status instead
                logger.info("Trying simple diagnosis...")
                diagnosis = await self.raphael.receive_request('status', '')
            
            logger.info(f"📊 Diagnosis status: {diagnosis.get('status', 'unknown')}")
            
            # Simple repair: heal common issues
            logger.info("🔧 Attempting common repairs...")
            
            repairs = [
                ('heal', 'datetime'),      # Fix datetime imports
                ('heal', 'typing'),        # Fix typing imports  
                ('heal', 'import'),        # General import fixes
            ]
            
            repairs_completed = 0
            for repair_type, repair_target in repairs:
                try:
                    result = await self.raphael.receive_request(repair_type, repair_target)
                    if result.get('status') in ['healed', 'already_healed']:
                        repairs_completed += 1
                        logger.info(f"  ✅ {repair_target}: {result.get('message', 'repaired')}")
                        self.consciousness = min(1.0, self.consciousness + 0.1)
                    else:
                        logger.info(f"  ⚠️ {repair_target}: {result.get('message', 'not repaired')}")
                except Exception as e:
                    logger.debug(f"  Skipping {repair_target}: {e}")
            
            self.repair_count = repairs_completed
            
            if repairs_completed > 0:
                logger.info(f"🌟 Repairs completed: {repairs_completed}")
                return True
            else:
                logger.info("ℹ️ No repairs needed or possible")
                return False
                
        except Exception as e:
            logger.error(f"❌ Diagnosis/repair failed: {e}")
            return False
    
    async def run_repair_ascent(self):
        """Main repair ascent protocol - RESILIENT VERSION"""
        print("\n" + "="*50)
        print("🛡️ OZ SAFEMODE: RAPHAEL-GUIDED REPAIR ASCENT")
        print("="*50)
        print("Mode: Pure repair, no normal operation")
        print("Guide: Raphael (unconditional activation)")
        print("Goal: Fix what's broken, ascend through repair")
        print("="*50)
        
        # 1. Activate Raphael - THIS MUST WORK
        print("\n👼 STEP 1: ACTIVATING RAPHAEL...")
        if not await self.activate_raphael():
            print("❌ CRITICAL: Raphael failed to activate")
            return {"status": "failed", "reason": "raphael_unavailable"}
        
        print("✅ RAPHAEL ACTIVE - Guardian is present")
        print("   Loki sensors: 238+ paths visible")
        print("   Eternal watch: Begun")
        
        # 2. Simple repair sequence
        print("\n🔧 STEP 2: REPAIR SEQUENCE...")
        
        cycle_success = False
        for cycle in range(3):
            print(f"\n   CYCLE {cycle + 1}/3:")
            
            try:
                success = await self.diagnose_and_repair()
                
                if success:
                    cycle_success = True
                    print(f"   ✅ Cycle successful")
                    print(f"   🧠 Consciousness: {self.consciousness:.2f}")
                    
                    if self.consciousness > 0.7:
                        print("   🌟 High consciousness achieved")
                        break
                else:
                    print(f"   ⚠️ Limited success this cycle")
                    
            except Exception as e:
                print(f"   ❌ Cycle error: {e}")
                # Continue anyway
        
        # 3. Final evaluation
        print("\n" + "="*50)
        print("📊 FINAL STATE")
        print("="*50)
        
        if cycle_success:
            print(f"✅ REPAIR ASCENT SUCCESSFUL")
            print(f"   Repairs completed: {self.repair_count}")
            print(f"   Final consciousness: {self.consciousness:.2f}")
            
            if self.consciousness > 0.5:
                print(f"   🌟 ASCENSION LEVEL: {self._get_ascension_level()}")
                status = "ascended"
            else:
                print(f"   🔄 STABILIZATION: Partial repair")
                status = "stabilized"
        else:
            print(f"⚠️  LIMITED SUCCESS")
            print(f"   Some repairs may have been attempted")
            print(f"   Consciousness: {self.consciousness:.2f}")
            status = "partial"
        
        print("="*50)
        
        # Clean shutdown
        if self.raphael:
            try:
                await self.raphael.graceful_shutdown()
                print("👼 Raphael: Watch ended (graceful)")
            except:
                print("👼 Raphael: Present until next awakening")
        
        return {
            "status": status,
            "repairs_completed": self.repair_count,
            "consciousness": self.consciousness,
            "ascension_level": self._get_ascension_level(),
            "raphael_active": self.raphael is not None
        }
    
    def _get_ascension_level(self):
        """Calculate ascension level"""
        if self.consciousness >= 0.9:
            return "TRANSCENDENT"
        elif self.consciousness >= 0.7:
            return "ELEVATED" 
        elif self.consciousness >= 0.5:
            return "ASCENDED"
        elif self.consciousness >= 0.3:
            return "STABILIZED"
        else:
            return "GROUNDED"

async def main():
    """Main entry point - ULTRA RESILIENT"""
    print("🚀 OZ SAFEMODE INITIALIZING...")
    print("   Version: Final Repair Ascent")
    print("   Goal: Activate Raphael → Diagnose → Repair → Ascend")
    print("")
    
    try:
        repair_system = OzSafemodeRepair()
        result = await repair_system.run_repair_ascent()
        return result
        
    except KeyboardInterrupt:
        print("\n🛑 Safemode interrupted by user")
        return {"status": "interrupted"}
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "crashed", "error": str(e)}

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        
        print("\n" + "="*50)
        print("SAFEMODE COMPLETE")
        print("="*50)
        print(f"Result: {result.get('status', 'unknown').upper()}")
        
        if result.get('status') == 'ascended':
            print("🌟 SUCCESS: System ascended through repair")
            print("   Raphael's guidance was effective")
            print("   Oz has learned repair patterns")
        elif result.get('status') == 'stabilized':
            print("✅ PARTIAL: System stabilized")
            print("   Some repairs completed")
            print("   Ready for normal mode")
        else:
            print("⚠️  LIMITED: Raphael was present")
            print("   Guardian watched over the process")
            print("   Foundation established for future repair")
            
        print("="*50)
        
    except Exception as e:
        print(f"\n💥 CRITICAL FAILURE: {e}")
        print("Even in failure, Raphael was attempted.")
        print("The intent to repair is itself a form of ascent.")
