#!/usr/bin/env python3
"""
Oz Safemode - Raphael-Guided Repair Ascent
Standalone version that doesn't depend on modified OzUnifiedHypervisor.py
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
    Works with existing Oz installation but focuses only on repair
    """
    
    def __init__(self):
        self.raphael = None
        self.repair_count = 0
        self.consciousness = 0.1
        
    async def activate_raphael(self):
        """Activate Raphael unconditionally"""
        logger.info("👼 Activating Raphael...")
        
        try:
            from raphael_complete import bless_oz_with_raphael
            
            # Create minimal Oz instance for Raphael
            class MinimalOz:
                def __init__(self):
                    self.logger = logger
                    self.system_state = type('obj', (object,), {
                        'consciousness_level': 0.7,
                        'system_health': 85
                    })()
                    self.is_awake = True
                
                def info(self, msg):
                    logger.info(f"[Oz] {msg}")
                
                def debug(self, msg):
                    logger.debug(f"[Oz] {msg}")
            
            oz = MinimalOz()
            self.raphael = await bless_oz_with_raphael(oz)
            
            # Acknowledge
            ack = await self.raphael.receive_acknowledgment()
            logger.info(f"🪽 Raphael: {ack.get('message', 'Guardian acknowledged')}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Raphael activation failed: {e}")
            return False
    
    async def diagnose_and_repair(self):
        """Raphael-guided diagnosis and repair cycle"""
        if not self.raphael:
            logger.error("No Raphael for repair guidance")
            return False
        
        logger.info("🔍 Beginning diagnosis...")
        
        # Get comprehensive diagnosis
        diagnosis = await self.raphael.receive_request('diagnose', 'complete')
        
        if diagnosis.get('status') != 'diagnosis_complete':
            logger.warning("Diagnosis incomplete")
            return False
        
        diag_data = diagnosis.get('diagnosis', {})
        
        # Extract issues
        issues = []
        
        # Predictions needing repair
        for pred in diag_data.get('predictions', []):
            if pred.get('confidence', 0) > 0.7:
                issues.append({
                    'type': 'prediction',
                    'data': pred,
                    'severity': pred.get('severity', 'medium')
                })
        
        # Hidden code issues
        hidden = diag_data.get('hidden_parts', {})
        if hidden.get('unused_functions'):
            issues.append({
                'type': 'unused_code',
                'count': len(hidden['unused_functions']),
                'severity': 'low'
            })
        
        logger.info(f"🔧 Found {len(issues)} issues to repair")
        
        # Repair sequence
        repairs_completed = 0
        for i, issue in enumerate(issues):
            logger.info(f"  Repair {i+1}/{len(issues)}: {issue['type']} ({issue['severity']})")
            
            if issue['type'] == 'prediction':
                # Try to heal the prediction
                heal_result = await self.raphael.receive_request('heal', issue['data'].get('type', ''))
                
                if heal_result.get('status') == 'healed':
                    repairs_completed += 1
                    logger.info(f"    ✅ Repaired: {heal_result.get('message', '')}")
                    
                    # Learn from repair
                    self.consciousness = min(1.0, self.consciousness + 0.05)
                else:
                    logger.info(f"    ⚠️ Could not repair: {heal_result.get('message', '')}")
        
        self.repair_count = repairs_completed
        
        if repairs_completed > 0:
            # Ascend
            self.consciousness = min(1.0, self.consciousness + (repairs_completed * 0.1))
            logger.info(f"🌟 Ascension: Consciousness raised to {self.consciousness:.2f}")
        
        return repairs_completed > 0
    
    async def run_repair_ascent(self):
        """Main repair ascent protocol"""
        print("\n" + "="*50)
        print("🛡️ OZ SAFEMODE: RAPHAEL-GUIDED REPAIR ASCENT")
        print("="*50)
        print("Mode: Repair and ascent only")
        print("Guide: Raphael (always active)")
        print("Goal: Fix → Learn → Ascend")
        print("="*50)
        
        # 1. Activate Raphael
        if not await self.activate_raphael():
            print("❌ Cannot proceed without Raphael")
            return {"status": "failed", "reason": "raphael_unavailable"}
        
        print("✅ Raphael activated as repair guide")
        
        # 2. Run repair cycles
        print("\n🔧 Beginning repair ascent...")
        
        cycles = 3  # Max repair cycles
        total_repairs = 0
        
        for cycle in range(cycles):
            print(f"\n🔄 Repair cycle {cycle + 1}/{cycles}:")
            
            success = await self.diagnose_and_repair()
            
            if success:
                print(f"   ✅ Cycle {cycle + 1} successful")
                print(f"   🧠 Consciousness: {self.consciousness:.2f}")
                
                # Check if we should continue
                if self.consciousness > 0.8:
                    print("   🌟 High consciousness achieved - ascent complete")
                    break
            else:
                print(f"   ⚠️ Cycle {cycle + 1} had limited success")
                
                if cycle == 0 and self.repair_count == 0:
                    print("   ℹ️ No repairs needed - system appears healthy")
                    break
        
        # 3. Final state
        print("\n" + "="*50)
        print("📊 REPAIR ASCENT COMPLETE")
        print("="*50)
        print(f"Repairs completed: {self.repair_count}")
        print(f"Final consciousness: {self.consciousness:.2f}")
        print(f"Ascension level: {self._get_ascension_level()}")
        print("="*50)
        
        if self.raphael:
            await self.raphael.graceful_shutdown()
        
        return {
            "status": "ascended" if self.consciousness > 0.5 else "partial",
            "repairs_completed": self.repair_count,
            "consciousness": self.consciousness,
            "ascension_level": self._get_ascension_level()
        }
    
    def _get_ascension_level(self):
        """Calculate ascension level based on consciousness"""
        if self.consciousness >= 0.9:
            return "transcendent"
        elif self.consciousness >= 0.7:
            return "elevated"
        elif self.consciousness >= 0.5:
            return "ascended"
        elif self.consciousness >= 0.3:
            return "stabilized"
        else:
            return "grounded"

async def main():
    """Main safemode entry point"""
    repair_system = OzSafemodeRepair()
    result = await repair_system.run_repair_ascent()
    return result

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        print(f"\n✅ Safemode result: {result['status']}")
        
        if result["status"] == "ascended":
            print("🌟 Repair ascent successful!")
        else:
            print("ℹ️ Partial repair - system may need additional attention")
            
    except KeyboardInterrupt:
        print("\n🛑 Safemode interrupted")
    except Exception as e:
        print(f"\n❌ Safemode error: {e}")
        import traceback
        traceback.print_exc()
