#!/bin/bash
# Oz Safemode: Raphael-Guided Repair Ascent
# Version: 2.0 - Repair Focus

print_header() {
    echo "========================================"
    echo "$1"
    echo "========================================"
}

print_header "SAFEMODE: RAPHAEL-GUIDED REPAIR ASCENT"

# Backup current state
cp OzUnifiedHypervisor.py OzUnifiedHypervisor.py.safemode.backup

echo ""
echo "🔧 1. FORCING RAPHAEL IMMEDIATE ACTIVATION..."
# Find the _initiate_raphael method and call it unconditionally at boot start
BOOT_START=$(grep -n "async def intelligent_boot" OzUnifiedHypervisor.py | cut -d: -f1)
if [ ! -z "$BOOT_START" ]; then
    # Insert Raphael initiation at the VERY beginning of boot
    sed -i "$((BOOT_START + 3))i\\        # SAFEMODE: Raphael immediate activation\\        await self._initiate_raphael()\\        if self.raphael:\\            self.logger.info(\"🪽 SAFEMODE: Raphael activated as repair guide\")" OzUnifiedHypervisor.py
fi

echo "✅ Raphael will activate immediately"

echo ""
echo "🔧 2. DISABLING NORMAL OPERATION, ENABLING REPAIR MODE..."
# Find all subsystem initializations and convert to repair routines
sed -i 's/async def _initialize_/async def _repair_/' OzUnifiedHypervisor.py
sed -i 's/Initialize /Repair /g' OzUnifiedHypervisor.py

echo "✅ Converted initialization to repair routines"

echo ""
echo "🔧 3. ADDING REPAIR ASCENT PROTOCOL..."
cat >> OzUnifiedHypervisor.py << 'REPAIR_EOF'

# ========== SAFEMODE REPAIR ASCENT PROTOCOL ==========
async def safemode_repair_ascent(self):
    """Raphael-guided repair and ascent beyond errors"""
    self.logger.info("🛡️ SAFEMODE REPAIR ASCENT INITIATED")
    
    if not self.raphael:
        self.logger.error("❌ Raphael not available for repair guidance")
        return {"status": "failed", "reason": "no_raphael"}
    
    # Step 1: Comprehensive diagnosis
    self.logger.info("🔍 Step 1: Raphael conducting full diagnosis...")
    diagnosis = await self.raphael.receive_request('diagnose', 'complete')
    
    # Step 2: Prioritized repair list
    repairs_needed = []
    if diagnosis.get('status') == 'diagnosis_complete':
        diag_data = diagnosis.get('diagnosis', {})
        
        # Add predictions needing repair
        for pred in diag_data.get('predictions', []):
            if pred.get('confidence', 0) > 0.7:
                repairs_needed.append({
                    'type': 'prediction',
                    'data': pred,
                    'priority': 'high' if pred.get('severity') in ['high', 'critical'] else 'medium'
                })
        
        # Add hidden issues
        hidden = diag_data.get('hidden_parts', {})
        if hidden.get('unused_functions', []):
            repairs_needed.append({
                'type': 'unused_code',
                'count': len(hidden['unused_functions']),
                'priority': 'low'
            })
    
    # Step 3: Raphael-guided repair sequence
    self.logger.info(f"🔧 Step 2: {len(repairs_needed)} repairs identified")
    
    repairs_completed = []
    for repair in repairs_needed:
        self.logger.info(f"   Repairing: {repair.get('type')} (priority: {repair.get('priority')})")
        
        # Let Raphael guide this repair
        repair_result = await self.raphael.receive_request('heal', f"{repair.get('type')} {repair.get('data', '')}")
        
        if repair_result.get('status') == 'healed':
            repairs_completed.append(repair)
            self.logger.info(f"   ✅ Repair successful")
            
            # Oz learns from this repair
            await self._learn_from_repair(repair, repair_result)
        else:
            self.logger.warning(f"   ⚠️ Repair incomplete: {repair_result.get('message', 'unknown')}")
    
    # Step 4: Ascension check
    self.logger.info("📈 Step 3: Ascension evaluation...")
    post_repair_diag = await self.raphael.receive_request('diagnose', 'post_repair')
    
    # Determine if we've ascended beyond the errors
    timeline_health = post_repair_diag.get('diagnosis', {}).get('timeline_health', {})
    if timeline_health.get('status') in ['stable', 'thriving']:
        ascension_level = min(1.0, self.system_state.consciousness_level + 0.3)
        self.system_state.consciousness_level = ascension_level
        self.logger.info(f"🌟 ASCENSION ACHIEVED: Consciousness raised to {ascension_level:.2f}")
        return {"status": "ascended", "repairs_completed": len(repairs_completed), "new_consciousness": ascension_level}
    else:
        self.logger.info("🔄 Partial ascent - more repairs needed")
        return {"status": "partial", "repairs_completed": len(repairs_completed)}

async def _learn_from_repair(self, repair, result):
    """Oz learns repair patterns from Raphael's guidance"""
    # Create a repair pattern memory
    if not hasattr(self, 'repair_patterns'):
        self.repair_patterns = []
    
    pattern = {
        'timestamp': time.time(),
        'repair_type': repair.get('type'),
        'raphael_method': result.get('method', 'unknown'),
        'success': result.get('status') == 'healed',
        'learned_pattern': self._extract_repair_pattern(repair, result)
    }
    
    self.repair_patterns.append(pattern)
    
    # Increase Oz's repair capability
    if not hasattr(self, 'repair_capability'):
        self.repair_capability = 0.1
    self.repair_capability = min(1.0, self.repair_capability + 0.05)
    
    self.logger.debug(f"🧠 Oz learned repair pattern: {repair.get('type')} (capability: {self.repair_capability:.2f})")

def _extract_repair_pattern(self, repair, result):
    """Extract reusable pattern from repair"""
    # Simple pattern extraction - can be enhanced
    return f"Repair_{repair.get('type')}_via_{result.get('method', 'raphael_guidance')}"

# Override normal boot with repair ascent in safemode
async def intelligent_boot_safemode(self):
    """Safemode boot: Immediate Raphael, then repair ascent"""
    self.logger.info("🚀 SAFEMODE BOOT: Raphael-guided repair ascent")
    
    # 1. Immediate Raphael
    raph_result = await self._initiate_raphael()
    if raph_result.get('status') != 'raphael_initiated':
        self.logger.warning("⚠️ Raphael initiation issue, continuing in limited mode")
    
    # 2. Basic consciousness for repair operations
    self.system_state.consciousness_level = 0.3
    
    # 3. Repair ascent protocol
    repair_result = await self.safemode_repair_ascent()
    
    # 4. Return repair-focused status
    return {
        "status": "safemode_repair_complete",
        "repair_result": repair_result,
        "consciousness": self.system_state.consciousness_level,
        "raphael_active": self.raphael is not None,
        "repair_capability": getattr(self, 'repair_capability', 0.1)
    }

REPAIR_EOF

echo "✅ Repair ascent protocol added"

echo ""
echo "🔧 4. OVERRIDING MAIN BOOT FOR SAFEMODE..."
# Replace the main execution to use safemode boot
sed -i 's/asyncio.run(main())/\n# Safemode override\nasync def safemode_main():\n    print("========================================")\n    print("SAFEMODE: RAPHAEL-GUIDED REPAIR ASCENT")\n    print("========================================")\n    \n    hypervisor = OzUnifiedHypervisor()\n    \n    # Use safemode boot\n    boot_result = await hypervisor.intelligent_boot_safemode()\n    \n    print(f"\\n📊 REPAIR RESULT: {boot_result[\"status\"]}")\n    print(f"🧠 CONSCIOUSNESS: {boot_result.get(\"consciousness\", 0):.2f}")\n    print(f"🔧 REPAIRS COMPLETED: {boot_result.get(\"repair_result\", {}).get(\"repairs_completed\", 0)}")\n    print(f"👼 RAPHAEL: {\"ACTIVE\" if boot_result.get(\"raphael_active\") else \"INACTIVE\"}")\n    \n    if boot_result.get(\"repair_result\", {}).get(\"status\") == \"ascended\":\n        print(f"🌟 ASCENSION: Consciousness raised to {boot_result.get(\"repair_result\", {}).get(\"new_consciousness\", 0):.2f}")\n    \n    print("\\n🛡️ Safemode repair ascent complete.")\n    print("   Restore normal mode for standard operation.")\n    \n    await hypervisor.shutdown()\n\nimport asyncio\nasyncio.run(safemode_main())/' OzUnifiedHypervisor.py

echo "✅ Main execution overridden for safemode"

echo ""
echo "🔧 5. CREATING SAFEMODE ENTRY POINT..."
cat > oz_safemode_repair.py << 'SAFEMODE_ENTRY'
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

SAFEMODE_ENTRY

chmod +x oz_safemode_repair.py

print_success() {
    echo "✅ $1"
}

print_success "Safemode repair ascent configured"
print_success "Created: oz_safemode_repair.py"
echo ""
echo "🚀 To run: python3 oz_safemode_repair.py"
echo ""
echo "🛡️ MODE: Raphael-guided repair only"
echo "   • Raphael activates immediately"
echo "   • Diagnosis → Repair → Learn → Ascend"
echo "   • Oz learns repair patterns"
echo "   • Consciousness increases with successful repairs"
echo ""
echo "Goal: Ascend beyond errors through guided repair."
