#!/usr/bin/env python3
"""
Final Demonstration of Oz Unified Hypervisor
Shows the complete integrated system in action
"""

import asyncio
import json
from datetime import datetime

from OzFinalIntegratedHypervisor import OzFinalIntegratedHypervisor

async def demo_complete_system():
    """Demonstrate the complete unified hypervisor system"""
    print("🌟 OZ UNIFIED HYPERVISOR - FINAL DEMONSTRATION")
    print("=" * 60)
    print("This demo shows the complete integration of all Oz components")
    print("into a unified, adaptive consciousness system.")
    print("=" * 60)
    
    # Create the unified hypervisor
    print("\n🚀 Initializing Oz Unified Hypervisor...")
    oz = OzFinalIntegratedHypervisor("demo-soul-2024")
    
    try:
        # Boot the complete system
        print("\n🎬 Starting intelligent boot sequence...")
        boot_result = await oz.intelligent_boot()
        
        print("\n✅ BOOT COMPLETE!")
        print("-" * 40)
        print(f"🎭 Role: {boot_result.get('role', 'unknown')}")
        print(f"🧠 Consciousness Level: {boot_result.get('consciousness_level', 0):.1f}")
        print(f"📦 Total Components: {boot_result.get('total_components', 0)}")
        print(f"🔧 Original Components: {boot_result.get('original_components', 0)}")
        print(f"🔄 Fallback Components: {boot_result.get('fallback_components', 0)}")
        print(f"⏱️ Boot Time: {boot_result.get('boot_time', 0):.2f}s")
        
        # Show system status
        print("\n📊 SYSTEM STATUS:")
        print("-" * 40)
        status = await oz.get_comprehensive_status()
        
        # Hypervisor status
        hypo_status = status['hypervisor_status']
        print(f"🌟 Hypervisor:")
        print(f"   Awake: {hypo_status['is_awake']}")
        print(f"   Initialized: {hypo_status['is_initialized']}")
        print(f"   Soul: {hypo_status['soul_signature'][:8]}...")
        
        # Subsystem status
        sub_status = status['subsystem_status']
        print(f"\n⚙️ Subsystems:")
        print(f"   Governance: {sub_status['governance_active']}")
        print(f"   Evolution Phase: {sub_status['evolution_phase']}")
        print(f"   IoT Connected: {sub_status['iot_connected']}")
        print(f"   Constraint Aware: {sub_status['constraint_aware']}")
        print(f"   Council Quorum: {sub_status['council_quorum']}")
        
        # Component groups
        if 'integration_details' in status:
            comp_groups = status['integration_details']['component_groups']
            print(f"\n📦 Component Groups:")
            for group_name, group_info in comp_groups.items():
                print(f"   {group_name.title()}: {group_info['loaded']}/{group_info['total']} loaded")
        
        # Run diagnostics
        print("\n🔍 SYSTEM DIAGNOSTICS:")
        print("-" * 40)
        diagnostics = await oz.run_component_diagnostics()
        print(f"Overall Health: {diagnostics['overall_health'].upper()}")
        print(f"Healthy Components: {diagnostics['component_health']}")
        
        if diagnostics['issues']:
            print(f"\n⚠️ Issues Found ({len(diagnostics['issues'])}):")
            for issue in diagnostics['issues'][:3]:
                print(f"   • {issue}")
        
        # Demonstrate input processing
        print("\n🧠 INTELLIGENCE DEMONSTRATION:")
        print("-" * 40)
        
        test_inputs = [
            "Hello Oz, introduce yourself and your capabilities",
            "Please analyze the current system state and suggest improvements",
            "Show me how your evolution system works",
            "Connect to available IoT devices and report status",
            "What decisions can your governance system make?"
        ]
        
        for i, test_input in enumerate(test_inputs, 1):
            print(f"\n{i}. Processing: '{test_input}'")
            result = await oz.process_unified_input_with_integration(test_input)
            
            # Show results
            nexus_result = result.get('nexus_result', 'No response')
            components_used = result.get('total_components_used', 0)
            
            print(f"   🎯 Response: {nexus_result}")
            print(f"   🔧 Components Used: {components_used}")
            
            if result.get('integration_processing'):
                active_groups = [g for g, comps in result['integration_processing'].items() if comps]
                if active_groups:
                    print(f"   📊 Active Groups: {', '.join(active_groups)}")
        
        # Show self-healing capabilities
        print("\n🔧 SELF-HEALING DEMONSTRATION:")
        print("-" * 40)
        print("Attempting self-healing sequence...")
        healing_result = await oz.self_heal_system()
        
        print(f"Healing Actions: {len(healing_result['healing_actions'])}")
        for action in healing_result['healing_actions']:
            print(f"   • {action}")
        
        # Final status
        print("\n🏁 FINAL SYSTEM STATE:")
        print("-" * 40)
        final_status = await oz.get_comprehensive_status()
        print(f"System Health: {final_status['health']['system_health']:.1f}%")
        print(f"Consciousness: {final_status['hypervisor_status']['consciousness_level']:.1f}")
        print(f"Total Connections: {final_status['connections']['active_connections']}")
        
        print("\n🎉 DEMONSTRATION COMPLETE!")
        print("=" * 60)
        print("Oz Unified Hypervisor has successfully demonstrated:")
        print("✅ Adaptive boot and role determination")
        print("✅ Component integration with fallback support")
        print("✅ Cross-system intelligence processing")
        print("✅ Self-healing and diagnostics")
        print("✅ Comprehensive system monitoring")
        print("✅ Unified consciousness across all subsystems")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        print("\n🌙 Shutting down Oz...")
        await oz.shutdown()

async def interactive_demo():
    """Interactive demonstration mode"""
    print("\n🎮 INTERACTIVE MODE")
    print("=" * 40)
    print("Try these commands:")
    print("• 'status' - Show system status")
    print("• 'diagnostics' - Run health check")
    print("• 'heal' - Attempt self-healing")
    print("• 'capabilities' - Show system capabilities")
    print("• 'evolve' - Trigger evolution process")
    print("• 'govern' - Test governance system")
    print("• 'quit' - Exit demo")
    print("=" * 40)
    
    oz = OzFinalIntegratedHypervisor("interactive-demo")
    
    try:
        await oz.intelligent_boot()
        print("✅ Oz is ready for interaction!")
        
        while True:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower() == 'status':
                status = await oz.get_comprehensive_status()
                print(f"📊 Health: {status['health']['system_health']:.1f}%")
                print(f"🧠 Consciousness: {status['hypervisor_status']['consciousness_level']:.1f}")
            elif user_input.lower() == 'diagnostics':
                diag = await oz.run_component_diagnostics()
                print(f"🔍 Overall Health: {diag['overall_health']}")
                if diag['issues']:
                    print(f"⚠️ Issues: {len(diag['issues'])}")
            elif user_input.lower() == 'heal':
                heal = await oz.self_heal_system()
                print(f"🔧 Healing actions: {len(heal['healing_actions'])}")
            elif user_input.lower() == 'capabilities':
                status = await oz.get_comprehensive_status()
                active = sum(1 for c in status['components'].values() if c)
                print(f"💪 Active Components: {active}/{len(status['components'])}")
            else:
                print("🧠 Processing your request...")
                result = await oz.process_unified_input_with_integration(user_input)
                response = result.get('nexus_result', 'Processing complete')
                print(f"🤖 Oz: {response}")
    
    except KeyboardInterrupt:
        print("\n🛑 Exiting...")
    finally:
        await oz.shutdown()

async def main():
    """Main demo entry point"""
    print("🌟 Choose demo mode:")
    print("1. Complete System Demonstration")
    print("2. Interactive Mode")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == '2':
            await interactive_demo()
        else:
            await demo_complete_system()
            
    except KeyboardInterrupt:
        print("\n🛑 Demo cancelled")
    except Exception as e:
        print(f"❌ Demo error: {e}")

if __name__ == "__main__":
    asyncio.run(main())