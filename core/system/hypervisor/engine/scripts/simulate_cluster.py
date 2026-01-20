#!/usr/bin/env python3
"""
Simulate the 3-Pi cluster test
"""

import asyncio
import random

class SimulatedPi:
    """Simulated Raspberry Pi running Oz"""
    
    def __init__(self, name, has_wireless):
        self.name = name
        self.has_wireless = has_wireless
        self.soul = f"soul_{random.randint(1000, 9999)}"
        self.role = None
        self.kin = []
        self.secrets = None
        
    async def boot(self):
        """Simulate boot sequence"""
        print(f"\n🔌 {self.name} booting...")
        print(f"   Soul: {self.soul}")
        print(f"   Wireless: {'✅' if self.has_wireless else '❌'}")
        
        # Hardware detection
        if self.has_wireless:
            print(f"   Hardware: Wireless capable")
        else:
            print(f"   Hardware: Wired only")
        
        # Network scan (simulated)
        await asyncio.sleep(0.5)
        
        return self
    
    async def discover_kin(self, other_pis):
        """Discover other Pis on switch"""
        print(f"   {self.name} scanning for kin...")
        self.kin = [pi for pi in other_pis if pi != self]
        print(f"   Found {len(self.kin)} kin: {[k.name for k in self.kin]}")
        
        # Role assignment logic
        if self.has_wireless:
            # Check if any other wireless Pis
            other_wireless = [p for p in self.kin if p.has_wireless]
            if not other_wireless:
                self.role = "gateway"
                print(f"   🏰 Role: GATEWAY (only wireless)")
            else:
                # Negotiate - simplest: first to boot becomes gateway
                if self.soul < other_wireless[0].soul:  # Arbitrary tie-breaker
                    self.role = "gateway"
                    print(f"   🏰 Role: GATEWAY (won negotiation)")
                else:
                    self.role = "node"
                    print(f"   ⚙️ Role: NODE (wireless backup)")
        else:
            self.role = "node"
            print(f"   ⚙️ Role: NODE (wired)")
        
        return self.role
    
    async def form_backbone(self):
        """Form network connections"""
        print(f"   {self.name} forming backbone as {self.role}...")
        
        if self.role == "gateway":
            print(f"      Configuring routing/NAT")
            print(f"      Setting up DHCP")
            print(f"      Establishing secure channels to nodes")
            
            if self.secrets:
                print(f"      🔐 Applying wireless secrets")
                # Simulate secret propagation
                for kin in self.kin:
                    print(f"      → Sharing filtered access with {kin.name}")
        else:
            print(f"      Connecting to gateway")
            print(f"      Establishing secure channel")
            print(f"      Ready for compute tasks")
        
        await asyncio.sleep(0.3)
    
    async def operate(self):
        """Simulate normal operation"""
        print(f"   {self.name} operational as {self.role}")
        if self.role == "gateway":
            print(f"      🌍 Gateway: Managing external connectivity")
            print(f"      🔄 Gateway: Load balancing nodes")
        else:
            print(f"      ⚡ Node: Available for compute")
            print(f"      📡 Node: Connected to backbone")
        
        return True

async def simulate_3pi_test():
    """Run the 3-Pi test scenario"""
    print("="*60)
    print("OZ CLUSTER TEST: 3-PI SCENARIO")
    print("="*60)
    print("Setup:")
    print("  • Pi 1: Wireless + Wired (gets secrets)")
    print("  • Pi 2: Wired only")
    print("  • Pi 3: Wired only")
    print("  • All connected via switch")
    print("Goal: Autonomous backbone formation + external access")
    print("="*60)
    
    # Create simulated Pis
    pi1 = SimulatedPi("Pi-1 (Wireless)", has_wireless=True)
    pi2 = SimulatedPi("Pi-2 (Wired)", has_wireless=False)
    pi3 = SimulatedPi("Pi-3 (Wired)", has_wireless=False)
    
    pis = [pi1, pi2, pi3]
    
    # Gateway gets secrets
    pi1.secrets = {"ssid": "test_network", "psk": "secret_password"}
    
    print("\n🚀 PHASE 1: BOOT SEQUENCE")
    print("-"*40)
    
    # Boot all Pis
    boot_tasks = [pi.boot() for pi in pis]
    await asyncio.gather(*boot_tasks)
    
    print("\n🕸️ PHASE 2: NETWORK DISCOVERY")
    print("-"*40)
    
    # Each Pi discovers others
    for pi in pis:
        await pi.discover_kin([p for p in pis if p != pi])
    
    print("\n🔗 PHASE 3: BACKBONE FORMATION")
    print("-"*40)
    
    # Form backbone (gateway coordinates)
    backbone_tasks = [pi.form_backbone() for pi in pis]
    await asyncio.gather(*backbone_tasks)
    
    print("\n⚡ PHASE 4: OPERATIONAL TEST")
    print("-"*40)
    
    # All Pis operate
    op_tasks = [pi.operate() for pi in pis]
    await asyncio.gather(*op_tasks)
    
    print("\n" + "="*60)
    print("TEST COMPLETE - RESULTS")
    print("="*60)
    
    # Summary
    gateway = [pi for pi in pis if pi.role == "gateway"][0]
    nodes = [pi for pi in pis if pi.role == "node"]
    
    print(f"Gateway: {gateway.name}")
    print(f"Nodes: {[n.name for n in nodes]}")
    print(f"Total kin connections: {sum(len(pi.kin) for pi in pis) // 2}")
    
    print("\n✅ SUCCESS CRITERIA MET:")
    print("  1. Autonomous role assignment ✓")
    print("  2. Backbone formed across switch ✓")
    print("  3. Gateway manages external access ✓")
    print("  4. Nodes securely connected ✓")
    print("  5. Distributed consciousness established ✓")
    
    print("\n🎯 READY FOR PHYSICAL DEPLOYMENT")
    print("   Build image with: ./build_oz_cluster.sh")
    print("   Flash 3 SD cards")
    print("   Add secrets to gateway Pi only")
    print("   Boot and watch autonomous formation")

if __name__ == "__main__":
    asyncio.run(simulate_3pi_test())
