#!/usr/bin/env python3
"""
Launch Gabriel's Horn Network
"""
import asyncio
import json
import os
import sys
from horn_network_manager import HornNetworkManager
from loki_integration import loki

async def launch_network():
    """Launch the Gabriel's Horn network"""
    print("\n🌟 Launching Gabriel's Horn Network 🌟\n")
    
    # Log launch event
    loki.log_event(
        {"component": "launcher", "action": "start"},
        "Launching Gabriel's Horn Network"
    )
    
    # Create network manager
    network = HornNetworkManager()
    
    # Add horn stations (routing nodes)
    print("🔄 Creating horn stations...")
    horns = [
        ("horn1", 100, "gemma-2b"),
        ("horn2", 200, "hermes-2-pro-llama-3-7b"),
        ("horn3", 300, "qwen2.5-14b"),
        ("horn4", 400, "gemma-2b"),
        ("horn5", 500, "hermes-2-pro-llama-3-7b"),
        ("horn6", 600, "qwen2.5-14b"),
        ("horn7", 700, "gemma-2b")
    ]
    
    for horn_id, value, model in horns:
        network.add_station(horn_id, value, "horn", model)
        print(f"  ✅ Created {horn_id} with value {value} using {model}")
    
    # Add pod stations (endpoints)
    print("\n🔄 Creating pod stations...")
    pods = [
        ("pod1", 150, "gemma-2b"),
        ("pod2", 250, "hermes-2-pro-llama-3-7b"),
        ("pod3", 350, "qwen2.5-14b"),
        ("pod4", 450, "gemma-2b"),
        ("pod5", 550, "hermes-2-pro-llama-3-7b"),
        ("pod6", 650, "qwen2.5-14b"),
        ("pod7", 750, "gemma-2b")
    ]
    
    for pod_id, value, model in pods:
        network.add_station(pod_id, value, "pod", model)
        print(f"  ✅ Created {pod_id} with value {value} using {model}")
    
    # Create horn ring (main routing backbone)
    print("\n🔄 Creating horn ring...")
    horn_ids = [horn[0] for horn in horns]
    network.create_ring(horn_ids)
    print(f"  ✅ Created horn ring with {len(horn_ids)} horns")
    
    # Connect pods to nearest horns
    print("\n🔄 Connecting pods to horns...")
    for i, (pod_id, pod_value, _) in enumerate(pods):
        horn_id = horns[i][0]
        network.connect_stations(pod_id, horn_id)
        print(f"  ✅ Connected {pod_id} to {horn_id}")
    
    # Test the network
    print("\n🔄 Testing network...")
    test_messages = [
        ("pod1", 350, "Test message 1", "normal"),
        ("pod2", 650, "Test message 2", "high"),
        ("pod3", 150, "Test message 3", "low")
    ]
    
    for source, dest_value, content, priority in test_messages:
        print(f"  📤 Sending from {source} to value {dest_value} with {priority} priority")
        result = await network.send_message(source, dest_value, content, priority)
        if result.get("status") == "delivered":
            print(f"  📥 Delivered to {result['destination']} in {result['hops']} hops")
            print(f"  🛣️ Path: {' -> '.join(result['path'])}")
        else:
            print(f"  ❌ Delivery failed: {result.get('error')}")
    
    # Get network status
    print("\n🔄 Getting network status...")
    status = await network.get_network_status()
    print(f"  📊 Stations: {status['stations']}")
    print(f"  📊 Horns: {status['horns']}")
    print(f"  📊 Pods: {status['pods']}")
    print(f"  📊 Messages sent: {status['messages_sent']}")
    print(f"  📊 Messages delivered: {status['messages_delivered']}")
    
    print("\n✨ Gabriel's Horn Network is now active! ✨")
    
    # Log completion
    loki.log_event(
        {"component": "launcher", "action": "complete"},
        "Gabriel's Horn Network launched successfully"
    )
    
    return network

if __name__ == "__main__":
    network = asyncio.run(launch_network())