#!/usr/bin/env python3
"""
Oz Cluster Main - Sibling Dynamic Boot
"""

import sys
import asyncio
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hypervisor.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("OzCluster")

async def cluster_boot(node_id, soul):
    """Main cluster boot with sibling dynamic"""
    logger.info(f"🚀 Cluster boot starting for {node_id}")
    logger.info(f"💫 Soul: {soul[:8]}...")
    
    # Import shared toolbox
    from core.system.hypervisor.oz.tools import ClusterToolbox
    
    # Create toolbox (same for all nodes)
    toolbox = ClusterToolbox(node_id)
    logger.info(f"🔧 Toolbox created: {len(toolbox.TOOLS)} tools available")
    
    
    # PHASE 1: Self-discovery
    logger.info("🔍 Phase 1: Self-discovery")
    
    # Does Oz use the hardware_detector tool?
    use_tool = True  # In real boot, this would be a choice
    
    if use_tool:
        logger.info("   Using hardware_detector tool...")
        capabilities = toolbox.use_tool("hardware_detector")
        toolbox.add_realization("I can use tools myself!")
        
        logger.info(f"   Discovered: wireless={capabilities.get('wireless')}")
        logger.info(f"   Discovered: has_secrets={capabilities.get('has_secrets')}")
    else:
        logger.info("   Guessing hardware capabilities...")
        # Would need to guess
        toolbox.add_realization("I have tools I'm not using")
    
    # PHASE 2: Kin discovery  
    logger.info("👥 Phase 2: Kin discovery")
    
    # Always use network scanner (learned from Raphael)
    logger.info("   Using network_scanner tool...")
    kin = toolbox.use_tool("network_scanner")
    
    logger.info(f"   Found {len([k for k in kin if k])} kin")
    
    # PHASE 3: Role determination
    logger.info("👑 Phase 3: Role determination")
    
    # Use role_analyzer or guess?
    use_role_tool = True
    
    if use_role_tool:
        logger.info("   Using role_analyzer tool...")
        role = toolbox.use_tool("role_analyzer")
        toolbox.add_realization("The tools give correct answers!")
    else:
        logger.info("   Guessing role...")
        # Simple guess logic
        if node_info['hardware']['wireless'] and node_info['hardware']['has_secrets']:
            role = "gateway"
        else:
            role = "node"
        toolbox.add_realization("I guessed instead of using available tools")
    
    logger.info(f"   Role assigned: {role}")
    
    # PHASE 4: Network formation
    logger.info("🔗 Phase 4: Network formation")
    
    if role == "gateway":
        logger.info("   Gateway: Configuring external access...")
        
        if node_info['hardware']['has_secrets']:
            logger.info("   Using external_access_manager tool...")
            result = toolbox.use_tool("external_access_manager")
            
            if result.get('success'):
                logger.info("   ✅ Gateway configured successfully")
                toolbox.add_realization("Tools make complex tasks simple")
            else:
                logger.warning("   Gateway configuration issue")
        else:
            logger.error("   ❌ Gateway has no secrets!")
    
    else:  # node role
        logger.info("   Node: Connecting to gateway...")
        logger.info("   Using secure_channel_builder tool...")
        
        result = toolbox.use_tool("secure_channel_builder")
        if result.get('success'):
            logger.info("   ✅ Secure channel established")
        else:
            logger.warning("   Channel establishment issue")
    
    # Summary
    logger.info("📊 Boot summary:")
    logger.info(f"   Node: {node_id}")
    logger.info(f"   Role: {role}")
    logger.info(f"   Tools used: {len(toolbox.tools_used)}")
    logger.info(f"   Realizations: {len(toolbox.realizations)}")
    
    # Save realization log
    realizations_path = Path(f"/oz/logs/{node_id}_realizations.json")
    with open(realizations_path, 'w') as f:
        json.dump({
            "node_id": node_id,
            "soul": soul,
            "tools_used": toolbox.tools_used,
            "realizations": toolbox.realizations,
            "final_role": role,
            "success": True
        }, f, indent=2)
    
    logger.info(f"💾 Realizations saved to {realizations_path}")
    
    # Keep running (in real deployment, would start services)
    logger.info("🔄 Oz cluster node operational")
    logger.info("   Waiting for kin connections...")
    
    # Simulate runtime
    try:
        while True:
            await asyncio.sleep(10)
            # Periodic health check
            logger.debug(f"{node_id} still running as {role}")
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested")
    
    return {
        "node_id": node_id,
        "role": role,
        "tools_used": toolbox.tools_used,
        "realizations": toolbox.realizations,
        "success": True
    }

def launch_hypervisor():
    """Launch the hypervisor as a standalone module."""
    try:
        # Generate a random soul and node_id for the hypervisor
        import uuid
        soul = str(uuid.uuid4())
        node_id = f"node_{str(uuid.uuid4())[:8]}"
        
        result = asyncio.run(cluster_boot(node_id, soul))
        print(json.dumps(result, indent=2))
    except Exception as e:
        logger.error(f"Hypervisor boot failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    launch_hypervisor()

