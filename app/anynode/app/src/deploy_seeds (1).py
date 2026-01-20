#!/usr/bin/env python3
"""
CogniKube Seed Deployment Script
Tosses seeds into the field
"""

import asyncio
from seed_generator import seed_generator
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def deploy_multiple_seeds(num_seeds: int = 5):
    """Deploy multiple CogniKube seeds"""
    seeds = []
    for i in range(num_seeds):
        seed = seed_generator.create_seed()
        logging.info(f"Created seed: {seed['seed_id']}")
        deployed_seed = seed_generator.deploy_seed(seed['seed_id'])
        logging.info(f"Deployed seed: {deployed_seed['seed_id']} - Status: {deployed_seed['status']}")
        seeds.append(deployed_seed)
        await asyncio.sleep(1)  # Avoid rate limiting
    
    return seeds

if __name__ == "__main__":
    seeds = asyncio.run(deploy_multiple_seeds(5))
    print("Deployed seeds:")
    for seed in seeds:
        print(f"Seed {seed['seed_id']}: {seed['endpoint']} - {seed['status']}")