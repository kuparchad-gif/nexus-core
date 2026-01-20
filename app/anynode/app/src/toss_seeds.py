#!/usr/bin/env python3
"""
Toss CogniKube Seeds Into The Field
Rapidly deploy multiple CogniKube seeds across various environments
"""

import asyncio
import argparse
import json
import random
import time
from typing import List, Dict, Any
from seed_generator import SeedDeployer

class SeedTosser:
    """
    Tosses CogniKube seeds into the field (deploys them widely)
    """
    
    def __init__(self):
        self.deployer = SeedDeployer()
        self.tossed_seeds = []
        self.environments = ["modal", "local"]  # Add more environments as needed
    
    async def toss_seeds(self, count: int, pattern: str = "random") -> List[Dict[str, Any]]:
        """
        Toss seeds into the field using the specified pattern
        
        Patterns:
        - random: Deploy randomly across environments
        - cluster: Deploy in clusters to the same environments
        - sequence: Deploy in sequence to different environments
        """
        results = []
        
        if pattern == "random":
            # Random deployment across environments
            environments = [random.choice(self.environments) for _ in range(count)]
            results = await self.deployer.deploy_multiple(count, environments)
        
        elif pattern == "cluster":
            # Deploy in clusters to the same environments
            clusters = {}
            for env in self.environments:
                # Distribute seeds evenly across environments
                clusters[env] = count // len(self.environments)
            
            # Distribute any remainder
            remainder = count % len(self.environments)
            for env in list(clusters.keys())[:remainder]:
                clusters[env] += 1
            
            # Deploy each cluster
            for env, cluster_count in clusters.items():
                if cluster_count > 0:
                    cluster_results = await self.deployer.deploy_multiple(cluster_count, [env])
                    results.extend(cluster_results)
        
        elif pattern == "sequence":
            # Deploy in sequence to different environments
            for i in range(count):
                env = self.environments[i % len(self.environments)]
                seed = self.deployer.generate_seed()
                result = await self.deployer.deploy_seed(seed, env)
                results.append(result)
                await asyncio.sleep(1)  # Small delay between deployments
        
        else:
            raise ValueError(f"Unknown deployment pattern: {pattern}")
        
        self.tossed_seeds.extend(results)
        return results
    
    def get_tossed_seeds(self) -> List[Dict[str, Any]]:
        """Get information about all tossed seeds"""
        return self.tossed_seeds
    
    async def monitor_seeds(self, duration: int = 60) -> Dict[str, Any]:
        """Monitor the tossed seeds for the specified duration (in seconds)"""
        start_time = time.time()
        end_time = start_time + duration
        
        seed_status = {seed["seed_id"]: {"status": "unknown"} for seed in self.tossed_seeds}
        
        while time.time() < end_time:
            for seed in self.tossed_seeds:
                seed_id = seed.get("seed_id")
                if not seed_id:
                    continue
                
                # Check seed health
                try:
                    if seed.get("environment") == "modal" and seed.get("url"):
                        import httpx
                        async with httpx.AsyncClient() as client:
                            response = await client.get(f"{seed['url']}/health", timeout=5)
                            if response.status_code == 200:
                                seed_status[seed_id] = response.json()
                except Exception as e:
                    seed_status[seed_id] = {"status": "error", "message": str(e)}
            
            # Wait before checking again
            await asyncio.sleep(10)
        
        return {
            "monitoring_duration": duration,
            "seeds_monitored": len(self.tossed_seeds),
            "seed_status": seed_status
        }

async def main():
    parser = argparse.ArgumentParser(description="Toss CogniKube seeds into the field")
    parser.add_argument("--count", type=int, default=5, help="Number of seeds to toss")
    parser.add_argument("--pattern", type=str, default="random", choices=["random", "cluster", "sequence"], 
                        help="Deployment pattern")
    parser.add_argument("--monitor", type=int, default=0, help="Monitor duration in seconds (0 to skip)")
    args = parser.parse_args()
    
    tosser = SeedTosser()
    
    print(f"Tossing {args.count} CogniKube seeds using {args.pattern} pattern...")
    results = await tosser.toss_seeds(args.count, args.pattern)
    
    print(f"Tossed {len(results)} seeds:")
    for i, result in enumerate(results):
        print(f"  Seed {i+1}: {result.get('seed_id')} - {result.get('status')} on {result.get('environment')}")
    
    if args.monitor > 0:
        print(f"\nMonitoring seeds for {args.monitor} seconds...")
        monitoring = await tosser.monitor_seeds(args.monitor)
        print("\nMonitoring results:")
        print(json.dumps(monitoring, indent=2))

if __name__ == "__main__":
    asyncio.run(main())