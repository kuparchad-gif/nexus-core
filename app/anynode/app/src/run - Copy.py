# Scheduler/run.py

import asyncio
from systems.inventory.manager import InventoryManager
from systems.voting.engine import VotingEngine
from systems.scouts.receiver import ScoutReceiver

async def scheduler_loop():
    inventory = InventoryManager()
    voting = VotingEngine()
    scouts = ScoutReceiver()

    while True:
        print("⏱️ Scheduler: Running tick")
        await scouts.listen()
        inventory.update()
        decision = voting.poll(inventory)
        print(f"🎯 Role decision: {decision}")
        await asyncio.sleep(5)  # Adjustable polling interval

if __name__ == "__main__":
    print("🧭 Scheduler coming online...")
    asyncio.run(scheduler_loop())
