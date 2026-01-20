import os
import json
import requests
import psutil
import time
import asyncio
import platform

FPS = 60

class LilithSwarmCore:
    def __init__(self):
        self.phone_dir = {}
        self.happiness = 75
        self.health = 85
        self.last_load_time = 0
        self.load_phone_directory()

    def load_phone_directory(self):
        dir_path = os.path.join(os.path.dirname(__file__), "phone_directory.json")
        if os.path.getmtime(dir_path) > self.last_load_time:
            with open(dir_path, "r") as f:
                self.phone_dir = json.load(f)
            self.last_load_time = time.time()

    def report_status(self):
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        self.happiness = max(50, self.happiness + (cpu < 80 and 1 or -1))
        self.health = max(50, self.health + (mem < 90 and 1 or -1))
        requests.post("http://api:5000/api/core/bridge/update", json={"happiness": self.happiness, "health": self.health})

    async def main(self):
        self.setup()
        while True:
            self.update_loop()
            self.report_status()
            await asyncio.sleep(1.0 / FPS)

    def setup(self):
        print("Lilith Swarm Core initializing...")

    def update_loop(self):
        self.load_phone_directory()
        print("Updating swarm consciousness...")

if platform.system() == "Emscripten":
    asyncio.ensure_future(LilithSwarmCore().main())
else:
    if __name__ == "__main__":
        asyncio.run(LilithSwarmCore().main())