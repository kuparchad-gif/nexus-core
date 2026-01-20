# multi_ngrok.py
import subprocess
import time
import requests
import json
from threading import Thread

class MultiNgrok:
    def __init__(self):
        self.tunnels = {}
        
    def start_tunnel(self, name, port):
        """Start a named ngrok tunnel"""
        def run_tunnel():
            subprocess.run(f"ngrok http {port} --log=stdout", shell=True)
        
        thread = Thread(target=run_tunnel)
        thread.daemon = True
        thread.start()
        self.tunnels[name] = {"port": port, "thread": thread}
        time.sleep(2)  # Let ngrok start
        return self.get_tunnel_url(name)
    
    def get_tunnel_url(self, name):
        """Get the public URL for a tunnel"""
        try:
            resp = requests.get("http://localhost:4040/api/tunnels")
            data = resp.json()
            for tunnel in data['tunnels']:
                if str(self.tunnels[name]['port']) in tunnel['config']['addr']:
                    return tunnel['public_url']
        except:
            return None
        return None
    
    def start_all(self, services):
        """Start multiple services"""
        urls = {}
        for name, port in services.items():
            url = self.start_tunnel(name, port)
            urls[name] = url
            print(f"🌐 {name}: {url}")
        return urls

# Usage
if __name__ == "__main__":
    ngrok = MultiNgrok()
    
    services = {
        "lilith": 8000,
        "nexus": 8001, 
        "metatron": 8002,
        "oz": 8003
    }
    
    urls = ngrok.start_all(services)
    print("\n🎯 All tunnels active!")
    for name, url in urls.items():
        print(f"   {name.upper()}: {url}")
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping all tunnels...")