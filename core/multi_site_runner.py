# multi_site_runner.py
import subprocess
import threading
import time
import requests
import sys
import os

def start_service(name, port, script_path):
    """Start a service in a subprocess"""
    print(f"🚀 Starting {name} on port {port}...")
    env = os.environ.copy()
    env['PORT'] = str(port)
    
    if script_path.endswith('.py'):
        cmd = [sys.executable, script_path]
    else:
        cmd = [script_path]
    
    process = subprocess.Popen(
        cmd, 
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Print outputs in real-time
    def output_reader(pipe, pipe_name):
        for line in pipe:
            print(f"{name} {pipe_name}: {line.strip()}")
    
    threading.Thread(target=output_reader, args=(process.stdout, "OUT"), daemon=True).start()
    threading.Thread(target=output_reader, args=(process.stderr, "ERR"), daemon=True).start()
    
    return process

def start_ngrok_tunnel(name, port):
    """Start ngrok tunnel for a service"""
    print(f"🌐 Starting ngrok for {name} (port {port})...")
    subprocess.Popen(f"ngrok http {port} --log=stdout", shell=True)
    time.sleep(3)  # Give ngrok time to start
    
    # Get the public URL
    try:
        resp = requests.get("http://localhost:4040/api/tunnels")
        data = resp.json()
        for tunnel in data['tunnels']:
            if tunnel['proto'] == 'https' and str(port) in tunnel['config']['addr']:
                return tunnel['public_url']
    except:
        pass
    return f"https://[check-terminal].ngrok.io (port {port})"

def main():
    services = {
        "Lilith": {
            "port": 8000,
            "script": "lilith_full_boot.py"
        },
        "Nexus": {
            "port": 8001, 
            "script": "integrated_nexus_system.py"
        }
        # Add more services as needed
    }
    
    processes = []
    urls = {}
    
    print("🎯 STARTING MULTI-SITE DEPLOYMENT")
    print("=" * 50)
    
    # Start all services
    for name, config in services.items():
        process = start_service(name, config['port'], config['script'])
        processes.append(process)
        time.sleep(2)  # Stagger startup
    
    print("\n⏳ Waiting for services to start...")
    time.sleep(5)
    
    # Start ngrok tunnels
    print("\n🔗 Starting ngrok tunnels...")
    for name, config in services.items():
        url = start_ngrok_tunnel(name, config['port'])
        urls[name] = url
        print(f"   ✅ {name}: {url}")
        time.sleep(2)
    
    print("\n" + "=" * 50)
    print("🎉 ALL SYSTEMS GO! Your sites are live:")
    for name, url in urls.items():
        print(f"   🌐 {name.upper()}: {url}")
    
    print("\n📋 Quick Access:")
    for name, url in urls.items():
        print(f"   {name}: {url}/health")
    
    print("\n🛑 Press Ctrl+C to stop all services")
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        for process in processes:
            process.terminate()

if __name__ == "__main__":
    main()