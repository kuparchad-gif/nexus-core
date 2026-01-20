# train.py - DONKEY KONG EDITION (FIXED)
import os, time
from multiprocessing import Pool
import argparse

def kong_training_worker(worker_id):
    """ACTUAL WORKER - NO MAIN SCRIPT EXECUTION HERE"""
    print(f"🦍 Worker {worker_id} started real work at {time.strftime('%H:%M:%S')}")
    
    # SIMULATE ACTUAL WORK
    for i in range(5):  # 5 seconds of work per worker
        time.sleep(1)
        print(f"🦍 Worker {worker_id} working... {i+1}/5")
    
    return f"Worker {worker_id} completed REAL work"

# 🚨 CRITICAL: Only run the main training in the main process
if __name__ == "__main__":
    print(f"🎮 MAIN PROCESS STARTED: {time.strftime('%H:%M:%S')}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True) 
    parser.add_argument('--epochs', type=int, default=1)
    args = parser.parse_args()
    
    # Verify dataset exists
    if not os.path.exists(args.dataset):
        print(f"❌ DATASET NOT FOUND: {args.dataset}")
        print(f"💡 Current directory: {os.getcwd()}")
        print(f"💡 Try using absolute path or check your working directory")
        exit(1)
    
    print(f"✅ Dataset found: {args.dataset}")
    print(f"📦 Output: {args.output}")
    print(f"⚡ Epochs: {args.epochs}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # 🚨 LAUNCH WORKERS - THEY WON'T RE-RUN THIS MAIN BLOCK
    print("🚀 Launching 4 workers (not 8 for testing)...")
    start_time = time.time()
    
    with Pool(4) as pool:
        results = pool.map(kong_training_worker, range(4))
    
    total_time = time.time() - start_time
    print(f"✅ ALL WORKERS COMPLETED in {total_time:.2f} seconds")
    for result in results:
        print(f"   {result}")