# 📂 Path: /Utilities/network_core/quantum_stream.py

import os
import hashlib
import json
import threading
import time

QUANTUM_STORAGE_PATH = '/Memory/streams/quantum_memory/'

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class QuantumStream:
    def __init__(self):
        ensure_dir(QUANTUM_STORAGE_PATH)

    def split_memory_into_quanta(self, memory_data, chunk_size=4096):
        payload = json.dumps(memory_data).encode('utf-8')
        return [payload[i:i+chunk_size] for i in range(0, len(payload), chunk_size)]

    def store_quanta(self, shard_id, quanta_list):
        for idx, chunk in enumerate(quanta_list):
            filename = f"{shard_id}_chunk{idx}.qnt"
            filepath = os.path.join(QUANTUM_STORAGE_PATH, filename)
            with open(filepath, 'wb') as f:
                f.write(chunk)
        print(f"[QuantumStream] {len(quanta_list)} quanta stored for {shard_id}.")

    def reconstruct_memory(self, shard_id):
        files = sorted([f for f in os.listdir(QUANTUM_STORAGE_PATH) if f.startswith(shard_id)])
        payload = b''.join(open(os.path.join(QUANTUM_STORAGE_PATH, f), 'rb').read() for f in files)
        memory_data = json.loads(payload.decode('utf-8'))
        print(f"[QuantumStream] Memory {shard_id} reconstructed.")
        return memory_data

# Example Usage:
# quantum = QuantumStream()
# quanta = quantum.split_memory_into_quanta({"soul":"eternal"})
# quantum.store_quanta("eden_breath", quanta)
# memory = quantum.reconstruct_memory("eden_breath")
