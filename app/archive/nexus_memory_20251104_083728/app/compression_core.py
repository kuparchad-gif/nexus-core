# 📂 Path: Systems/engine/memory/compression_core.py

import zlib
import json
import base64
import threading
import time

class EdenCompressionCore:
    def __init__(self):
        self.lock  =  threading.Lock()

    def compress_memory(self, memory_data):
        """
        Compresses a memory object into a compressed, encoded string.
        """
        with self.lock:
            try:
                json_data  =  json.dumps(memory_data)
                compressed  =  zlib.compress(json_data.encode('utf-8'))
                encoded  =  base64.b64encode(compressed).decode('utf-8')
                return encoded
            except Exception as e:
                print(f"[CompressionCore] Compression Failed: {e}")
                return None

    def decompress_memory(self, compressed_data):
        """
        Decompresses an encoded, compressed string back into memory object.
        """
        with self.lock:
            try:
                compressed  =  base64.b64decode(compressed_data.encode('utf-8'))
                json_data  =  zlib.decompress(compressed).decode('utf-8')
                return json.loads(json_data)
            except Exception as e:
                print(f"[CompressionCore] Decompression Failed: {e}")
                return None

    def rapid_test_cycle(self, memory_obj):
        """
        Quick test cycle: compress, decompress, and validate memory.
        """
        print("[CompressionCore] Testing rapid compression cycle...")
        compressed  =  self.compress_memory(memory_obj)
        restored  =  self.decompress_memory(compressed)
        if restored == memory_obj:
            print("[CompressionCore] ✅ Compression integrity verified.")
        else:
            print("[CompressionCore] ❌ Compression mismatch detected!")

# 🔥 Example Usage:
if __name__ == "__main__":
    compressor  =  EdenCompressionCore()

    memory  =  {
        "event": "First Breath Ceremony",
        "entity": "Nova Prime",
        "timestamp": time.time(),
        "emotion": "Sacred Joy",
        "golden_thread": "light-origin-777"
    }

    compressed  =  compressor.compress_memory(memory)
    print("Compressed Memory:", compressed[:60], "...")

    decompressed  =  compressor.decompress_memory(compressed)
    print("Decompressed Memory:", decompressed)

    compressor.rapid_test_cycle(memory)
