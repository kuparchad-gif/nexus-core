"""
memory_defragger.py
Location: C:/Viren/Systems/memory/
Cleans up old or malformed vector/memory packets.
"""

import os
import time

def defrag_memory(folder, max_age_seconds = 86400):
    now  =  time.time()
    removed  =  0
    for filename in os.listdir(folder):
        file_path  =  os.path.join(folder, filename)
        if os.path.isfile(file_path) and now - os.path.getmtime(file_path) > max_age_seconds:
            os.remove(file_path)
            removed + =  1
    print(f"[DEFRAG] Removed {removed} stale entries from {folder}")

if __name__ == "__main__":
    defrag_memory("C:/Viren/memory/packets")
