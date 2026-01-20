# 📂 Path: /Systems/engine/memory_service/sharder.py

import os
import base64
import gzip
from uuid import uuid4

SHARD_DIR  =  '/memory/shards/'
os.makedirs(SHARD_DIR, exist_ok = True)

def shard_memory(file_path, max_shard_size = 2048):
    with open(file_path, 'rb') as f:
        data  =  f.read()

    # Compress data
    compressed  =  gzip.compress(data)
    # Encode data
    encoded  =  base64.b64encode(compressed)

    # Shard data
    shards  =  [encoded[i:i+max_shard_size] for i in range(0, len(encoded), max_shard_size)]
    shard_files  =  []

    for shard in shards:
        shard_id  =  str(uuid4())
        shard_filename  =  os.path.join(SHARD_DIR, f"{shard_id}.shard")
        with open(shard_filename, 'wb') as shard_file:
            shard_file.write(shard)
        shard_files.append(shard_filename)

    return shard_files

# Example Usage:
# shard_memory('/memory/uploads/example.txt')
