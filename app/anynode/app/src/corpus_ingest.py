# File: root/scripts/corpus_ingest.py
# Purpose: Load and preprocess training corpus for all Engineers

import os

def ingest_training_corpus(corpus_dir):
    """Aggregate all .md and .txt training corpus files into a single string."""
    training_data = ""
    for root, dirs, files in os.walk(corpus_dir):
        for file in files:
            if file.endswith((".md", ".txt")):
                with open(os.path.join(root, file), encoding="utf-8") as f:
                    training_data += f.read() + "\n"
    return training_data

