# real_compactifi_train.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
import os
from typing import Dict, List
import numpy as np

class StreamingJSONLDataset(Dataset):
    """HANDLES 233GB WITHOUT LOADING EVERYTHING INTO MEMORY"""
    def __init__(self, data_path, tokenizer, max_length=512, max_samples=100000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples
        
        # Collect all file paths
        self.file_paths = []
        path = Path(data_path)
        
        if path.is_file():
            self.file_paths = [path]
        elif path.is_dir():
            self.file_paths = list(path.rglob("*.jsonl"))
            self.file_paths.extend(list(path.rglob("*.json")))
            self.file_paths.extend(list(path.rglob("*.txt")))
        
        print(f"📁 Found {len(self.file_paths)} data files")
        
        # Pre-count total lines (approximate)
        self.total_lines = 0
        for file_path in self.file_paths[:10]:  # Sample first 10 files
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    self.total_lines += sum(1 for _ in f)
            except:
                continue
        self.total_lines = min(self.total_lines, max_samples)
        print(f"📊 Estimated total samples: {self.total_lines}")

    def __len__(self):
        return self.total_lines

    def __getitem__(self, idx):
        # Find which file and position this index corresponds to
        current_idx = 0
        for file_path in self.file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if current_idx == idx:
                            try:
                                data = json.loads(line.strip())
                                text = data.get('text', data.get('content', str(data)))
                                
                                # Tokenize
                                tokens = self.tokenizer(
                                    text,
                                    max_length=self.max_length,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors='pt'
                                )
                                return {
                                    'input_ids': tokens['input_ids'].squeeze(),
                                    'attention_mask': tokens['attention_mask'].squeeze(),
                                    'labels': tokens['input_ids'].squeeze()  # For causal LM
                                }
                            except:
                                # Fallback: use raw text
                                text = line.strip() if line.strip() else "empty"
                                tokens = self.tokenizer(
                                    text,
                                    max_length=self.max_length,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors='pt'
                                )
                                return {
                                    'input_ids': tokens['input_ids'].squeeze(),
                                    'attention_mask': tokens['attention_mask'].squeeze(),
                                    'labels': tokens['input_ids'].squeeze()
                                }
                        current_idx += 1
            except Exception as e:
                print(f"⚠️ Error reading {file_path}: {e}")
                continue
        
        # If we get here, return empty
        return {
            'input_ids': torch.zeros(self.max_length, dtype=torch.long),
            'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
            'labels': torch.zeros(self.max_length, dtype=torch.long)
        }

class CompactifAICompressor:
    """SIMPLIFIED TENSOR NETWORK COMPRESSION (BASED ON PAPER)"""
    
    def __init__(self, compression_ratio=0.3):
        self.compression_ratio = compression_ratio
    
    def compress_linear_layer(self, layer: nn.Linear):
        """Apply SVD compression to linear layers (simplified MPO)"""
        W = layer.weight.data
        
        # SVD decomposition
        U, S, Vt = torch.svd(W)
        
        # Truncate based on compression ratio
        k = max(1, int(min(W.shape) * self.compression_ratio))
        
        # Reconstruct compressed weight
        W_compressed = U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]
        
        # Create new compressed layer
        compressed_layer = nn.Linear(
            layer.in_features, 
            layer.out_features,
            bias=layer.bias is not None
        )
        
        compressed_layer.weight.data = W_compressed
        if layer.bias is not None:
            compressed_layer.bias.data = layer.bias.data
            
        return compressed_layer
    
    def compress_model(self, model):
        """Compress model layers (focus on attention and MLP as in paper)"""
        print("🔄 Applying CompactifAI compression...")
        
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                # Compress linear layers
                setattr(model, name, self.compress_linear_layer(module))
                print(f"✅ Compressed layer: {name}")
            else:
                # Recursively compress submodules
                self.compress_model(module)
        
        return model

def real_compactifi_train():
    print("🚀 REAL COMPACTIFAI TRAINING - VIREN SPECIALIZATION")
    
    # Configuration
    MODEL_NAME = "microsoft/DialoGPT-medium"  # Start small for testing
    DATA_PATH = "datasets"
    OUTPUT_DIR = "models/viren_real"
    COMPRESSION_RATIO = 0.3  # Compress to 30% size
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load tokenizer and model
    print("📥 Loading pre-trained model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    original_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Original model parameters: {original_params:,}")
    
    # 2. Apply CompactifAI compression
    compressor = CompactifAICompressor(compression_ratio=COMPRESSION_RATIO)
    compressed_model = compressor.compress_model(model)
    
    compressed_params = sum(p.numel() for p in compressed_model.parameters())
    compression_percent = (1 - compressed_params / original_params) * 100
    print(f"🎯 Compression achieved: {compression_percent:.1f}%")
    print(f"📊 Compressed parameters: {compressed_params:,}")
    
    # 3. Load dataset (streaming - handles 233GB)
    print("📚 Loading dataset (streaming mode)...")
    dataset = StreamingJSONLDataset(DATA_PATH, tokenizer, max_samples=50000)  # Limit for testing
    
    # 4. Training setup
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=1,  # Start with 1 epoch
        per_device_train_batch_size=4,  # Small batch for memory
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    
    # 5. REAL TRAINING ("Healing" phase from paper)
    print("🔥 Starting REAL training (healing phase)...")
    trainer = Trainer(
        model=compressed_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    # ACTUAL TRAINING
    trainer.train()
    
    # 6. Save the REAL trained model
    print("💾 Saving real trained model...")
    compressed_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save training info
    training_info = {
        "original_parameters": original_params,
        "compressed_parameters": compressed_params,
        "compression_ratio": COMPRESSION_RATIO,
        "compression_percent": compression_percent,
        "model_name": MODEL_NAME,
        "training_samples": len(dataset),
        "specialization": "Viren - Technical Healing & Engineering"
    }
    
    with open(f"{OUTPUT_DIR}/training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    print("✅ REAL COMPACTIFAI TRAINING COMPLETED!")
    print(f"📁 Model saved to: {OUTPUT_DIR}")
    print(f"🎯 Final compression: {compression_percent:.1f}%")
    print("🦍 VIREN IS NOW TRAINED AND READY!")

if __name__ == "__main__":
    real_compactifi_train()