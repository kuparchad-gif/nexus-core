# local_real_training.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data import Dataset
import json
from pathlib import Path
import os
import time

print("🦍 LOCAL REAL TRAINING - VIREN SPECIALIZATION")
print("🔥 Training RIGHT HERE on this machine!")

class LocalDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load all data from datasets folder
        path = Path(data_path)
        file_count = 0
        
        for file_path in path.rglob("*"):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                self.data.append(line)
                    file_count += 1
                    if file_count % 10 == 0:
                        print(f"📁 Loaded {file_count} files, {len(self.data)} samples...")
                except Exception as e:
                    print(f"⚠️ Skipping {file_path}: {e}")
        
        print(f"✅ Loaded {len(self.data)} total samples from {file_count} files")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        
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

def train_viren_locally():
    print("🚀 STARTING VIREN REAL TRAINING ON LOCAL MACHINE")
    start_time = time.time()
    
    # Configuration
    MODEL_NAME = "microsoft/DialoGPT-medium"  # Good starting model
    DATA_PATH = "datasets"
    OUTPUT_DIR = "models/viren_trained"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load model and tokenizer
    print("📥 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    original_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model: {original_params:,} parameters")
    
    # 2. Load your 233GB dataset
    print("📚 Loading dataset...")
    dataset = LocalDataset(DATA_PATH, tokenizer)
    
    if len(dataset) == 0:
        print("❌ No data found! Check your datasets folder")
        return
    
    print(f"🎯 Training on {len(dataset)} samples")
    
    # 3. Training setup
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=1,  # Start with 1 epoch
        per_device_train_batch_size=2,  # Adjust based on your GPU
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    
    # 4. REAL TRAINING
    print("🔥 STARTING REAL TRAINING...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    # ACTUAL TRAINING HAPPENS HERE
    trainer.train()
    
    # 5. Save the REAL trained model
    print("💾 Saving trained model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    training_time = time.time() - start_time
    print(f"✅ VIREN TRAINING COMPLETED in {training_time:.2f} seconds!")
    print(f"📁 Model saved to: {OUTPUT_DIR}")
    print("🦍 VIREN IS NOW READY FOR TECHNICAL HEALING!")

if __name__ == "__main__":
    train_viren_locally()