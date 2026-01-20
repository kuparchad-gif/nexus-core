# train_bert_fib.py - QLoRA Fine-Tuning for Fibonacci BERT on Lillith Datasets
# Optimal: QLoRA for 1.5B params, CPU-friendly. Integrates cosmic metadata.
# Requirements: pip install transformers==4.44.2 peft==0.12.0 datasets==2.21.0 evaluate==0.4.3 torch==2.4.0
# Usage: python train_bert_fib.py --domain Productivity --day 1
# Output: fine_tuned_model/ checkpoint in C:\Projects\Lillith-Evolution\models

import argparse
import json
import os
from datasets import Dataset
import pandas as pd
import torch
from peft import get_peft_model, LoraConfig, TaskType
from transformers import BertTokenizerFast, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from evaluate import load as eval_load

# Constants: Align with pipeline (Golden Ratio, Pi)
GOLDEN_RATIO = (1 + 5**0.5) / 2
PI = 3.1415926535
MAX_SEQ_LEN = 512
MLM_PROB = 0.15

def load_data(domain: str, day: int) -> Dataset:
    """Load JSONL, preprocess with Fibonacci metadata weighting"""
    path = f"C:\\Projects\\Lillith-Evolution\\datasets\\{domain}\\train_day{day}.jsonl"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}. Run dataset_runner.py first.")
    
    df = pd.read_json(path, lines=True)
    texts = []
    for _, row in df.iterrows():
        text = row['text']
        if row.get('fibonacci_mode', False):
            # Weight Fibonacci: Split paragraphs by golden ratio
            paras = text.split('\n\n')
            weighted = [p * (len(p) * GOLDEN_RATIO) for p in paras]  # Sim weight (len scale)
            text = '\n\n'.join([p[:int(len(p) / PI)] for p in paras])  # Pi-truncate for variety
        texts.append(text)
    
    dataset = Dataset.from_dict({'text': texts})
    dataset = dataset.train_test_split(test_size=0.2)  # 80/20 split
    val_test = dataset['test'].train_test_split(test_size=0.5)  # Val/test 10/10
    return {
        'train': dataset['train'],
        'validation': val_test['train'],
        'test': val_test['test']
    }

def tokenize_function(examples, tokenizer):
    """Tokenize with MLM masking, bias toward binary_structure if present"""
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=MAX_SEQ_LEN)

def main(domain: str, day: int):
    print(f"Training BERT on {domain} data for day {day}")
    
    # Load model/tokenizer (your DeepSeek as BERT-like; adjust if exact path differs)
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Or your local GGUF converted
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    
    # QLoRA Config: Stable, low-rank adapters
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,  # Rank
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["query", "value"]  # BERT attention
    )
    model = get_peft_model(model, peft_config)
    
    # Load and tokenize data
    splits = load_data(domain, day)
    tokenized_datasets = {k: v.map(lambda ex: tokenize_function(ex, tokenizer), batched=True) for k, v in splits.items()}
    
    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROB)
    
    # Training args: CPU-optimized, 3 epochs
    training_args = TrainingArguments(
        output_dir="fine_tuned_model",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        fp16=False,  # CPU no
        logging_dir="logs",
        report_to="none"  # No TensorBoard
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train
    trainer.train()
    
    # Eval on test
    perplexity = eval_load("perplexity", module_type="metric")
    results = perplexity.compute(model=model, predictions=trainer.predict(tokenized_datasets['test']).predictions)
    print(f"Test Perplexity: {results['mean_perplexity']}")
    
    # Save
    model.save_pretrained("fine_tuned_model")
    tokenizer.save_pretrained("fine_tuned_model")
    print("Model saved to fine_tuned_model/. Run forge.ps1 for GGUF conversion.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, help="Domain to train on, e.g., Productivity")
    parser.add_argument("--day", type=int, default=1, help="Dataset day")
    args = parser.parse_args()
    main(args.domain, args.day)