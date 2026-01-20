# === FILE: Systems/compactifi/shrinker.py ===
import os
import shutil
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch

def shrink_model(payload):
    print("[Compactifi Shrinker] Starting quantization/shrink...")

    model_path = payload.get("model_path")
    quant = payload.get("quantization", "int4")
    output_dir = payload.get("output_dir", "./compactifi_output")
    strategy = payload.get("strategy", "auto")

    os.makedirs(output_dir, exist_ok=True)

    try:
        print(f"🔍 Loading model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16 if quant == "int4" else torch.float32,
            load_in_4bit=True if quant == "int4" else False
        )

        print("📦 Saving quantized model...")
        out_path = os.path.join(output_dir, os.path.basename(model_path) + f"_{quant}_shrunk")
        model.save_pretrained(out_path)

        return {
            "status": "shrunk",
            "original_model": model_path,
            "compressed_model": out_path,
            "quantization": quant,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        return {"error": str(e), "status": "failed"}


# === FILE: Systems/compactifi/trainer.py ===
import os
from datetime import datetime

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch

def train_model(payload):
    print("[Compactifi Trainer] Beginning LoRA fine-tuning...")

    base_model = payload.get("base_model")
    dataset_path = payload.get("dataset_path")
    output_model_name = payload.get("output_model_name", "finetuned-model")
    output_dir = payload.get("output_dir", "./compactifi_output")
    epochs = int(payload.get("epochs", 3))
    lr = float(payload.get("learning_rate", 5e-5))
    batch_size = int(payload.get("batch_size", 4))

    try:
        print("📥 Loading model & tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token

        print("🧠 Preparing for 4-bit LoRA training...")
        model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)

        print(f"📚 Loading dataset from {dataset_path}")
        dataset = load_dataset("json", data_files=dataset_path)["train"]

        def tokenize(sample):
            return tokenizer(sample["text"], padding="max_length", truncation=True, max_length=512)

        tokenized = dataset.map(tokenize, batched=True)

        args = TrainingArguments(
            output_dir=os.path.join(output_dir, output_model_name),
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=lr,
            save_total_limit=1,
            save_strategy="epoch",
            logging_dir="./logs",
            fp16=True,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized,
            tokenizer=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )

        print("🚀 Starting training...")
        trainer.train()
        model.save_pretrained(os.path.join(output_dir, output_model_name))

        return {
            "status": "trained",
            "base_model": base_model,
            "dataset": dataset_path,
            "output_model": os.path.join(output_dir, output_model_name),
            "epochs": epochs,
            "learning_rate": lr,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        return {"status": "failed", "error": str(e)}