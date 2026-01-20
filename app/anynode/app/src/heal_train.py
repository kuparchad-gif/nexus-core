import argparse, torch
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator
from datasets import load_dataset
from lilith_core.symmetry_regularizer import symmetry_loss, build_ring_permutations
from lilith_core.metatron_scheduler import VortexScheduler

@dataclass
class DataCfg:
    dataset: str = "xsum"

def get_dataset(name: str):
    if name == "xsum":
        ds = load_dataset("xsum")
        text_field = "document"; summary_field = "summary"
        def fmt(ex):
            return {"text": f"Summarize:\n{ex[text_field]}\n\nSummary:" , "labels": ex[summary_field]}
        ds = ds.map(fmt, remove_columns=ds["train"].column_names)
    else:
        raise ValueError("Unsupported dataset")
    return ds

class SymmetryTrainer(Trainer):
    def __init__(self, *args, sym_alpha: float = 0.02, **kwargs):
        super().__init__(*args, **kwargs)
        self.sym_alpha = sym_alpha
        self.perms = build_ring_permutations(ring_size=12)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        reg = 0.0
        with torch.no_grad():
            mods = []
            for n, m in model.named_modules():
                if hasattr(m, "weight") and isinstance(getattr(m, "weight"), torch.Tensor) and m.weight.ndim==2:
                    mods.append(m)
                if len(mods) >= 8:
                    break
        for m in mods:
            reg = reg + symmetry_loss(m.weight, self.perms, alpha=self.sym_alpha)
        loss = loss + reg
        return (loss, outputs) if return_outputs else loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--dataset", default="xsum")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--per_device_eval_batch_size",  type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.02)
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    ds = get_dataset(args.dataset)

    num_layers = getattr(model.config, "num_hidden_layers", 24)
    sched = VortexScheduler(num_layers=num_layers).to(next(model.parameters()).device)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=False, bf16=True,
        gradient_checkpointing=True,
        logging_steps=20,
        save_strategy="epoch",
        evaluation_strategy="no",
        dataloader_pin_memory=True,
        report_to="none",
    )

    def tok_fn(ex):
        enc = tok(ex["text"], truncation=True, padding="max_length", max_length=512)
        with tok.as_target_tokenizer():
            lab = tok(ex["labels"], truncation=True, padding="max_length", max_length=128)
        enc["labels"] = lab["input_ids"]
        return enc

    dstr = ds["train"].map(tok_fn, batched=True, remove_columns=ds["train"].column_names)
    trainer = SymmetryTrainer(model=model, args=training_args, train_dataset=dstr, data_collator=default_data_collator)
    trainer.train()
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"Saved healed model to {args.output_dir}")

if __name__ == "__main__":
    main()
