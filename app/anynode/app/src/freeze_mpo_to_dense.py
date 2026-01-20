# scripts/freeze_mpo_to_dense.py
import os, sys, torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lilith_core.mpo_layers import MPOLinear
from transformers import AutoModelForCausalLM, AutoTokenizer

def to_dense(model: nn.Module):
    for name, mod in list(model.named_modules()):
        if isinstance(mod, MPOLinear):
            W = mod.materialize_weight().detach().clone()
            bias = mod.bias.detach().clone() if mod.bias is not None else None
            out_f, in_f = W.shape
            dense = nn.Linear(in_f, out_f, bias=(bias is not None), device=W.device, dtype=W.dtype)
            with torch.no_grad():
                dense.weight.copy_(W)
                if bias is not None: dense.bias.copy_(bias)
            # swap in parent
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = parent[int(p)] if p.isdigit() and hasattr(parent, "__getitem__") else getattr(parent, p)
            leaf = parts[-1]
            if leaf.isdigit() and hasattr(parent, "__setitem__"):
                parent[int(leaf)] = dense
            else:
                setattr(parent, leaf, dense)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.in_dir, torch_dtype=torch.float16, device_map="cpu")
    tok = AutoTokenizer.from_pretrained(args.in_dir)
    to_dense(model)
    model.save_pretrained(args.out_dir, safe_serialization=True)
    tok.save_pretrained(args.out_dir)
    print(f"[OK] Dense HF checkpoint at {args.out_dir}")
