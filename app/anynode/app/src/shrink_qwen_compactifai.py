# shrink_qwen_compactifai.py
import os, math
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from .compactifai_mpo import LayerCompressor

ATTN_KEYS = ("q_proj","k_proj","v_proj","o_proj","out_proj")
MLP_KEYS  = ("up_proj","gate_proj","down_proj","fc1","fc2")

def schedule_rank_frac(block_idx, total_blocks, base=0.12, early_safe=6, late_strong=16):
    if block_idx < early_safe: return base * 0.5
    if block_idx >= total_blocks - late_strong: return base * 0.5
    return base

def compress_qwen_modules(model: nn.Module, total_blocks=32, rank_max=192, base_frac=0.12,
                          early_safe=6, late_strong=16) -> int:
    replaced = 0
    for name, module in model.named_modules():
        if not name.startswith("model.layers."):
            continue
        try:
            i = int(name.split(".")[2])
        except:
            continue
        frac = schedule_rank_frac(i, total_blocks, base=base_frac, early_safe=early_safe, late_strong=late_strong)
        C = LayerCompressor(rank_max=rank_max, rank_frac=frac)

        keyhit = any(k in name for k in ATTN_KEYS) or any(k in name for k in MLP_KEYS)
        if not keyhit:
            continue

        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                new_mod = C.compress_linear(child)
                if new_mod is not child:
                    setattr(module, child_name, new_mod)
                    replaced += 1
    return replaced

def run_shrink(src_hf: str, dst_hf: str, blocks=32, rank_max=192, rank_frac=0.12, early_safe=6, late_strong=16):
    model = AutoModelForCausalLM.from_pretrained(src_hf, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.eval()
    n = compress_qwen_modules(model, total_blocks=blocks, rank_max=rank_max, base_frac=rank_frac,
                              early_safe=early_safe, late_strong=late_strong)
    os.makedirs(dst_hf, exist_ok=True)
    model.save_pretrained(dst_hf, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(src_hf, use_fast=True)
    tok.save_pretrained(dst_hf)
    return n