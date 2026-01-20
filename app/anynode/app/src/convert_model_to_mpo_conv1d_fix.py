# convert_model_to_mpo_conv1d_fix.py
# Drop-in converter that supports both nn.Linear and HF GPT-2-style Conv1D.
# Usage (from repo root):
#   python /path/to/convert_model_to_mpo_conv1d_fix.py --model_id sshleifer/tiny-gpt2 --output_dir ./checkpoints/tiny_mpo --device cpu --dtype f32 --dry_run
# You can also copy this file into your repo's scripts/ and run it from there.

import argparse, os, re, sys
import torch
from torch import nn
import numpy as np

from transformers import AutoModel, AutoModelForCausalLM, AutoConfig

# Allow running from anywhere by adding repo root if present
sys.path.insert(0, os.path.abspath(os.getcwd()))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lilith_core.mpo_layers import MPOLinear
from lilith_core.convert_utils import chi_list_from_graph

INCLUDE_DEFAULT = r"(attn|attention|mlp|proj|q_proj|k_proj|v_proj|o_proj|gate|up|down|dense|c_fc|c_proj|c_attn)"
EXCLUDE_DEFAULT = r"(embed|lm_head|norm|layernorm|ln|rope|rotary|position|token|classifier)"

def detect_arch(cfg) -> str:
    mt = (getattr(cfg, 'model_type', '') or '').lower()
    if 'gemma' in mt: return 'gemma'
    if 'llama' in mt or 'mistral' in mt: return 'llama'
    if mt in {'gpt2'}: return 'gpt2'
    if 'gpt_neox' in mt or 'neox' in mt: return 'gpt_neox'
    if 'falcon' in mt: return 'falcon'
    if mt in {'t5', 'ul2'}: return 't5'
    return 'generic'

def spectral_perm(dim: int):
    base = np.arange(13, dtype=int)
    reps = dim // len(base); rem = dim % len(base)
    tiled = np.concatenate([base for _ in range(reps)] + [base[:rem]]) if reps or rem else base.copy()
    idx = np.arange(dim, dtype=int)
    return idx[np.argsort(tiled, kind='mergesort')]

def get_parent_and_leaf(model: nn.Module, qualname: str):
    parent = model
    parts = qualname.split('.')
    for p in parts[:-1]:
        if p.isdigit() and hasattr(parent, '__getitem__'):
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    return parent, parts[-1]

def is_linear_like(m: nn.Module) -> bool:
    if isinstance(m, nn.Linear):
        return True
    cls = m.__class__.__name__.lower()
    if cls == 'conv1d':
        if hasattr(m, 'weight') and isinstance(m.weight, torch.Tensor) and m.weight.ndim == 2:
            return True
    return False

def get_io_shapes(m: nn.Module):
    if isinstance(m, nn.Linear):
        W = m.weight  # [out, in]
        return (W.shape[1], W.shape[0])
    # HF Conv1D: weight [in, out]
    W = m.weight
    return (W.shape[0], W.shape[1])

def get_weights_and_bias(m: nn.Module):
    if isinstance(m, nn.Linear):
        W = m.weight.data  # [out, in]
        b = m.bias.data.clone() if m.bias is not None else None
        return W, b, False
    # Conv1D: weight [in, out] -> transpose to [out, in]
    W = m.weight.data.t().contiguous()
    b = m.bias.data.clone() if getattr(m, 'bias', None) is not None else None
    return W, b, True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_id', required=True)
    ap.add_argument('--arch', default='auto')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--chi_base', type=int, default=96)
    ap.add_argument('--ncores_attn', type=int, default=3)
    ap.add_argument('--ncores_mlp', type=int, default=4)
    ap.add_argument('--init_rank_cap', type=int, default=128)
    ap.add_argument('--include_regex', default=INCLUDE_DEFAULT)
    ap.add_argument('--exclude_regex', default=EXCLUDE_DEFAULT)
    ap.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    ap.add_argument('--dtype',  choices=['auto','fp16','bf16','f32'], default='auto')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    cfg = AutoConfig.from_pretrained(args.model_id)
    arch = args.arch if args.arch != 'auto' else detect_arch(cfg)
    print(f"[INFO] arch={arch} model_type={cfg.model_type}")

    use_cuda = (args.device=='cuda') or (args.device=='auto' and torch.cuda.is_available())
    device_map = 'auto' if use_cuda else 'cpu'
    if args.dtype == 'fp16': torch_dtype = torch.float16
    elif args.dtype == 'bf16': torch_dtype = torch.bfloat16
    elif args.dtype == 'f32': torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16 if use_cuda else torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch_dtype, device_map=device_map)
    except Exception as e:
        print(f"[WARN] AutoModelForCausalLM load failed: {e}. Falling back to AutoModel.")
        model = AutoModel.from_pretrained(args.model_id, torch_dtype=torch_dtype, device_map=device_map)

    include_re = re.compile(args.include_regex, re.IGNORECASE)
    exclude_re = re.compile(args.exclude_regex, re.IGNORECASE)

    targeted = []
    replaced = 0
    for name, mod in list(model.named_modules()):
        if not is_linear_like(mod):
            continue
        lname = name.lower()
        if exclude_re.search(lname):
            continue
        if not include_re.search(lname):
            continue

        in_f, out_f = get_io_shapes(mod)
        is_attn = ('attn' in lname) or any(k in lname for k in ['q_proj','k_proj','v_proj','o_proj','c_attn','c_proj'])
        ncores = args.ncores_attn if is_attn else args.ncores_mlp
        targeted.append((name, in_f, out_f, 'attn' if is_attn else 'mlp'))
        if args.dry_run:
            continue

        W, b, _ = get_weights_and_bias(mod)
        perm_out = spectral_perm(out_f)
        perm_in  = spectral_perm(in_f)
        with torch.no_grad():
            Wp = W.index_select(0, torch.tensor(perm_out, device=W.device)) \
                 .index_select(1, torch.tensor(perm_in,  device=W.device))
            bp = b.index_select(0, torch.tensor(perm_out, device=b.device)) if b is not None else None

        chis = chi_list_from_graph(ncores, args.chi_base)
        mpo = MPOLinear(in_f, out_f, ncores, chis, bias=(b is not None),
                        device=Wp.device, dtype=Wp.dtype)
        with torch.no_grad():
            mpo.init_from_dense(Wp, max_rank=args.init_rank_cap)
            if bp is not None:
                mpo.bias.copy_(bp)

        parent, leaf = get_parent_and_leaf(model, name)
        if leaf.isdigit() and hasattr(parent, '__setitem__'):
            parent[int(leaf)] = mpo
        else:
            setattr(parent, leaf, mpo)
        replaced += 1

    for name, i, o, kind in targeted:
        print(f"  - {name:80s} | {kind} | W[{o},{i}]")
    if args.dry_run:
        print(f"[DRY-RUN] Would replace {len(targeted)} Linear/Conv1D modules.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    try:
        model.save_pretrained(args.output_dir, safe_serialization=True)
    except TypeError:
        model.save_pretrained(args.output_dir)
    print(f"[DONE] replaced={replaced} saved={args.output_dir}")

if __name__ == '__main__':
    main()