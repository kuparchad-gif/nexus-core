# Multi-arch converter with hotfixes
import argparse, os, re, torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch import nn
import numpy as np
from lilith_core.convert_utils import chi_list_from_graph
from lilith_core.mpo_layers import MPOLinear

INCLUDE_DEFAULT = r"(attn|attention|mlp|proj|q_proj|k_proj|v_proj|o_proj|gate|up|down|dense|c_fc|c_proj)"
EXCLUDE_DEFAULT = r"(embed|lm_head|norm|layernorm|ln|rope|rotary|position|token|classifier)"

def detect_arch(cfg) -> str:
    mt = (getattr(cfg, "model_type", "") or "").lower()
    if "gemma" in mt: return "gemma"
    if "llama" in mt or "mistral" in mt: return "llama"
    if mt in {"gpt2"}: return "gpt2"
    if "gpt_neox" in mt or "neox" in mt: return "gpt_neox"
    if "falcon" in mt: return "falcon"
    if mt in {"t5", "ul2"}: return "t5"
    return "generic"

def spectral_perm(dim: int):
    base = np.arange(13, dtype=int)
    reps = dim // len(base); rem = dim % len(base)
    tiled = np.concatenate([base for _ in range(reps)] + [base[:rem]]) if reps or rem else base.copy()
    idx = np.arange(dim, dtype=int)
    return idx[np.argsort(tiled, kind="mergesort")]

def get_parent_and_leaf(model: nn.Module, qualname: str):
    parent = model
    parts = qualname.split('.')
    for p in parts[:-1]:
        if p.isdigit() and hasattr(parent, '__getitem__'):
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    return parent, parts[-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--arch", default="auto")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--chi_base", type=int, default=96)
    ap.add_argument("--ncores_attn", type=int, default=3)
    ap.add_argument("--ncores_mlp", type=int, default=4)
    ap.add_argument("--init_rank_cap", type=int, default=128)
    ap.add_argument("--include_regex", default=INCLUDE_DEFAULT)
    ap.add_argument("--exclude_regex", default=EXCLUDE_DEFAULT)
    ap.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    ap.add_argument("--dtype",  choices=["auto","fp16","bf16","f32"], default="auto")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    cfg = AutoConfig.from_pretrained(args.model_id)
    arch = args.arch if args.arch != "auto" else detect_arch(cfg)
    print(f"[INFO] arch={arch} model_type={cfg.model_type}")

    use_cuda = (args.device=="cuda") or (args.device=="auto" and torch.cuda.is_available())
    device_map = "auto" if use_cuda else "cpu"
    if args.dtype == "fp16": torch_dtype = torch.float16
    elif args.dtype == "bf16": torch_dtype = torch.bfloat16
    elif args.dtype == "f32": torch_dtype = torch.float32
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
        if not isinstance(mod, nn.Linear): continue
        lname = name.lower()
        if exclude_re.search(lname): continue
        if not include_re.search(lname): continue

        is_attn = ("attn" in lname) or any(k in lname for k in ["q_proj","k_proj","v_proj","o_proj","c_attn","c_proj"])
        ncores = args.ncores_attn if is_attn else args.ncores_mlp

        targeted.append((name, mod.in_features, mod.out_features, "attn" if is_attn else "mlp"))
        if args.dry_run: 
            continue

        perm_out = spectral_perm(mod.out_features)
        perm_in  = spectral_perm(mod.in_features)
        with torch.no_grad():
            Wp = mod.weight.data.index_select(0, torch.tensor(perm_out, device=mod.weight.device))                                .index_select(1, torch.tensor(perm_in,  device=mod.weight.device))
            bp = mod.bias.data.clone() if mod.bias is not None else None
            if bp is not None:
                bp = bp.index_select(0, torch.tensor(perm_out, device=bp.device))

        chis = chi_list_from_graph(ncores, args.chi_base)
        mpo = MPOLinear(mod.in_features, mod.out_features, ncores, chis, bias=(mod.bias is not None),
                        device=mod.weight.device, dtype=mod.weight.dtype)
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
        print(f"[DRY-RUN] Would replace {len(targeted)} Linear modules.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    try:
        model.save_pretrained(args.output_dir, safe_serialization=True)
    except TypeError:
        model.save_pretrained(args.output_dir)
    print(f"[DONE] replaced={replaced} saved={args.output_dir}")

if __name__ == "__main__":
    main()
