import argparse, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lilith_core.convert_utils import replace_llama_modules_with_mpo

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--compress_attn", choices=["yes","no"], default="yes")
    ap.add_argument("--compress_mlp",  choices=["yes","no"], default="yes")
    ap.add_argument("--exclude_early_layers", type=int, default=0)
    ap.add_argument("--chi_base", type=int, default=96)
    ap.add_argument("--chi_base_attn", type=int, default=None)
    ap.add_argument("--chi_base_mlp", type=int, default=None)
    ap.add_argument("--ncores_attn", type=int, default=3)
    ap.add_argument("--ncores_mlp", type=int, default=4)
    ap.add_argument("--init_rank_cap", type=int, default=128)
    args = ap.parse_args()

    chi_attn = args.chi_base_attn if args.chi_base_attn else args.chi_base
    chi_mlp  = args.chi_base_mlp  if args.chi_base_mlp  else args.chi_base

    print(f"Loading: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    replaced = replace_llama_modules_with_mpo(
        model,
        compress_attn = args.compress_attn=="yes",
        compress_mlp  = args.compress_mlp=="yes",
        exclude_early_layers = args.exclude_early_layers,
        chi_base_attn = chi_attn,
        chi_base_mlp  = chi_mlp,
        ncores_attn   = args.ncores_attn,
        ncores_mlp    = args.ncores_mlp,
        init_rank_cap = args.init_rank_cap,
    )
    print(f"Replaced modules: {replaced}")

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")

if __name__ == "__main__":
    main()
