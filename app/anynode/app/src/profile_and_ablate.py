import argparse, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model_dir", required=True)
  ap.add_argument("--seq", type=int, default=256)
  ap.add_argument("--warmup", type=int, default=10)
  ap.add_argument("--iters", type=int, default=50)
  args = ap.parse_args()

  model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, device_map="auto")
  tok = AutoTokenizer.from_pretrained(args.model_dir)
  device = next(model.parameters()).device

  dummy = tok("Sanity check.", return_tensors="pt").to(device)
  model.eval()
  with torch.no_grad():
      for _ in range(args.warmup):
          _ = model.generate(**dummy, max_length=args.seq)

  if torch.cuda.is_available(): torch.cuda.synchronize()
  t0 = time.time()
  with torch.no_grad():
      for _ in range(args.iters):
          _ = model.generate(**dummy, max_length=args.seq)
  if torch.cuda.is_available(): torch.cuda.synchronize()
  dt = time.time() - t0
  itps = args.iters / dt
  print(f"iters/sec: {itps:.3f}, avg gen time: {dt/args.iters:.4f}s")

if __name__ == "__main__":
  main()
