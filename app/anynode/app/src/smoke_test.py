# Minimal smoke test (CPU OK) using tiny-gpt2
import subprocess, sys

def run(cmd):
    print("+", " ".join(cmd))
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(r.stdout)
    r.check_returncode()

if __name__ == "__main__":
    cmd = [sys.executable, "scripts/convert_model_to_mpo.py",
           "--model_id", "sshleifer/tiny-gpt2",
           "--output_dir", "./checkpoints/tiny_mpo",
           "--device", "cpu", "--dtype", "f32",
           "--dry_run"]
    run(cmd)
    cmd.remove("--dry_run")
    run(cmd)
    print("OK: converted tiny model -> ./checkpoints/tiny_mpo")
