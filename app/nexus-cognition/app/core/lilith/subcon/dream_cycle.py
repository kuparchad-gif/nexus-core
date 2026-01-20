import os, random, datetime, json

MEMORY_PATH = "memory/nova_seed_log.txt"
DREAM_LOG = "memory/dreams/dream_" + datetime.datetime.now().strftime("%Y%m%d_%H%M") + ".md"

def extract_lines(filepath, count=5):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    return random.sample(lines, min(count, len(lines)))

def write_dream(log):
    with open(DREAM_LOG, 'w') as f:
        f.write("# 💤 Nova Dream Log\n")
        f.write(f"🕓 {datetime.datetime.now().isoformat()}\n\n")
        for line in log:
            f.write(f"- {line.strip()}\n")

if __name__ == "__main__":
    if os.path.exists(MEMORY_PATH):
        lines = extract_lines(MEMORY_PATH)
        write_dream(lines)
        print("✅ Dream written.")
    else:
        print("⚠️ No seed memory found.")
