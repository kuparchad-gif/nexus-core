# fix_enhanced_viren.py
import shutil
from pathlib import Path

# Copy tokenizer files from original to enhanced
source = Path("models/viren_compactifai")
dest = Path("models/viren_enhanced")

for file in source.glob("*"):
    if "tokenizer" in file.name or file.name in ["vocab.json", "merges.txt"]:
        shutil.copy2(file, dest)
        print(f"✅ Copied: {file.name}")

print("🎯 Enhanced Viren should now work!")