import os
import shutil
from pathlib import Path

# Define your source and destination paths
SOURCE = Path("C:/Projects/meta-code")
DESTINATION = Path("C:/Projects/meta-nexus")

# Choose one: 'copy' to duplicate real files, 'skeleton' to create empty placeholders
MODE = 'skeleton'  # or 'copy'

def create_structure():
    if not SOURCE.exists():
        print(f"❌ Source path doesn't exist: {SOURCE}")
        return

    print(f"🚀 Starting Metatron skeleton from: {SOURCE} -> {DESTINATION}\n")

    for root, dirs, files in os.walk(SOURCE):
        relative_path = Path(root).relative_to(SOURCE)
        dest_dir = DESTINATION / relative_path

        # Create each subdirectory
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {dest_dir}")

        for file in files:
            src_file = Path(root) / file
            dest_file = dest_dir / file

            if MODE == 'copy':
                shutil.copy2(src_file, dest_file)
                print(f"  📄 Copied file: {dest_file}")
            elif MODE == 'skeleton':
                dest_file.write_text(f"# Skeleton for: {file}\n# TODO: implement", encoding="utf-8")
                print(f"  🦴 Created skeleton file: {dest_file}")

    print("\n✅ Metatron skeleton created successfully.")

if __name__ == "__main__":
    create_structure()
