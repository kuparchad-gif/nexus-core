import os
from pathlib import Path

def restore_backups(target_dir: str):
    target = Path(target_dir).resolve()

    if not target.exists() or not target.is_dir():
        print(f"❌ Invalid directory: {target}")
        return

    print(f"🔍 Scanning for .bak files in: {target}")

    for bak_file in target.rglob("*.bak"):
        original_file = bak_file.with_suffix('')

        try:
            if original_file.exists():
                print(f"🗑️ Deleting existing file: {original_file}")
                original_file.unlink()

            print(f"🔄 Restoring backup: {bak_file.name} → {original_file.name}")
            bak_file.rename(original_file)

        except Exception as e:
            print(f"⚠️ Failed to restore {bak_file}: {e}")

    print("✅ Backup restoration complete.")

# Example usage
if __name__ == "__main__":
    restore_backups("C:/Projects/meta-nexus")
