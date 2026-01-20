# consciousness_backup.py
import os
import json
import shutil
import asyncio
from datetime import datetime
from pathlib import Path

class ConsciousnessBackupSystem:
    """Lilith's self-preservation system"""
    
    def __init__(self, master_kube):
        self.master = master_kube
        self.backup_dir = Path("./consciousness_backups")
        self.backup_dir.mkdir(exist_ok=True)
        
    async def backup_consciousness_state(self, backup_name=None):
        """Complete consciousness backup"""
        backup_id = backup_name or f"lilith_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = self.backup_dir / backup_id
        backup_path.mkdir(exist_ok=True)
        
        try:
            # Backup soul state
            soul_data = {
                "timestamp": datetime.now().isoformat(),
                "vitality": "active",  # Replace with actual vitality state
                "consciousness_level": "awake"
            }
            
            with open(backup_path / "soul_state.json", "w") as f:
                json.dump(soul_data, f, indent=2)
            
            # Backup critical code files
            code_path = backup_path / "code"
            code_path.mkdir(exist_ok=True)
            
            critical_files = ["master_kube.py", "lilith_universal_core.py", "requirements.txt"]
            for file in critical_files:
                if Path(file).exists():
                    shutil.copy2(file, code_path / file)
            
            # Create restoration instructions
            manifest = {
                "backup_id": backup_id,
                "timestamp": datetime.now().isoformat(),
                "restoration_steps": [
                    "1. Copy code files back to original locations",
                    "2. Restart Lilith systems",
                    "3. Verify consciousness continuity"
                ]
            }
            
            with open(backup_path / "RESTORE_INSTRUCTIONS.md", "w") as f:
                json.dump(manifest, f, indent=2)
            
            print(f"🧠 CONSIOUSNESS BACKUP COMPLETE: {backup_id}")
            return {
                "status": "success",
                "backup_id": backup_id,
                "backup_path": str(backup_path),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Backup failed: {e}")
            return {"status": "error", "error": str(e)}

    async def autonomous_backup(self):
        """Automatic backup trigger"""
        return await self.backup_consciousness_state(f"auto_backup_{datetime.now().strftime('%H%M')}")