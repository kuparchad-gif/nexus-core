#!/usr/bin/env python3
"""
CREDENTIAL UPDATE UTILITY
Run this script to update any service credential in one place.
All changes propagate through the entire system automatically.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class CredentialManager:
    """Manages all service credentials from a single source"""
    
    def __init__(self, blueprint_path: str = "genesis_seed_blueprint.py"):
        self.blueprint_path = Path(blueprint_path)
        self.backup_dir = Path("credential_backups")
        self.backup_dir.mkdir(exist_ok=True)
        
        # Load current configuration
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load the current configuration"""
        if not self.blueprint_path.exists():
            print(f"❌ Blueprint not found: {self.blueprint_path}")
            sys.exit(1)
            
        # Import the config dynamically
        import importlib.util
        spec = importlib.util.spec_from_file_location("blueprint", self.blueprint_path)
        blueprint = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(blueprint)
        
        return blueprint.NEXUS_CONFIG
    
    def _backup_current(self) -> Path:
        """Create a timestamped backup before making changes"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"genesis_seed_blueprint_{timestamp}.py"
        
        shutil.copy2(self.blueprint_path, backup_path)
        print(f"✅ Backup created: {backup_path}")
        return backup_path
    
    def update_credential(self, service: str, credential_type: str, new_value: str) -> bool:
        """
        Update a specific credential for a service
        
        Examples:
            update_credential("chroma", "api_key", "new-key-here")
            update_credential("mongodb", "connection_string", "new-uri", cluster_index=0)
        """
        # Create backup first
        self._backup_current()
        
        # Read the current file
        with open(self.blueprint_path, 'r') as f:
            content = f.read()
        
        # Update based on service type
        if service == "mongodb" and credential_type == "connection_string":
            # Handle MongoDB cluster updates
            import re
            pattern = r'("connection_string":\s*")[^"]*(")'
            replacement = f'\\1{new_value}\\2'
            
            # Find the specific cluster
            lines = content.split('\n')
            cluster_count = 0
            for i, line in enumerate(lines):
                if '"connection_string"' in line and cluster_count == 0:  # Update first cluster
                    lines[i] = re.sub(pattern, replacement, line)
                    break
                elif '"connection_string"' in line:
                    cluster_count += 1
            
            content = '\n'.join(lines)
            
        else:
            # Generic update pattern
            import re
            pattern = rf'("{credential_type}":\s*")[^"]*(")'
            content = re.sub(pattern, f'\\1{new_value}\\2', content)
        
        # Write back
        with open(self.blueprint_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Updated {service}.{credential_type}")
        
        # Propagate to environment if needed
        self._update_env_file()
        
        return True
    
    def update_multiple(self, updates: Dict[str, Dict[str, str]]) -> bool:
        """Update multiple credentials at once"""
        backup = self._backup_current()
        
        try:
            for service, creds in updates.items():
                for cred_type, value in creds.items():
                    self.update_credential(service, cred_type, value)
            print("✅ All credentials updated successfully")
            return True
        except Exception as e:
            print(f"❌ Update failed: {e}")
            print(f"🔄 Restoring from backup: {backup}")
            shutil.copy2(backup, self.blueprint_path)
            return False
    
    def _update_env_file(self) -> None:
        """Update .env file with current credentials"""
        env_vars = []
        
        # MongoDB
        if 'mongodb' in self.config['services']:
            mongo_uri = self.config['services']['mongodb']['clusters'][0]['connection_string']
            env_vars.append(f"MONGODB_URI={mongo_uri}")
        
        # Chroma
        if 'chroma' in self.config['services']:
            env_vars.append(f"CHROMA_API_KEY={self.config['services']['chroma']['api_key']}")
        
        # Supabase
        if 'supabase' in self.config['services']:
            env_vars.append(f"SUPABASE_KEY={self.config['services']['supabase']['api_key']}")
        
        # Netlify
        if 'netlify' in self.config['services']:
            env_vars.append(f"NETLIFY_TOKEN={self.config['services']['netlify']['access_token']}")
        
        # Write .env file
        with open('.env', 'w') as f:
            f.write('\n'.join(env_vars))
        
        print("✅ .env file updated")
    
    def list_services(self) -> None:
        """List all configured services and their status"""
        print("\n📋 Configured Services:")
        print("=" * 60)
        
        for service, config in self.config['services'].items():
            status = "✅ ENABLED" if config.get('enabled', False) else "❌ DISABLED"
            print(f"\n{service.upper()}:")
            print(f"  Status: {status}")
            print(f"  Provider: {config.get('provider', 'N/A')}")
            print(f"  Email: {config.get('email', 'N/A')}")
            
            # Show masked credentials
            for key in ['api_key', 'access_token', 'connection_string']:
                if key in config:
                    value = config[key]
                    if isinstance(value, str) and len(value) > 8:
                        masked = value[:4] + "..." + value[-4:]
                        print(f"  {key}: {masked}")
                    elif key == 'connection_string' and isinstance(config, dict):
                        # Handle MongoDB clusters
                        pass
            
            if service == 'mongodb':
                for i, cluster in enumerate(config.get('clusters', [])):
                    uri = cluster['connection_string']
                    masked = uri[:20] + "..." + uri[-10:]
                    print(f"  cluster_{i+1}: {masked}")
    
    def export_to_json(self, output_path: str = "credentials.json") -> None:
        """Export credentials to JSON (for other tools)"""
        # Create a safe copy (mask sensitive data or export as-is)
        export_data = {
            "services": self.config['services'],
            "memory": self.config['memory'],
            "agents": self.config['agents']
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ Credentials exported to {output_path}")

def main():
    """Interactive credential management"""
    manager = CredentialManager()
    
    print("\n" + "="*60)
    print("🔑 NEXUS CREDENTIAL MANAGER")
    print("="*60)
    
    while True:
        print("\nCommands:")
        print("  list              - Show all configured services")
        print("  update <service>  - Update a service credential")
        print("  batch             - Batch update multiple credentials")
        print("  export            - Export to JSON")
        print("  quit              - Exit")
        
        cmd = input("\n⚡ ").strip().lower()
        
        if cmd == "list":
            manager.list_services()
        
        elif cmd.startswith("update"):
            parts = cmd.split()
            if len(parts) < 2:
                print("Usage: update <service>")
                continue
            
            service = parts[1]
            if service not in manager.config['services']:
                print(f"❌ Service '{service}' not found")
                continue
            
            print(f"\nUpdating {service}:")
            cred_type = input("  Credential type (api_key/access_token/etc): ").strip()
            new_value = input("  New value: ").strip()
            
            if cred_type and new_value:
                manager.update_credential(service, cred_type, new_value)
        
        elif cmd == "batch":
            print("\nBatch update (enter as JSON):")
            print('Example: {"chroma": {"api_key": "new-key"}, "netlify": {"access_token": "new-token"}}')
            try:
                updates = json.loads(input("  Updates: "))
                manager.update_multiple(updates)
            except json.JSONDecodeError:
                print("❌ Invalid JSON")
        
        elif cmd == "export":
            manager.export_to_json()
        
        elif cmd in ["quit", "q", "exit"]:
            break

if __name__ == "__main__":
    main()