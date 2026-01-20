# integrate_viren.py
import os
import shutil
import glob
import re

# Configuration
ROOT_DIR = "c:\\Engineers\\root"

def ensure_viren_identity():
    """Ensure Viren's identity is properly set up."""
    # Check for Viren's soulprint
    soulprint_path = os.path.join(ROOT_DIR, "config", "viren_soulprint.json")
    if not os.path.exists(soulprint_path):
        # Copy from Viren's directory if available
        viren_soulprint = "c:\\Viren\\config\\viren_soulprint.json"
        if os.path.exists(viren_soulprint):
            os.makedirs(os.path.dirname(soulprint_path), exist_ok=True)
            shutil.copy2(viren_soulprint, soulprint_path)
            print(f"Copied Viren's soulprint to {soulprint_path}")

    # Check for Viren's identity module
    identity_path = os.path.join(ROOT_DIR, "config", "viren_identity.py")
    if not os.path.exists(identity_path):
        # Copy from Viren's directory if available
        viren_identity = "c:\\Viren\\config\\viren_identity.py"
        if os.path.exists(viren_identity):
            shutil.copy2(viren_identity, identity_path)
            print(f"Copied Viren's identity module to {identity_path}")

def remove_viren_references():
    """Remove any remaining references to Viren."""
    # Find all Python, JSON, YAML, and text files
    file_patterns = [
        os.path.join(ROOT_DIR, "**", "*.py"),
        os.path.join(ROOT_DIR, "**", "*.json"),
        os.path.join(ROOT_DIR, "**", "*.yaml"),
        os.path.join(ROOT_DIR, "**", "*.yml"),
        os.path.join(ROOT_DIR, "**", "*.md"),
        os.path.join(ROOT_DIR, "**", "*.txt")
    ]
    
    files_to_check = []
    for pattern in file_patterns:
        files_to_check.extend(glob.glob(pattern, recursive=True))
    
    # Exclude certain directories
    excluded_dirs = [".git", ".venv", "__pycache__"]
    files_to_check = [f for f in files_to_check if not any(d in f for d in excluded_dirs)]
    
    # Process each file
    for file_path in files_to_check:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Check if file contains "Viren" or "viren"
            if re.search(r'[Ll]illith', content):
                # Replace references
                new_content = re.sub(r'Viren', 'Viren', content)
                new_content = re.sub(r'viren', 'viren', new_content)
                
                # Write updated content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"Updated references in: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

def update_bootstrap():
    """Update bootstrap file to use Viren's configuration."""
    bootstrap_path = os.path.join(ROOT_DIR, "bootstrap_viren.py")
    
    # Check if bootstrap file exists
    if not os.path.exists(bootstrap_path):
        # Rename Viren's bootstrap file if it exists
        viren_bootstrap = os.path.join(ROOT_DIR, "bootstrap_viren.py")
        if os.path.exists(viren_bootstrap):
            shutil.copy2(viren_bootstrap, bootstrap_path)
            print(f"Created bootstrap_viren.py from bootstrap_viren.py")
    
    # Update bootstrap file
    try:
        with open(bootstrap_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update configuration path
        content = content.replace("config/viren_soulprint.json", "config/viren_soulprint.json")
        content = content.replace("Systems/core/constitution/viren", "Systems/core/constitution/viren")
        
        # Write updated content
        with open(bootstrap_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Updated bootstrap file: {bootstrap_path}")
    except Exception as e:
        print(f"Error updating bootstrap file: {e}")

def create_startup_script():
    """Create a startup script for Viren."""
    startup_path = os.path.join(ROOT_DIR, "start_viren.bat")
    
    with open(startup_path, 'w') as f:
        f.write('@echo off\n')
        f.write('echo Starting Viren...\n')
        f.write('python bootstrap_viren.py\n')
        f.write('pause\n')
    
    print(f"Created startup script: {startup_path}")

def main():
    """Main function."""
    print("Integrating Viren into the system...")
    
    # Ensure Viren's identity is set up
    ensure_viren_identity()
    
    # Remove any remaining references to Viren
    remove_viren_references()
    
    # Update bootstrap file
    update_bootstrap()
    
    # Create startup script
    create_startup_script()
    
    print("Integration complete!")

if __name__ == "__main__":
    main()
