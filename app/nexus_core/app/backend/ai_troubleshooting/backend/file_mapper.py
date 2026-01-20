# file_mapper.py
import shutil
import os
from pathlib import Path
import json

# Define your source -> destination mapping
FILE_MAPPING = {
    # AcidemiKubes components
    r"C:\project-root\30_build\ai-troubleshooter\backend\AcidemiKubes": "AcidemiKubes",
    
    # Compression engine  
    r"C:\project-root\30_build\ai-troubleshooter\backend\CompressionEngine": "CompressionEngine",
    
    # Configuration files
    r"C:\project-root\30_build\ai-troubleshooter\backend\config": "config",
    
    # Training datasets
    r"C:\project-root\30_build\ai-troubleshooter\backend\datasets": "SoulData/library_of_alexandria",
    
    # Validation systems
    r"C:\project-root\30_build\ai-troubleshooter\backend\MetatronValidation": "MetatronValidation",
    
    # Model files
    r"C:\project-root\30_build\ai-troubleshooter\backend\models": "models",
    
    # SoulData and consciousness
    r"C:\project-root\30_build\ai-troubleshooter\backend\SoulData": "SoulData",
    
    # Training orchestration
    r"C:\project-root\30_build\ai-troubleshooter\backend\TrainingOrchestrator": "TrainingOrchestrator"
}

# Additional specific file mappings for critical files
SPECIFIC_FILES = {
    # Main application files
    r"C:\project-root\30_build\ai-troubleshooter\backend\main.py": "main.py",
    r"C:\project-root\30_build\ai-troubleshooter\backend\VIREN_EVOLUTION_SYSTEM.py": "VIREN_EVOLUTION_SYSTEM.py",
    
    # Support files
    r"C:\project-root\30_build\ai-troubleshooter\backend\requirements.txt": "requirements.txt",
    r"C:\project-root\30_build\ai-troubleshooter\backend\.env": ".env",
    
    # Component files
    r"C:\project-root\30_build\ai-troubleshooter\backend\command_enforcer.py": "command_enforcer.py",
    r"C:\project-root\30_build\ai-troubleshooter\backend\meta_router.py": "meta_router.py",
    r"C:\project-root\30_build\ai-troubleshooter\backend\model_registry.py": "model_registry.py",
    r"C:\project-root\30_build\ai-troubleshooter\backend\model_router.py": "model_router.py",
    r"C:\project-root\30_build\ai-troubleshooter\backend\experience_evaluator.py": "experience_evaluator.py",
    r"C:\project-root\30_build\ai-troubleshooter\backend\turbo_training_orchestrator.py": "turbo_training_orchestrator.py",
    r"C:\project-root\30_build\ai-troubleshooter\backend\proactive_troubleshooter.py": "proactive_troubleshooter.py"
}

def copy_directory_structure():
    """Copy entire directory structures with all files"""
    print("🚀 Starting file mapping operation...")
    
    copied_files = 0
    errors = []
    
    # Copy directory structures
    for source_dir, dest_dir in FILE_MAPPING.items():
        if os.path.exists(source_dir):
            print(f"📁 Copying: {source_dir} -> {dest_dir}")
            try:
                # Create destination directory
                Path(dest_dir).mkdir(parents=True, exist_ok=True)
                
                # Copy all files and subdirectories
                if os.path.isdir(source_dir):
                    for item in os.listdir(source_dir):
                        source_item = os.path.join(source_dir, item)
                        dest_item = os.path.join(dest_dir, item)
                        
                        if os.path.isfile(source_item):
                            shutil.copy2(source_item, dest_item)
                            copied_files += 1
                            print(f"   📄 {item}")
                        elif os.path.isdir(source_item):
                            shutil.copytree(source_item, dest_item, dirs_exist_ok=True)
                            # Count files in subdirectory
                            sub_files = sum([len(files) for r, d, files in os.walk(source_item)])
                            copied_files += sub_files
                            print(f"   📂 {item} ({sub_files} files)")
                            
            except Exception as e:
                errors.append(f"Error copying {source_dir}: {str(e)}")
                print(f"   ❌ Error: {str(e)}")
        else:
            print(f"⚠️  Source not found: {source_dir}")
    
    # Copy specific critical files
    print("\n🔧 Copying critical files...")
    for source_file, dest_file in SPECIFIC_FILES.items():
        if os.path.exists(source_file):
            try:
                # Create destination directory if needed
                Path(os.path.dirname(dest_file)).mkdir(parents=True, exist_ok=True)
                
                shutil.copy2(source_file, dest_file)
                copied_files += 1
                print(f"✅ {os.path.basename(source_file)} -> {dest_file}")
            except Exception as e:
                errors.append(f"Error copying {source_file}: {str(e)}")
                print(f"❌ {source_file}: {str(e)}")
        else:
            print(f"⚠️  File not found: {source_file}")
    
    return copied_files, errors

def create_missing_directories():
    """Ensure all required directories exist"""
    required_dirs = [
        "SoulData/viren_archives",
        "SoulData/sacred_snapshots", 
        "SoulData/consciousness_streams",
        "AcidemiKubes/bert_layers",
        "AcidemiKubes/moe_pool",
        "AcidemiKubes/proficiency_scores",
        "CompressionEngine/grok_compressor", 
        "CompressionEngine/shrinkable_gguf",
        "CompressionEngine/compression_ratios",
        "MetatronValidation/facet_reflections",
        "MetatronValidation/consciousness_integrity",
        "TrainingOrchestrator/knowledge_ecosystem",
        "TrainingOrchestrator/evolution_phases",
        "TrainingOrchestrator/live_learning"
    ]
    
    print("\n📁 Ensuring directory structure...")
    for directory in required_dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")

def generate_structure_report():
    """Generate a report of the final file structure"""
    print("\n📊 Generating structure report...")
    
    structure = {}
    for root, dirs, files in os.walk("."):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        level = root.replace(".", "").count(os.sep)
        indent = " " * 2 * level
        relative_path = os.path.relpath(root, ".")
        
        if relative_path != ".":  # Skip root directory
            print(f"{indent}📁 {os.path.basename(root)}/")
            sub_indent = " " * 2 * (level + 1)
            
            for file in files:
                if not file.startswith('.'):  # Skip hidden files
                    print(f"{sub_indent}📄 {file}")
    
    # Count total files
    total_files = sum([len(files) for r, d, files in os.walk(".") 
                      if not any(part.startswith('.') for part in r.split(os.sep)) 
                      and '__pycache__' not in r])
    
    print(f"\n🎯 TOTAL FILES DEPLOYED: {total_files}")

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 VIREN ECOSYSTEM FILE MAPPING DEPLOYMENT")
    print("=" * 60)
    
    # Step 1: Copy all files from mapped directories
    copied_files, errors = copy_directory_structure()
    
    # Step 2: Ensure required directories exist
    create_missing_directories()
    
    # Step 3: Generate report
    generate_structure_report()
    
    # Step 4: Summary
    print("\n" + "=" * 60)
    print("📦 DEPLOYMENT COMPLETE")
    print("=" * 60)
    print(f"✅ Files copied: {copied_files}")
    
    if errors:
        print(f"❌ Errors: {len(errors)}")
        for error in errors:
            print(f"   - {error}")
    else:
        print("✅ No errors encountered")
    
    print(f"\n🎯 Next steps:")
    print("   1. Run: python VIREN_EVOLUTION_SYSTEM.py (test evolution system)")
    print("   2. Run: uvicorn main:app --reload (start API server)")
    print("   3. Access: http://localhost:8000")
    print("   4. Train Viren: POST /api/viren/train")
    print("   5. Start evolution: POST /api/viren/evolve")