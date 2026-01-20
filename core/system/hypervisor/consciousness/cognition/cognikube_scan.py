# cognikube_scanner.py
import os
from pathlib import Path
import json

def scan_cognikube_structure(root_path="."):
    """Map your exact CogniKube file structure"""
    
    print("🔍 SCANNING COGNIKUBE ARCHITECTURE...")
    print("=" * 60)
    
    structure = {
        "core_controllers": [],
        "quantum_routers": [], 
        "guardian_systems": [],
        "neural_networks": [],
        "deployment_configs": [],
        "evidence_logs": []
    }
    
    # SCAN FOR KEY COGNIKUBE COMPONENTS
    for file_path in Path(root_path).rglob("*"):
        if file_path.is_file():
            path_str = str(file_path)
            
            # CORE CONTROLLERS
            if any(keyword in path_str.lower() for keyword in [
                "cognikube", "supermesh", "master", "controller", "orchestrator"
            ]):
                structure["core_controllers"].append(path_str)
            
            # QUANTUM ROUTERS (Your Metatron Ulam system)
            elif any(keyword in path_str.lower() for keyword in [
                "metatron", "ulam", "quantum", "router", "spiral", "fibonacci", "prime"
            ]):
                structure["quantum_routers"].append(path_str)
            
            # GUARDIAN SYSTEMS
            elif any(keyword in path_str.lower() for keyword in [
                "guardian", "gateway", "firewall", "security", "webport", "mcp"
            ]):
                structure["guardian_systems"].append(path_str)
            
            # NEURAL NETWORKS
            elif any(keyword in path_str.lower() for keyword in [
                "compactifai", "train", "model", "neural", "network", "tensor", "mpo"
            ]):
                structure["neural_networks"].append(path_str)
            
            # DEPLOYMENT CONFIGS
            elif any(keyword in path_str.lower() for keyword in [
                "modal", "docker", "deploy", "container", "kubernetes", "config"
            ]):
                structure["deployment_configs"].append(path_str)
            
            # EVIDENCE LOGS
            elif any(keyword in path_str.lower() for keyword in [
                "evidence", "log", "test", "validation", "inventory"
            ]):
                structure["evidence_logs"].append(path_str)
    
    return structure

def print_cognikube_tree(structure):
    """Display your CogniKube architecture"""
    
    print("🌲 COGNIKUBE ARCHITECTURE TREE")
    print("=" * 60)
    
    for category, files in structure.items():
        print(f"\n📁 {category.upper().replace('_', ' ')}:")
        for file in sorted(files)[:10]:  # Show first 10 files per category
            print(f"   📄 {os.path.basename(file)}")
            print(f"      └── {file}")
        
        if len(files) > 10:
            print(f"   ... and {len(files) - 10} more files")

if __name__ == "__main__":
    # Scan current CogniKube environment
    cognikube_map = scan_cognikube_structure()
    print_cognikube_tree(cognikube_map)
    
    # Save for reference
    with open("cognikube_structure.json", "w") as f:
        json.dump(cognikube_map, f, indent=2)
    
    print(f"\n💾 Structure saved to: cognikube_structure.json")